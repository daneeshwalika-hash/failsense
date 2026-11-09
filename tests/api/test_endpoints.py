"""HTTP tests with TestClient and the tiny pipeline (dependency override, no MLflow)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from failsense.api.main import create_app, get_service
from failsense.api.prediction_log import PredictionLog
from failsense.api.service import ModelConfigError, ModelService

Payload = Callable[..., dict[str, Any]]


@pytest.fixture
def client(service: ModelService) -> TestClient:
    """App whose model dependency is replaced; the lifespan is not started (no `with`)."""
    app = create_app(loader=lambda: service)
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app)


def test_health_reports_the_loaded_model(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_name": "failsense-q1",
        "model_version": "3",
        "model_alias": "candidate",
    }


def test_predict_returns_calibrated_risk_alert_threshold_and_version(
    client: TestClient, service: ModelService, make_payload: Payload
) -> None:
    body = make_payload()
    response = client.post("/predict", json=body)
    assert response.status_code == 200
    data = response.json()
    assert set(data) == {"risk", "alert", "threshold", "model_name", "model_version"}
    assert "score" not in data  # the raw score is never exposed as a probability
    assert 0 <= data["risk"] <= 1
    assert data["threshold"] == service.threshold
    assert data["alert"] == (data["risk"] >= data["threshold"])
    assert (data["model_name"], data["model_version"]) == ("failsense-q1", "3")
    expected = service.predict(body["current"], body["previous"], body["specifications"])
    assert data["risk"] == pytest.approx(expected.risk)


def test_predict_without_previous_works_and_differs_from_with_previous(
    client: TestClient, make_payload: Payload
) -> None:
    without = client.post("/predict", json=make_payload(with_previous=False))
    with_prev = client.post("/predict", json=make_payload())
    assert without.status_code == with_prev.status_code == 200
    assert without.json()["risk"] != with_prev.json()["risk"]  # the delta feature matters


def test_vehicle_id_is_optional_and_does_not_change_the_risk(
    client: TestClient, make_payload: Payload
) -> None:
    a = client.post("/predict", json=make_payload(vehicle_id=1)).json()
    b = client.post("/predict", json=make_payload(vehicle_id=None)).json()
    c = client.post("/predict", json=make_payload(with_previous=True, vehicle_id=99)).json()
    assert a["risk"] == b["risk"] == c["risk"]


@pytest.mark.parametrize("previous_time", [100.0, 101.0])
def test_previous_not_earlier_than_current_is_422(
    client: TestClient, make_payload: Payload, previous_time: float
) -> None:
    body = make_payload()
    body["previous"]["time_step"] = previous_time
    response = client.post("/predict", json=body)
    assert response.status_code == 422
    assert "strictly smaller" in response.text


def test_label_column_in_body_is_422(client: TestClient, make_payload: Payload) -> None:
    body = make_payload()
    body["current"]["class_label"] = 3
    response = client.post("/predict", json=body)
    assert response.status_code == 422
    assert "label or identifier" in response.text


def test_wrong_dtype_is_422(client: TestClient, make_payload: Payload) -> None:
    body = make_payload()
    body["current"]["171_0"] = "lots"
    assert client.post("/predict", json=body).status_code == 422


def test_negative_counter_is_422(client: TestClient, make_payload: Payload) -> None:
    body = make_payload()
    body["current"]["171_0"] = -5
    assert client.post("/predict", json=body).status_code == 422


def test_missing_column_and_missing_specifications_are_422(
    client: TestClient, make_payload: Payload
) -> None:
    body = make_payload()
    del body["current"]["397_12"]
    assert client.post("/predict", json=body).status_code == 422
    body = make_payload()
    del body["specifications"]
    assert client.post("/predict", json=body).status_code == 422


def test_empty_and_non_json_body_are_422(client: TestClient) -> None:
    assert client.post("/predict", json={}).status_code == 422
    assert client.post("/predict", content=b"not json").status_code == 422


def test_503_when_no_model_is_loaded(make_payload: Payload, service: ModelService) -> None:
    client = TestClient(create_app(loader=lambda: service))  # no lifespan, no override
    assert client.get("/health").status_code == 503
    assert client.post("/predict", json=make_payload()).status_code == 503


def test_lifespan_loads_the_model_once(service: ModelService, make_payload: Payload) -> None:
    calls: list[int] = []

    def loader() -> ModelService:
        calls.append(1)
        return service

    with TestClient(create_app(loader=loader)) as client:  # runs the lifespan
        assert client.get("/health").status_code == 200
        assert client.post("/predict", json=make_payload()).status_code == 200
        assert client.post("/predict", json=make_payload()).status_code == 200
    assert calls == [1]


def test_startup_fails_fast_when_the_model_config_is_wrong() -> None:
    def broken() -> ModelService:
        raise ModelConfigError("summary model_uri differs")

    with pytest.raises(ModelConfigError, match="differs"), TestClient(create_app(loader=broken)):
        pass


# --- prediction log -----------------------------------------------------------------------
@pytest.fixture
def logged_client(service: ModelService) -> tuple[TestClient, PredictionLog]:
    log = PredictionLog(":memory:")
    app = create_app(loader=lambda: service, log_factory=lambda: log)
    app.state.prediction_log = log  # no lifespan in this fixture
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app), log


def test_predict_writes_one_log_row_with_the_answer(
    logged_client: tuple[TestClient, PredictionLog], make_payload: Payload
) -> None:
    client, log = logged_client
    answer = client.post("/predict", json=make_payload(vehicle_id=42)).json()
    client.post("/predict", json=make_payload(with_previous=False))
    assert log.count() == 2
    newest, oldest = log.latest(2)
    assert oldest["vehicle_id"] == 42 and oldest["has_previous"] is True
    assert oldest["risk"] == answer["risk"] and oldest["alert"] == answer["alert"]
    assert oldest["threshold"] == answer["threshold"] and oldest["model_version"] == "3"
    assert newest["has_previous"] is False and newest["time_step"] == 100.0


def test_rejected_requests_are_not_logged(
    logged_client: tuple[TestClient, PredictionLog], make_payload: Payload
) -> None:
    client, log = logged_client
    body = make_payload()
    body["current"]["class_label"] = 1
    assert client.post("/predict", json=body).status_code == 422
    assert log.count() == 0


def test_latest_endpoint_returns_rows_newest_first_and_validates_limit(
    logged_client: tuple[TestClient, PredictionLog], make_payload: Payload
) -> None:
    client, _ = logged_client
    for vehicle in (1, 2, 3):
        client.post("/predict", json=make_payload(vehicle_id=vehicle))
    rows = client.get("/predictions/latest", params={"limit": 2}).json()
    assert [r["vehicle_id"] for r in rows] == [3, 2]
    assert set(rows[0]) >= {"id", "logged_at", "risk", "alert", "threshold", "model_version"}
    assert "score" not in rows[0]
    assert len(client.get("/predictions/latest").json()) == 3  # default limit 20
    assert client.get("/predictions/latest", params={"limit": 0}).status_code == 422
    assert client.get("/predictions/latest", params={"limit": 5000}).status_code == 422


def test_latest_endpoint_is_503_when_the_log_is_disabled(client: TestClient) -> None:
    response = client.get("/predictions/latest")
    assert response.status_code == 503 and "disabled" in response.text


def test_a_failing_log_does_not_break_the_prediction(
    service: ModelService, make_payload: Payload, caplog: pytest.LogCaptureFixture
) -> None:
    class BrokenLog:
        def log(self, record: object) -> None:
            raise RuntimeError("disk full")

        def close(self) -> None:
            pass

    app = create_app(loader=lambda: service)
    app.state.prediction_log = BrokenLog()
    app.dependency_overrides[get_service] = lambda: service
    response = TestClient(app).post("/predict", json=make_payload())
    assert response.status_code == 200 and 0 <= response.json()["risk"] <= 1
    assert "could not write the prediction log" in caplog.text


def test_lifespan_opens_and_closes_the_log(service: ModelService, make_payload: Payload) -> None:
    log = PredictionLog(":memory:")
    closed: list[bool] = []
    original_close = log.close
    log.close = lambda: (closed.append(True), original_close())  # type: ignore[method-assign]
    with TestClient(create_app(loader=lambda: service, log_factory=lambda: log)) as client:
        assert client.post("/predict", json=make_payload()).status_code == 200
        assert len(client.get("/predictions/latest").json()) == 1
    assert closed == [True]
