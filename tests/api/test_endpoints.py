"""HTTP tests with TestClient and the tiny pipeline (dependency override, no MLflow)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from failsense.api.main import create_app, get_service
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
