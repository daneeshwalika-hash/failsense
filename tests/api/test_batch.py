"""/predict/batch: same answers as single calls, order, limits, logging, validation."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from failsense.api.main import create_app, get_service
from failsense.api.prediction_log import PredictionLog
from failsense.api.schemas import MAX_BATCH_SIZE, BatchRequest
from failsense.api.service import ModelService

Payload = Callable[..., dict[str, Any]]


def varied(make_payload: Payload, n: int) -> list[dict[str, Any]]:
    """n different valid requests (the counters grow with i; every third has no previous)."""
    items = []
    for i in range(n):
        body = make_payload(with_previous=i % 3 != 0, vehicle_id=100 + i)
        body["current"]["171_0"] += 50.0 * i
        body["current"]["167_0"] += 7.0 * i
        body["current"]["time_step"] += i
        if "previous" in body:
            body["previous"]["time_step"] = body["current"]["time_step"] - 10
        items.append(body)
    return items


@pytest.fixture
def client(service: ModelService) -> TestClient:
    app = create_app(loader=lambda: service)
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app)


# --- service ----------------------------------------------------------------------------------
def test_batch_equals_single_predictions(service: ModelService, make_payload: Payload) -> None:
    items = varied(make_payload, 8)
    batch = service.predict_batch(
        [(i["current"], i.get("previous"), i["specifications"]) for i in items]
    )
    singles = [service.predict(i["current"], i.get("previous"), i["specifications"]) for i in items]
    assert len(batch) == 8
    for b, s in zip(batch, singles, strict=True):
        assert b.risk == pytest.approx(s.risk, rel=1e-6) and b.alert == s.alert
        assert b.score == pytest.approx(s.score, rel=1e-6)
    assert len({round(b.risk, 6) for b in batch}) > 1  # the items really differ


def test_batch_rejects_an_empty_list(service: ModelService) -> None:
    with pytest.raises(ValueError, match="empty"):
        service.predict_batch([])


# --- schema -----------------------------------------------------------------------------------
def test_batch_size_limits(make_payload: Payload) -> None:
    with pytest.raises(ValidationError):
        BatchRequest(items=[])
    BatchRequest(items=[make_payload()] * MAX_BATCH_SIZE)
    with pytest.raises(ValidationError):
        BatchRequest(items=[make_payload()] * (MAX_BATCH_SIZE + 1))
    with pytest.raises(ValidationError):
        BatchRequest(items=[make_payload()], extra=1)  # type: ignore[call-arg]


# --- endpoint ---------------------------------------------------------------------------------
def test_endpoint_returns_one_answer_per_item_in_order(
    client: TestClient, make_payload: Payload
) -> None:
    items = varied(make_payload, 6)
    response = client.post("/predict/batch", json={"items": items})
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 6
    for item, result in zip(items, results, strict=True):
        single = client.post("/predict", json=item).json()
        assert result == pytest.approx(single) or result["risk"] == pytest.approx(single["risk"])
        assert set(result) == {"risk", "alert", "threshold", "model_name", "model_version"}


def test_one_invalid_item_rejects_the_whole_request_and_names_the_index(
    client: TestClient, make_payload: Payload
) -> None:
    items = varied(make_payload, 4)
    bad = copy.deepcopy(items[2])
    bad["current"]["class_label"] = 1
    items[2] = bad
    response = client.post("/predict/batch", json={"items": items})
    assert response.status_code == 422
    assert (
        "items" in response.text and "2" in response.text and "label or identifier" in response.text
    )


@pytest.mark.parametrize("body", [{"items": []}, {}, {"items": "x"}, {"things": []}])
def test_endpoint_rejects_malformed_bodies(client: TestClient, body: dict[str, Any]) -> None:
    assert client.post("/predict/batch", json=body).status_code == 422


def test_too_many_items_are_rejected(client: TestClient, make_payload: Payload) -> None:
    items = [make_payload()] * (MAX_BATCH_SIZE + 1)
    assert client.post("/predict/batch", json={"items": items}).status_code == 422


def test_every_item_is_written_to_the_prediction_log(
    service: ModelService, make_payload: Payload
) -> None:
    log = PredictionLog(":memory:")
    app = create_app(loader=lambda: service)
    app.state.prediction_log = log
    app.dependency_overrides[get_service] = lambda: service
    items = varied(make_payload, 5)
    TestClient(app).post("/predict/batch", json={"items": items})
    rows = log.latest(10)
    assert len(rows) == 5
    assert [r["vehicle_id"] for r in rows] == [104, 103, 102, 101, 100]  # newest first


def test_batch_is_503_without_a_model(service: ModelService, make_payload: Payload) -> None:
    bare = TestClient(create_app(loader=lambda: service))
    assert bare.post("/predict/batch", json={"items": [make_payload()]}).status_code == 503
