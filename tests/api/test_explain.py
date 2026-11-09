"""/explain: SHAP factors in calibrated log-odds, additivity, validation, no logging."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from failsense.api.main import create_app, get_service
from failsense.api.prediction_log import PredictionLog
from failsense.api.schemas import ExplainRequest
from failsense.api.service import ModelService

Payload = Callable[..., dict[str, Any]]


def logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


@pytest.fixture
def client(service: ModelService) -> TestClient:
    app = create_app(loader=lambda: service)
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app)


# --- service ----------------------------------------------------------------------------------
def test_factors_plus_baseline_add_up_to_the_calibrated_log_odds(
    service: ModelService, make_payload: Payload
) -> None:
    """The core promise: logit(risk) = logit(baseline) + sum of ALL contributions."""
    body = make_payload()
    result = service.explain(body["current"], body["previous"], body["specifications"],
                             k=len(service.input_columns))  # fmt: skip
    total = sum(f.contribution for f in result.factors)
    assert len(result.factors) == len(service.input_columns)
    assert logit(result.prediction.risk) == pytest.approx(
        logit(result.baseline_risk) + total, abs=1e-3
    )


def test_factors_are_sorted_by_absolute_contribution_and_limited_to_k(
    service: ModelService, make_payload: Payload
) -> None:
    body = make_payload()
    result = service.explain(body["current"], body["previous"], body["specifications"], k=3)
    sizes = [abs(f.contribution) for f in result.factors]
    assert len(result.factors) == 3 and sizes == sorted(sizes, reverse=True)
    names = {f.feature for f in result.factors}
    assert names <= set(service.input_columns)


def test_explanation_prediction_equals_the_predict_result(
    service: ModelService, make_payload: Payload
) -> None:
    body = make_payload()
    args = (body["current"], body["previous"], body["specifications"])
    assert service.explain(*args).prediction == service.predict(*args)


def test_values_are_json_safe_nan_becomes_none_and_specs_stay_text(
    service: ModelService, make_payload: Payload
) -> None:
    body = make_payload(with_previous=False)  # the deltas are NaN
    result = service.explain(body["current"], None, body["specifications"],
                             k=len(service.input_columns))  # fmt: skip
    by_name = {f.feature: f.value for f in result.factors}
    assert by_name["171_0__delta"] is None
    assert by_name["Spec_0"] == "Cat1" and isinstance(by_name["171_0"], float)


def test_k_below_one_is_rejected(service: ModelService, make_payload: Payload) -> None:
    body = make_payload()
    with pytest.raises(ValueError, match="k must be"):
        service.explain(body["current"], body["previous"], body["specifications"], k=0)


def test_the_explainer_is_built_once(service: ModelService) -> None:
    assert service.explainer is service.explainer


# --- schema -----------------------------------------------------------------------------------
@pytest.mark.parametrize("k", [0, -1, 21, "5", 2.5])
def test_top_k_must_be_an_int_between_1_and_20(make_payload: Payload, k: Any) -> None:
    with pytest.raises(ValidationError):
        ExplainRequest(**make_payload(top_k=k))


def test_explain_request_keeps_every_predict_rule(make_payload: Payload) -> None:
    assert ExplainRequest(**make_payload()).top_k == 5
    body = make_payload()
    body["current"]["class_label"] = 1
    with pytest.raises(ValidationError, match="label or identifier"):
        ExplainRequest(**body)
    body = make_payload()
    body["previous"]["time_step"] = 100.0
    with pytest.raises(ValidationError, match="strictly smaller"):
        ExplainRequest(**body)


# --- endpoint ---------------------------------------------------------------------------------
def test_explain_endpoint_returns_risk_factors_and_baseline(
    client: TestClient, service: ModelService, make_payload: Payload
) -> None:
    body = make_payload(top_k=4)
    response = client.post("/explain", json=body)
    assert response.status_code == 200
    data = response.json()
    assert set(data) == {"risk", "alert", "threshold", "model_name", "model_version",
                         "baseline_risk", "factors"}  # fmt: skip
    assert "score" not in data
    assert len(data["factors"]) == 4
    for factor in data["factors"]:
        assert set(factor) == {"feature", "value", "contribution", "direction"}
        assert factor["direction"] == ("raises" if factor["contribution"] > 0 else "lowers")
    predict = client.post("/predict", json=make_payload()).json()
    assert data["risk"] == predict["risk"] and data["alert"] == predict["alert"]
    assert 0 < data["baseline_risk"] < 1 and data["threshold"] == service.threshold


def test_explain_endpoint_defaults_to_five_factors(
    client: TestClient, make_payload: Payload
) -> None:
    assert len(client.post("/explain", json=make_payload()).json()["factors"]) == 5


@pytest.mark.parametrize("top_k", [0, 21, "many"])
def test_explain_endpoint_rejects_a_bad_top_k(
    client: TestClient, make_payload: Payload, top_k: Any
) -> None:
    assert client.post("/explain", json=make_payload(top_k=top_k)).status_code == 422


def test_explain_endpoint_rejects_label_columns_and_503_without_a_model(
    client: TestClient, make_payload: Payload, service: ModelService
) -> None:
    body = make_payload()
    body["current"]["in_study_repair"] = 1
    assert client.post("/explain", json=body).status_code == 422
    bare = TestClient(create_app(loader=lambda: service))  # no override, no lifespan
    assert bare.post("/explain", json=make_payload()).status_code == 503


def test_explain_requests_are_not_written_to_the_prediction_log(
    service: ModelService, make_payload: Payload
) -> None:
    log = PredictionLog(":memory:")
    app = create_app(loader=lambda: service)
    app.state.prediction_log = log
    app.dependency_overrides[get_service] = lambda: service
    http = TestClient(app)
    assert http.post("/explain", json=make_payload()).status_code == 200
    assert log.count() == 0  # an explanation is an analysis, not a served prediction
    http.post("/predict", json=make_payload())
    assert log.count() == 1
