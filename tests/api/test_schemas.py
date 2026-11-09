"""Request validation: every violation of the data contract must be a ValidationError (422)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from failsense.api.schemas import PredictRequest

Payload = Callable[..., dict[str, Any]]


def test_valid_request_with_and_without_previous(make_payload: Payload) -> None:
    assert PredictRequest(**make_payload()).previous is not None
    assert PredictRequest(**make_payload(with_previous=False)).previous is None


def test_null_values_are_allowed_everywhere(make_payload: Payload) -> None:
    body = make_payload()
    body["current"]["167_0"] = None
    body["previous"]["171_0"] = None
    PredictRequest(**body)


def test_missing_column_is_rejected(make_payload: Payload) -> None:
    body = make_payload()
    del body["current"]["167_3"]
    with pytest.raises(ValidationError, match="missing column"):
        PredictRequest(**body)


def test_missing_time_step_and_null_time_step_are_rejected(make_payload: Payload) -> None:
    body = make_payload()
    body["current"]["time_step"] = None
    with pytest.raises(ValidationError, match="must not be null"):
        PredictRequest(**body)
    body = make_payload()
    del body["current"]["time_step"]
    with pytest.raises(ValidationError, match="missing column"):
        PredictRequest(**body)


@pytest.mark.parametrize("bad", ["5", True, [1.0], {"a": 1}])
def test_wrong_dtype_is_rejected(make_payload: Payload, bad: Any) -> None:
    body = make_payload()
    body["current"]["171_0"] = bad
    with pytest.raises(ValidationError):
        PredictRequest(**body)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_negative_or_non_finite_values_are_rejected(make_payload: Payload, bad: float) -> None:
    for where in ("current", "previous"):
        body = make_payload()
        body[where]["171_0"] = bad
        with pytest.raises(ValidationError):
            PredictRequest(**body)


@pytest.mark.parametrize(
    "label",
    ["class_label", "in_study_repair", "length_of_study_time_step", "repair_within_horizon"],
)
def test_label_columns_are_rejected_everywhere(make_payload: Payload, label: str) -> None:
    body = make_payload()
    body["current"][label] = 1
    with pytest.raises(ValidationError, match="label or identifier"):
        PredictRequest(**body)
    body = make_payload()
    body["previous"][label] = 1
    with pytest.raises(ValidationError, match="label or identifier"):
        PredictRequest(**body)
    with pytest.raises(ValidationError):  # top level: extra="forbid"
        PredictRequest(**make_payload(**{label: 1}))


def test_vehicle_id_inside_a_readout_is_rejected(make_payload: Payload) -> None:
    body = make_payload()
    body["current"]["vehicle_id"] = 7
    with pytest.raises(ValidationError, match="top-level"):
        PredictRequest(**body)


def test_unknown_column_is_rejected(make_payload: Payload) -> None:
    body = make_payload()
    body["current"]["999_9"] = 1.0
    with pytest.raises(ValidationError, match="unknown column"):
        PredictRequest(**body)


def test_previous_takes_only_counters(make_payload: Payload) -> None:
    body = make_payload()
    body["previous"]["167_0"] = 1.0  # histogram bins of the previous readout are not needed
    with pytest.raises(ValidationError, match="unknown column"):
        PredictRequest(**body)


@pytest.mark.parametrize("previous_time", [100.0, 150.0])
def test_previous_must_be_strictly_earlier(make_payload: Payload, previous_time: float) -> None:
    body = make_payload()
    body["previous"]["time_step"] = previous_time
    with pytest.raises(ValidationError, match="strictly smaller"):
        PredictRequest(**body)


def test_specifications_must_be_the_eight_strings(make_payload: Payload) -> None:
    body = make_payload()
    body["specifications"]["Spec_3"] = 5
    with pytest.raises(ValidationError):
        PredictRequest(**body)
    body = make_payload()
    del body["specifications"]["Spec_3"]
    with pytest.raises(ValidationError, match="missing column"):
        PredictRequest(**body)


@pytest.mark.parametrize("vehicle_id", [-1, "7", 1.5])
def test_vehicle_id_must_be_a_non_negative_int(make_payload: Payload, vehicle_id: Any) -> None:
    with pytest.raises(ValidationError):
        PredictRequest(**make_payload(vehicle_id=vehicle_id))
