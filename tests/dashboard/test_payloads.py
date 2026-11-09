"""Payload builder: bodies must satisfy the API's own schema (the contract is shared)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from failsense.api.schemas import PredictRequest
from failsense.dashboard.payloads import build_payload, risk_history, sorted_readouts
from failsense.data.schema import SPEC_COLUMNS

SPECS = {c: "Cat1" for c in SPEC_COLUMNS}


def truck(mini_frames: dict[str, pd.DataFrame], vehicle: int = 3) -> pd.DataFrame:
    return mini_frames["readouts"].query("vehicle_id == @vehicle")


def test_payload_validates_against_the_api_schema(mini_frames: dict[str, pd.DataFrame]) -> None:
    body = build_payload(truck(mini_frames), SPECS, vehicle_id=3)
    request = PredictRequest(**json.loads(json.dumps(body)))  # also JSON-serialisable
    assert request.current["time_step"] == 110.0 and request.previous["time_step"] == 100.0
    assert request.vehicle_id == 3


def test_previous_is_the_immediately_preceding_readout_even_if_shuffled(
    mini_frames: dict[str, pd.DataFrame],
) -> None:
    shuffled = truck(mini_frames).sample(frac=1, random_state=0)
    body = build_payload(shuffled, SPECS, position=5)
    assert (body["current"]["time_step"], body["previous"]["time_step"]) == (50.0, 40.0)


def test_first_readout_and_opt_out_have_no_previous(mini_frames: dict[str, pd.DataFrame]) -> None:
    assert "previous" not in build_payload(truck(mini_frames), SPECS, position=0)
    assert "previous" not in build_payload(truck(mini_frames), SPECS, include_previous=False)
    assert "vehicle_id" not in build_payload(truck(mini_frames), SPECS)


def test_nan_becomes_null(mini_frames: dict[str, pd.DataFrame]) -> None:
    frame = truck(mini_frames).copy()
    frame.loc[frame.index[-1], "167_0"] = np.nan
    frame.loc[frame.index[-2], "171_0"] = np.nan
    body = build_payload(frame, SPECS)
    assert body["current"]["167_0"] is None and body["previous"]["171_0"] is None
    PredictRequest(**body)


def test_bad_inputs_raise(mini_frames: dict[str, pd.DataFrame]) -> None:
    frame = truck(mini_frames)
    with pytest.raises(ValueError, match="outside"):
        build_payload(frame, SPECS, position=99)
    with pytest.raises(ValueError, match="no readouts"):
        build_payload(frame.iloc[0:0], SPECS)
    with pytest.raises(ValueError, match="unique"):
        sorted_readouts(pd.concat([frame, frame.iloc[[0]]]))
    with pytest.raises(ValueError, match="missing columns"):
        build_payload(frame.drop(columns=["167_0"]), SPECS)
    with pytest.raises(KeyError):
        build_payload(frame, {"Spec_0": "Cat1"})


class RecordingClient:
    """Stands in for ApiClient: risk grows with time_step; records the bodies."""

    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []

    def predict(self, body: dict[str, Any]) -> dict[str, Any]:
        self.bodies.append(body)
        risk = body["current"]["time_step"] / 1000
        return {"risk": risk, "threshold": 0.05, "alert": risk >= 0.05}


def test_risk_history_scores_the_latest_readouts_in_time_order(
    mini_frames: dict[str, pd.DataFrame],
) -> None:
    client = RecordingClient()
    history = risk_history(client, truck(mini_frames), SPECS, n_latest=4, vehicle_id=3)  # type: ignore[arg-type]
    assert history["time_step"].tolist() == [80.0, 90.0, 100.0, 110.0]
    assert history["alert"].tolist() == [True] * 4
    assert all("previous" in b for b in client.bodies)
    assert risk_history(client, truck(mini_frames), SPECS, n_latest=99).shape[0] == 12  # type: ignore[arg-type]
