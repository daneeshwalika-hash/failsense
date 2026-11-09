"""The Streamlit page, run headless with streamlit.testing.AppTest (API and data are faked)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from failsense.dashboard import data
from failsense.dashboard.client import ApiClient, ApiError
from failsense.data.schema import SPEC_COLUMNS

APP = Path(__file__).resolve().parents[2] / "src" / "failsense" / "dashboard" / "app.py"


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch, mini_frames: dict[str, pd.DataFrame]) -> dict:
    """Patch data loading and the API client; returns the settings the test may change."""
    st.cache_data.clear()
    state: dict[str, Any] = {"threshold": 0.05, "calls": [], "health_error": None,
                             "predict_error": None}  # fmt: skip
    readouts = mini_frames["readouts"].query("vehicle_id < 5")
    specs = mini_frames["specs"].query("vehicle_id < 5")
    label = pd.Series([0, 1, 2, 3, 4], index=pd.Index(range(5), name="vehicle_id"))
    monkeypatch.setattr(data, "load_demo_trucks", lambda config: (readouts, specs, label))

    def health(self: ApiClient) -> dict[str, str]:
        if state["health_error"]:
            raise ApiError(state["health_error"])
        return {"status": "ok", "model_name": "failsense-q1", "model_version": "7",
                "model_alias": "candidate"}  # fmt: skip

    def predict(self: ApiClient, payload: dict[str, Any]) -> dict[str, Any]:
        state["calls"].append(payload)
        if state["predict_error"]:
            raise ApiError(state["predict_error"], 422)
        risk = payload["current"]["time_step"] / 1000  # 0.11 at the last readout
        return {"risk": risk, "alert": risk >= state["threshold"], "threshold": state["threshold"],
                "model_name": "failsense-q1", "model_version": "7"}  # fmt: skip

    def latest(self: ApiClient, limit: int = 20) -> list[dict[str, Any]]:
        state["latest_limit"] = limit
        if state.get("latest_error"):
            raise ApiError(state["latest_error"])
        return state.get("latest_rows", [])

    monkeypatch.setattr(ApiClient, "latest_predictions", latest)
    monkeypatch.setattr(ApiClient, "health", health)
    monkeypatch.setattr(ApiClient, "predict", predict)
    return state


def run() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=30).run()


def test_page_shows_model_risk_threshold_and_alert(fake_backend: dict) -> None:
    at = run()
    assert not at.exception
    assert "failsense-q1 v7" in at.sidebar.success[0].value
    values = {m.label: m.value for m in at.metric}
    assert values["Calibrated risk"] == "11.0%" and values["Alert threshold"] == "5.00%"
    assert values["Risk / threshold"] == "2.2 x"
    assert "ALERT" in at.error[0].value


def test_no_alert_below_the_threshold(fake_backend: dict) -> None:
    fake_backend["threshold"] = 0.5
    at = run()
    assert not at.exception and not at.error
    assert any("No alert" in s.value for s in at.success)


def test_request_sent_to_the_api_has_the_expected_shape(fake_backend: dict) -> None:
    run()
    body = fake_backend["calls"][0]
    assert set(body) == {"current", "previous", "specifications", "vehicle_id"}
    assert set(body["specifications"]) == set(SPEC_COLUMNS)
    assert body["previous"]["time_step"] < body["current"]["time_step"]
    assert not {"class_label", "proximity_class"} & set(body["current"])  # no label is sent


def test_previous_readout_can_be_switched_off(fake_backend: dict) -> None:
    at = run()
    assert "previous" in fake_backend["calls"][0]
    first_run_calls = len(fake_backend["calls"])
    at.sidebar.checkbox[0].uncheck().run()
    # The first call of the second run is the main prediction; the trajectory calls that follow
    # always send each readout's predecessor.
    assert "previous" not in fake_backend["calls"][first_run_calls]


def test_unreachable_api_shows_an_error_and_does_not_crash(fake_backend: dict) -> None:
    fake_backend["health_error"] = "API not reachable at http://127.0.0.1:8000"
    at = run()
    assert not at.exception
    assert "make serve" in at.error[0].value
    assert fake_backend["calls"] == []  # nothing is scored without a healthy API


def test_api_rejection_is_shown_to_the_user(fake_backend: dict) -> None:
    fake_backend["predict_error"] = "POST /predict -> HTTP 422: body.current: missing column"
    at = run()
    assert not at.exception
    assert any("HTTP 422" in e.value for e in at.error)


def logged(i: int, alert: bool) -> dict[str, Any]:
    return {"id": i, "logged_at": f"2026-11-09T10:00:0{i}", "vehicle_id": 100 + i,
            "time_step": 50.0 + i, "has_previous": True, "risk": 0.02 + 0.04 * alert,
            "alert": alert, "threshold": 0.0353, "model_name": "failsense-q1",
            "model_version": "7"}  # fmt: skip


def test_latest_predictions_table_shows_rows_and_summary(fake_backend: dict) -> None:
    fake_backend["latest_rows"] = [
        logged(1, True),
        logged(2, False),
        logged(3, False),
        logged(4, True),
    ]
    at = run()
    assert not at.exception
    values = {m.label: m.value for m in at.metric}
    assert values["Alerts in the last 4 predictions"] == "50%"
    assert values["Mean risk"] == "4.0%"
    assert len(at.dataframe[0].value) == 4
    assert fake_backend["latest_limit"] == 15  # dashboard.latest_rows


def test_empty_log_and_unavailable_log_are_explained_not_errors(fake_backend: dict) -> None:
    at = run()
    assert any("No predictions logged yet" in i.value for i in at.info)
    fake_backend["latest_error"] = "GET /predictions/latest -> HTTP 503: prediction log is disabled"
    at = run()
    assert not at.exception
    assert any("not available" in i.value for i in at.info)


def test_rerun_does_not_resend_the_same_requests(fake_backend: dict) -> None:
    """A click that changes nothing about the scored readout must not flood the prediction log."""
    at = run()
    n = len(fake_backend["calls"])
    at.run()  # same truck, readout and switch -> served from the cache
    assert len(fake_backend["calls"]) == n
