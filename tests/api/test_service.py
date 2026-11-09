"""ModelService: feature parity with validation, calibration, alert rule, startup checks."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from failsense.api.service import (
    ModelConfigError,
    ModelInfo,
    ModelService,
    known_spec_categories,
    load_deployment,
)
from failsense.data.labels import TIME_COL, VEHICLE_COL
from failsense.data.schema import SPEC_COLUMNS, counter_columns, feature_columns, histogram_groups
from failsense.models.calibration import apply_sigmoid_parameters
from failsense.models.validation_data import build_last_readout_frame

Payload = Callable[..., dict[str, Any]]


def _full_input_columns() -> list[str]:
    """All columns the builder produces (what the real registry draws its 239 from)."""
    from failsense.features.build import build_model_features

    rows = pd.DataFrame(
        [
            {VEHICLE_COL: 0, TIME_COL: t, **{c: 1.0 + t for c in feature_columns()}}
            for t in (0.0, 10.0)
        ]
    )
    feats = build_model_features(rows, pd.Index([1]), counter_columns(), histogram_groups(), 0.2)
    return list(feats.columns)


def test_api_features_equal_the_validation_features(make_payload: Payload) -> None:
    """Same columns and values as build_last_readout_frame for the same two readouts."""
    body = make_payload()
    cols = [TIME_COL, *feature_columns()]
    readouts = pd.DataFrame(
        [
            {VEHICLE_COL: 5, **{c: body["previous"].get(c, np.nan) for c in cols}},
            {VEHICLE_COL: 5, **{c: body["current"][c] for c in cols}},
        ]
    )
    readouts[TIME_COL] = [body["previous"][TIME_COL], body["current"][TIME_COL]]
    specs = pd.DataFrame([{VEHICLE_COL: 5, **body["specifications"]}])
    labels = pd.DataFrame({VEHICLE_COL: [5], "class_label": [0]})
    validation = build_last_readout_frame(
        readouts, specs, labels, counter_columns(), histogram_groups(), 0.2
    )

    registered = [c for c in validation.columns if c not in (VEHICLE_COL, "class_label")]
    registered = [c for c in registered if c not in ("repair_within_horizon", "proximity_class")]
    calibrator = {"kind": "sigmoid", "slope": 1.0, "intercept": 0.0, "logit_clip_eps": 1e-6}
    # build_inputs needs no fitted pipeline
    svc = ModelService(None, calibrator, 0.1, registered, ModelInfo("m", "1", "a"), 0.2)
    api_row = svc.build_inputs(body["current"], body["previous"], body["specifications"])

    assert list(api_row.columns) == registered
    assert len(registered) == 114 + 97 + 16 + 12  # raw+spec+time_step, shares, deltas, hist agg
    pd.testing.assert_frame_equal(api_row, validation[registered], check_dtype=False)


def test_real_registry_columns_are_all_buildable() -> None:
    """The builder yields every column the registered model asks for (239 in the registry)."""
    from failsense.config import load_config, resolve_path

    path = resolve_path(load_config()["paths"]["reports"]) / "registry_q1.json"
    if not path.is_file():
        pytest.skip("reports/registry_q1.json not present")
    wanted = set(json.loads(path.read_text())["input_columns"])
    buildable = set(_full_input_columns()) | set(SPEC_COLUMNS)
    assert wanted <= buildable


def test_missing_previous_gives_nan_deltas(service: ModelService, make_payload: Payload) -> None:
    body = make_payload(with_previous=False)
    inputs = service.build_inputs(body["current"], None, body["specifications"])
    assert inputs["171_0__delta"].isna().all() and inputs["171_0__rate"].isna().all()
    assert service.predict(body["current"], None, body["specifications"]).risk >= 0


def test_previous_changes_the_delta(service: ModelService, make_payload: Payload) -> None:
    body = make_payload()
    inputs = service.build_inputs(body["current"], body["previous"], body["specifications"])
    assert inputs.loc[0, "171_0__delta"] == pytest.approx(10.0 - 8.0)
    assert inputs.loc[0, "171_0__rate"] == pytest.approx((10.0 - 8.0) / (100.0 - 90.0))


def test_counter_reset_keeps_the_negative_delta_and_logs(
    service: ModelService, make_payload: Payload, caplog: pytest.LogCaptureFixture
) -> None:
    body = make_payload()
    body["previous"]["171_0"] = 50.0  # larger than current 10.0
    with caplog.at_level(logging.WARNING):
        inputs = service.build_inputs(body["current"], body["previous"], body["specifications"])
    assert inputs.loc[0, "171_0__delta"] == pytest.approx(-40.0)
    assert "decreases" in caplog.text


def test_only_registered_columns_reach_the_model(
    service: ModelService, make_payload: Payload
) -> None:
    body = make_payload()
    inputs = service.build_inputs(body["current"], body["previous"], body["specifications"])
    assert list(inputs.columns) == service.input_columns
    assert VEHICLE_COL not in inputs.columns


def test_vehicle_id_does_not_change_the_prediction(
    service: ModelService, make_payload: Payload
) -> None:
    body = make_payload()
    args = (body["current"], body["previous"], body["specifications"])
    assert service.predict(*args, vehicle_id=1) == service.predict(*args, vehicle_id=99999)


def test_risk_is_the_calibrated_score_and_alert_follows_the_threshold(
    service: ModelService, make_payload: Payload
) -> None:
    body = make_payload()
    result = service.predict(body["current"], body["previous"], body["specifications"])
    expected = apply_sigmoid_parameters(service.calibrator, [result.score])[0]
    assert result.risk == pytest.approx(expected)
    assert result.risk != pytest.approx(result.score)  # the raw score is not the probability
    assert result.alert == (result.risk >= service.threshold)
    for threshold in (result.risk, result.risk + 1e-9, result.risk - 1e-9):
        service.threshold = min(max(threshold, 1e-6), 1 - 1e-6)
        again = service.predict(body["current"], body["previous"], body["specifications"])
        assert again.alert == (again.risk >= service.threshold)  # >= : equal counts as alert


def test_unseen_spec_category_is_logged_not_rejected(
    service: ModelService, make_payload: Payload, caplog: pytest.LogCaptureFixture
) -> None:
    body = make_payload()
    body["specifications"]["Spec_0"] = "Cat999"
    with caplog.at_level(logging.WARNING):
        service.predict(body["current"], body["previous"], body["specifications"])
    assert "not seen in training" in caplog.text


def test_known_spec_categories_reads_the_encoder(service: ModelService) -> None:
    assert known_spec_categories(service.pipeline)["Spec_0"] == {"A", "B", "C"}
    assert known_spec_categories(object()) == {}


def test_smoke_test_passes_with_buildable_columns_and_fails_otherwise(
    service: ModelService,
) -> None:
    # The tiny pipeline's columns exist in the builder's output, so the smoke test passes.
    service.smoke_test()
    service.input_columns = [*service.input_columns, "no_such_column"]
    with pytest.raises(ValueError, match="missing"):
        service.smoke_test()


def test_constructor_rejects_a_non_sigmoid_calibrator(
    tiny_pipeline: tuple[Any, Any, Any],
) -> None:
    pipeline, frame, _ = tiny_pipeline
    with pytest.raises(ModelConfigError, match="sigmoid"):
        ModelService(
            pipeline, {"kind": "isotonic"}, 0.1, list(frame.columns), ModelInfo("m", "1", "a"), 0.2
        )


@pytest.mark.parametrize("threshold", [0.0, 1.0, -0.1, 2.0])
def test_constructor_rejects_thresholds_outside_zero_one(
    tiny_pipeline: tuple[Any, Any, Any], threshold: float
) -> None:
    pipeline, frame, _ = tiny_pipeline
    with pytest.raises(ModelConfigError, match="threshold"):
        ModelService(
            pipeline,
            {"kind": "sigmoid", "slope": 1, "intercept": 0, "logit_clip_eps": 1e-6},
            threshold,
            list(frame.columns),
            ModelInfo("m", "1", "a"),
            0.2,
        )


# --- loading: summary file and URI check -------------------------------------------------
def _config(tmp_path: Path, block: dict[str, Any] | None) -> dict[str, Any]:
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({} if block is None else {"deployment_choice": block}))
    return {
        "api": {"summary_path": str(summary), "summary_key": "deployment_choice"},
        "tuning": {"registered_model": "failsense-q1", "alias": "candidate"},
    }


def test_load_deployment_reads_the_block(tmp_path: Path) -> None:
    block = {"model_uri": "models:/failsense-q1@candidate", "calibrator": {}, "threshold": 0.03}
    assert load_deployment(_config(tmp_path, block)) == block


def test_load_deployment_rejects_missing_file_and_incomplete_block(tmp_path: Path) -> None:
    config = _config(tmp_path, {"model_uri": "x"})
    with pytest.raises(ModelConfigError, match="needs"):
        load_deployment(config)
    config["api"]["summary_path"] = str(tmp_path / "nope.json")
    with pytest.raises(ModelConfigError, match="not found"):
        load_deployment(config)


def test_from_config_refuses_to_start_on_a_model_uri_mismatch(tmp_path: Path) -> None:
    block = {
        "model_uri": "models:/failsense-q1@champion",  # config says @candidate
        "calibrator": {"kind": "sigmoid"},
        "threshold": 0.03,
    }
    with pytest.raises(ModelConfigError, match="belong to another model"):
        ModelService.from_config(_config(tmp_path, block))
