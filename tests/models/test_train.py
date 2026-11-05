"""Tests for failsense.models.train (tiny fixtures, temporary MLflow database)."""

from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from failsense.data.labels import BINARY_COL, LABEL_COLUMNS, VEHICLE_COL
from failsense.data.schema import counter_columns, histogram_groups
from failsense.models.baseline_cv import build_training_frame
from failsense.models.pipeline import split_feature_columns
from failsense.models.train import RUN_NAME, fit_reference_model, run

BOUNDS = (48.0, 24.0, 12.0, 6.0)


@pytest.fixture
def frame(mini_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    from failsense.features.build import attach_specifications

    built = build_training_frame(
        mini_frames["readouts"],
        mini_frames["tte"],
        BOUNDS,
        n_per_vehicle=5,
        seed=42,
        counter_cols=counter_columns(),
        hist_groups=histogram_groups(),
    )
    return attach_specifications(built, mini_frames["specs"])


def test_fit_reference_model_predicts_probabilities(frame: pd.DataFrame) -> None:
    pipeline, cols, x = fit_reference_model(frame, random_state=42)
    proba = pipeline.predict_proba(x)
    assert proba.shape == (len(frame), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert [n for n, _ in pipeline.steps] == ["preprocess", "model"]  # no SMOTE
    assert pipeline.named_steps["model"].class_weight == "balanced"


def test_reference_model_uses_log1p_only_on_raw_counters_and_histograms(
    frame: pd.DataFrame,
) -> None:
    pipeline, cols, _ = fit_reference_model(frame, random_state=42)
    names = [t[0] for t in pipeline.named_steps["preprocess"].transformers]
    assert names == ["num", "num_log1p", "cat"]
    assert "time_step" not in cols.log1p
    assert not set(cols.log1p) & set(cols.engineered)
    assert len(cols.log1p) == 105  # 8 counters + 97 histogram bins


@pytest.mark.leakage
def test_model_input_has_no_labels_ids_or_engineered_columns(frame: pd.DataFrame) -> None:
    _, cols, x = fit_reference_model(frame, random_state=42)
    assert not set(x.columns) & (LABEL_COLUMNS | {VEHICLE_COL})
    assert not set(x.columns) & set(cols.engineered)  # engineered features are off


def test_fit_rejects_empty_frame_and_missing_label(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="empty"):
        fit_reference_model(frame.iloc[:0], random_state=42)
    with pytest.raises(ValueError, match="label"):
        fit_reference_model(frame.drop(columns=BINARY_COL), random_state=42)


def test_run_without_tracking_returns_a_fitted_pipeline(make_mini_scania) -> None:
    pipeline = run(make_mini_scania(), track=False)
    assert hasattr(pipeline.named_steps["model"], "coef_")


def test_run_logs_params_metric_and_model_to_mlflow(make_mini_scania) -> None:
    from failsense.models.baseline_cv import load_training_frame
    from failsense.tracking import setup_mlflow

    config = make_mini_scania()
    setup_mlflow(config)
    fitted = run(config, track=True)
    full = load_training_frame(config, 5)
    cols = split_feature_columns(full)
    frame_input = full[cols.raw + cols.categorical]
    client = MlflowClient()
    experiment = client.get_experiment_by_name("test-train")
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    logged = runs[0]
    assert logged.info.run_name == RUN_NAME
    assert logged.data.params["imbalance_strategy"] == "balanced"
    assert logged.data.params["log1p"] == "True"
    assert logged.data.tags["stage"] == "train"
    assert 0.0 <= logged.data.metrics["pr_auc_in_sample"] <= 1.0
    # The saved model can be loaded again and predicts exactly like the fitted one.
    reloaded = mlflow.sklearn.load_model(f"runs:/{logged.info.run_id}/model")
    x = frame_input
    np.testing.assert_allclose(reloaded.predict_proba(x), fitted.predict_proba(x))
