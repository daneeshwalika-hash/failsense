"""Tests for failsense.models.learning_curve.

Synthetic data only (sklearn's make_classification); the SCANIA data is never
loaded. MLflow writes to a temporary SQLite database.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from failsense.models.learning_curve import (
    compute_learning_curve,
    log_learning_curve,
    plot_learning_curve,
)
from failsense.tracking import setup_mlflow

SIZES = (0.2, 0.5, 1.0)


@pytest.fixture
def data() -> tuple[pd.DataFrame, np.ndarray, list]:
    """600 rows, about 20 % positives, 3 stratified folds."""
    x, y = make_classification(n_samples=600, n_features=5, weights=[0.8], random_state=0)
    folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=0).split(x, y))
    return pd.DataFrame(x), y, folds


@pytest.fixture
def curve(data: tuple) -> pd.DataFrame:
    x, y, folds = data
    return compute_learning_curve(LogisticRegression(), x, y, folds, SIZES, n_jobs=1)


def test_compute_learning_curve_shape_and_columns(curve: pd.DataFrame) -> None:
    assert list(curve.columns) == ["train_rows", "train_mean", "train_std", "val_mean", "val_std"]
    assert len(curve) == len(SIZES)
    assert curve["train_rows"].is_monotonic_increasing
    # 100 % = the full training part of a fold (2 of 3 folds = 400 rows).
    assert curve["train_rows"].iloc[-1] == 400


def test_compute_learning_curve_scores_are_probabilities(curve: pd.DataFrame) -> None:
    for col in ("train_mean", "val_mean"):
        assert curve[col].between(0, 1).all()
    assert (curve[["train_std", "val_std"]] >= 0).all().all()


@pytest.mark.parametrize("sizes", [(), (0.0, 1.0), (0.5, 1.5)])
def test_compute_learning_curve_invalid_sizes_raise(data: tuple, sizes: tuple) -> None:
    x, y, folds = data
    with pytest.raises(ValueError, match="fractions"):
        compute_learning_curve(LogisticRegression(), x, y, folds, sizes, n_jobs=1)


def test_plot_learning_curve_writes_png(curve: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "nested" / "lc.png"  # parent folder does not exist yet

    result = plot_learning_curve(curve, baseline=0.2, path=path)

    assert result == path
    assert path.exists()
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"  # PNG file signature


def test_log_learning_curve_metrics_and_artifacts(curve: pd.DataFrame, tmp_path: Path) -> None:
    config = {
        "mlflow": {
            "db_path": str(tmp_path / "mlflow.db"),
            "artifact_dir": str(tmp_path / "artifacts"),
            "experiment": "test-learning-curve",
        }
    }
    setup_mlflow(config)
    figure = plot_learning_curve(curve, baseline=0.2, path=tmp_path / "lc.png")

    run_id = log_learning_curve(curve, [figure], params={"model": "logreg"})

    client = MlflowClient()
    history = client.get_metric_history(run_id, "lc_val_pr_auc")
    assert sorted(m.step for m in history) == curve["train_rows"].tolist()
    artifacts = [a.path for a in client.list_artifacts(run_id, "learning_curve")]
    assert artifacts == ["learning_curve/lc.png"]
