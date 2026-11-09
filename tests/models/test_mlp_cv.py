"""MLP comparison module: the model set, the pairs, and a full run on mini data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from failsense.models import mlp_cv
from failsense.models.baseline_cv import load_training_frame
from failsense.models.pipeline import split_feature_columns


@pytest.fixture
def mini(make_mini_scania):  # type: ignore[no-untyped-def]
    config = make_mini_scania()
    frame = load_training_frame(config, 5)
    return config, frame, split_feature_columns(frame)


def test_the_five_models_and_their_columns(mini) -> None:  # type: ignore[no-untyped-def]
    _, _, cols = mini
    specs = mlp_cv.build_specs(cols, 42, 1)
    assert list(specs) == ["logreg_reference", "xgboost", "xgboost__engineered", "mlp",
                           "mlp_engineered"]  # fmt: skip
    assert specs["mlp"].columns is None  # raw numeric + specification columns
    engineered = set(specs["mlp_engineered"].columns)
    assert set(cols.engineered) <= engineered and set(cols.raw) <= engineered
    assert set(specs["mlp_engineered"].columns) == set(specs["xgboost__engineered"].columns)
    assert specs["mlp"].family == specs["mlp_engineered"].family == "neural_network"


def test_every_pair_refers_to_a_model_of_the_set(mini) -> None:  # type: ignore[no-untyped-def]
    _, _, cols = mini
    names = set(mlp_cv.build_specs(cols, 42, 1))
    assert all(a in names and b in names for a, b in mlp_cv.PAIRS)


def test_no_label_or_vehicle_column_reaches_the_networks(mini) -> None:  # type: ignore[no-untyped-def]
    from failsense.models.pipeline import FORBIDDEN_FEATURES

    _, _, cols = mini
    for spec in mlp_cv.build_specs(cols, 42, 1).values():
        columns = set(spec.columns or (cols.raw + cols.categorical))
        assert not columns & FORBIDDEN_FEATURES, spec.name


def test_run_on_mini_data_writes_the_report_and_pairs(mini) -> None:  # type: ignore[no-untyped-def]
    config, _, _ = mini
    summary = mlp_cv.run(config, track=False)
    assert set(summary) == {"logreg_reference", "xgboost", "xgboost__engineered", "mlp",
                            "mlp_engineered"}  # fmt: skip
    assert set(summary["mlp"]["paired"]) == {"logreg_reference", "xgboost"}
    assert set(summary["mlp_engineered"]["paired"]) == {"mlp", "xgboost__engineered"}
    assert all(0 <= s["pr_auc"] <= 1 for s in summary.values())
    report = pd.read_csv(Path(config["paths"]["reports"]) / mlp_cv.REPORT_FILE)
    assert len(report) == 5 and "diff_vs_xgboost" in report.columns
    assert report.loc[report["model"] == "mlp", "folds_won_vs_xgboost"].iloc[0] in range(6)
