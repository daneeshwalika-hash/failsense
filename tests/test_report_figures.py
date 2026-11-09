"""Report figures: input validation, loaders for the saved report files, one full run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from failsense import report_figures as rf

CONFUSION = pd.DataFrame(
    np.arange(25).reshape(5, 5),
    index=[f"true {k}" for k in range(5)],
    columns=[f"class {k}" for k in range(5)],
)


def write_reports(reports: Path) -> None:
    """Minimal versions of the files the loaders read."""
    reports.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "model": ["logreg_reference", "xgboost", "not_listed"],
            "pr_auc": [0.11, 0.13, 0.5],
            "pr_auc_std": [0.005, 0.008, 0.1],
        }
    ).to_csv(reports / "compare_models_q1.csv", index=False)
    pd.DataFrame(
        {
            "model": ["xgboost__engineered", "xgboost__no_specs"],
            "pr_auc": [0.14, 0.12],
            "pr_auc_std": [0.008, 0.004],
        }
    ).to_csv(reports / "compare_models_q1_recheck.csv", index=False)
    pd.DataFrame(
        {
            "variant": ["raw", "sigmoid"],
            "cost_never": [57400, 57400],
            "cost_always": [34370, 34370],
            "cost_top_5%": [50231, 50231],
            "cost_cross_fitted": [28220, 28220],
        }
    ).to_csv(reports / "threshold_q1.csv", index=False)
    pd.DataFrame(
        {
            "model": [
                "dummy_mean",
                "ridge_raw",
                "ridge_rolling",
                "lightgbm_raw",
                "lightgbm_rolling",
            ],
            "rmse": [41.7, 21.6, 19.6, 18.7, 18.1],
            "rmse_std": [0.0, 1.4, 0.9, 0.6, 0.8],
        }
    ).to_csv(reports / "rul_timeseries_fd001.csv", index=False)
    summary = {
        "constant_baselines": {f"always_{k}": 50_000.0 + 1000 * k for k in range(5)},
        "decisions": {
            key: {"total_cost": cost}
            for key, cost in (
                ("uncorrected_argmax", 56_930.0),
                ("uncorrected_expected_cost", 44_280.0),
                ("corrected_expected_cost", 43_871.0),
            )  # fmt: skip
        },
    }
    (reports / "q2_classes_summary.json").write_text(json.dumps(summary))
    for key in ("uncorrected_expected_cost", "corrected_expected_cost"):
        CONFUSION.to_csv(reports / f"q2_confusion_{key}.csv")


# --- drawing ----------------------------------------------------------------------------------
def test_bar_chart_writes_a_png(tmp_path: Path) -> None:
    path = rf.bar_chart(["a", "b", "c"], [3.0, 1.0, 2.0], tmp_path / "x" / "bars.png", "t", "x",
                        highlight="b", errors=[0.1, 0.1, 0.1], reference=(2.0, "ref"),
                        subtitle="a long subtitle " * 10, lower_is_better=True)  # fmt: skip
    assert path.is_file() and path.stat().st_size > 2000


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"labels": [], "values": []}, "non-empty"),
        ({"labels": ["a"], "values": [1.0, 2.0]}, "equal length"),
        ({"labels": ["a"], "values": [1.0], "errors": [0.1, 0.2]}, "errors"),
        ({"labels": ["a"], "values": [1.0], "highlight": "z"}, "highlight"),
    ],
)
def test_bar_chart_rejects_bad_inputs(tmp_path: Path, kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        rf.bar_chart(path=tmp_path / "b.png", title="t", xlabel="x", **kwargs)


def test_confusion_panels(tmp_path: Path) -> None:
    path = rf.confusion_panels({"A": CONFUSION, "B": CONFUSION}, tmp_path / "c.png", "t")
    assert path.stat().st_size > 2000
    with pytest.raises(ValueError, match="at least one"):
        rf.confusion_panels({}, tmp_path / "d.png", "t")


# --- loaders ----------------------------------------------------------------------------------
def test_loaders_read_the_saved_files(tmp_path: Path) -> None:
    write_reports(tmp_path)
    labels, values, stds = rf.q1_model_comparison(tmp_path)
    assert labels == ["Logistic regression", "XGBoost", "XGBoost + engineered features"]
    assert values == [0.11, 0.13, 0.14] and len(stds) == 3  # unlisted models are ignored
    costs = rf.q1_costs(tmp_path)
    assert (
        costs["Never alert"] == 57_400
        and costs["Model, cost-based threshold (cross-fitted)"] == 28_220
    )
    q2, best, tables = rf.q2_costs(tmp_path)
    assert q2["Always class 4"] == 54_000 and q2[best] == 43_871
    assert set(tables) == {"Uncorrected", "Prior-shift corrected"}
    assert rf.q3_rmse(tmp_path)[1] == [41.7, 21.6, 19.6, 18.7, 18.1]


def test_a_missing_report_names_the_make_target(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make compare"):
        rf.q1_model_comparison(tmp_path)
    with pytest.raises(FileNotFoundError, match="make q2"):
        rf.q2_costs(tmp_path)
    with pytest.raises(FileNotFoundError, match="make calibrate"):
        rf.q1_costs(tmp_path)
    with pytest.raises(FileNotFoundError, match="make rul-timeseries"):
        rf.q3_rmse(tmp_path)


def test_q3_without_any_known_model_raises(tmp_path: Path) -> None:
    write_reports(tmp_path)
    pd.DataFrame({"model": ["other"], "rmse": [1.0], "rmse_std": [0.1]}).to_csv(
        tmp_path / "rul_timeseries_fd001.csv", index=False
    )
    with pytest.raises(ValueError, match="none of the expected"):
        rf.q3_rmse(tmp_path)


def test_run_draws_all_five_figures(tmp_path: Path) -> None:
    write_reports(tmp_path / "reports")
    paths = rf.run({"paths": {"reports": str(tmp_path / "reports")}})
    assert [p.name for p in paths] == [
        "results_q1_model_comparison.png", "results_q1_cost.png", "results_q2_cost.png",
        "results_q2_confusion.png", "results_q3_rmse.png",
    ]  # fmt: skip
    assert all(p.is_file() for p in paths)
