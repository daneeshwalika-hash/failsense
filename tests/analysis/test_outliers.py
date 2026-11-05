"""Tests for failsense.analysis.outliers (small fixtures, no raw data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from failsense.analysis.outliers import (
    SUMMARY_COLUMNS,
    plot_repair_rate,
    plot_skew_vs_share,
    render_markdown,
    repair_rate_by_outlier,
    select_top_features,
    summarize_features,
)
from failsense.data.labels import BINARY_COL


@pytest.fixture
def df() -> pd.DataFrame:
    """100 rows: `x` has 20 huge values; `const` is constant; `zeros` is zero-heavy."""
    x = np.r_[np.arange(1, 81, dtype=float), np.full(20, 10_000.0)]
    y = np.zeros(100, dtype=int)
    y[80:] = 1  # every outlier row is a repair
    y[:4] = 1  # 4 of the 80 normal rows are repairs
    return pd.DataFrame(
        {
            "x": x,
            "const": np.full(100, 5.0),
            "zeros": np.r_[np.zeros(95), np.arange(1.0, 6.0)],
            BINARY_COL: y,
        }
    )


def test_summary_has_one_row_per_feature_and_known_columns(df: pd.DataFrame) -> None:
    out = summarize_features(df, ["x", "const", "zeros"])
    assert list(out.columns) == SUMMARY_COLUMNS
    assert out["feature"].tolist() == ["x", "const", "zeros"]


def test_summary_counts_values_above_the_fence(df: pd.DataFrame) -> None:
    row = summarize_features(df, ["x"]).iloc[0]
    assert row["n_above_fence"] == 20
    assert row["share_above_fence"] == pytest.approx(0.2)
    assert row["max"] == 10_000.0
    assert row["p99"] == pytest.approx(10_000.0)
    assert row["skew"] > 1  # strongly right-skewed


def test_constant_and_zero_heavy_columns_have_no_fence(df: pd.DataFrame) -> None:
    out = summarize_features(df, ["const", "zeros"]).set_index("feature")
    # IQR = 0 -> a fence at Q3 would call every non-zero value an outlier: undefined instead.
    assert out["iqr"].tolist() == [0.0, 0.0]
    assert out["share_above_fence"].isna().all()
    assert (out["n_above_fence"] == 0).all()


def test_summary_counts_missing_values() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0, 5.0, 100.0]})
    row = summarize_features(frame, ["a"]).iloc[0]
    assert row["n"] == 5 and row["n_missing"] == 1


def test_summary_empty_frame_and_single_row_give_nan_not_errors() -> None:
    empty = summarize_features(pd.DataFrame({"a": pd.Series([], dtype=float)}), ["a"])
    assert empty.loc[0, "n"] == 0 and np.isnan(empty.loc[0, "share_above_fence"])
    single = summarize_features(pd.DataFrame({"a": [3.0]}), ["a"])
    assert single.loc[0, "p50"] == 3.0 and np.isnan(single.loc[0, "skew"])


def test_summary_all_nan_column() -> None:
    out = summarize_features(pd.DataFrame({"a": [np.nan, np.nan]}), ["a"])
    assert out.loc[0, "n"] == 0 and np.isnan(out.loc[0, "p50"])


@pytest.mark.leakage
def test_summary_missing_and_forbidden_columns_raise(df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing"):
        summarize_features(df, ["nope"])
    with pytest.raises(ValueError, match="leakage"):
        summarize_features(df, [BINARY_COL])
    with pytest.raises(ValueError, match="leakage"):
        summarize_features(df.assign(vehicle_id=1), ["vehicle_id"])


def test_select_top_features_needs_a_fence_and_enough_outliers(df: pd.DataFrame) -> None:
    summary = summarize_features(df, ["x", "const", "zeros"])
    assert select_top_features(summary, n=5, min_outliers=10) == ["x"]
    assert select_top_features(summary, n=5, min_outliers=21) == []  # only 20 outliers
    assert select_top_features(summary, n=0, min_outliers=1) == []


def test_repair_rate_outlier_vs_normal(df: pd.DataFrame) -> None:
    out = repair_rate_by_outlier(df, ["x"], min_outliers=10)
    row = out.iloc[0]
    assert row["n_outlier"] == 20 and row["rate_outlier"] == 1.0
    assert row["n_normal"] == 80 and row["rate_normal"] == pytest.approx(0.05)
    assert row["ratio"] == pytest.approx(20.0)


def test_repair_rate_skips_features_without_fence_or_too_few_outliers(df: pd.DataFrame) -> None:
    assert repair_rate_by_outlier(df, ["const", "zeros"], min_outliers=1).empty
    assert repair_rate_by_outlier(df, ["x"], min_outliers=21).empty


def test_repair_rate_ratio_is_nan_when_normal_rate_is_zero(df: pd.DataFrame) -> None:
    frame = df.assign(**{BINARY_COL: np.r_[np.zeros(80), np.ones(20)].astype(int)})
    out = repair_rate_by_outlier(frame, ["x"], min_outliers=10)
    assert out.loc[0, "rate_normal"] == 0.0 and np.isnan(out.loc[0, "ratio"])


def test_repair_rate_ignores_nan_rows(df: pd.DataFrame) -> None:
    frame = df.copy()
    frame.loc[:9, "x"] = np.nan
    out = repair_rate_by_outlier(frame, ["x"], min_outliers=10)
    assert out.loc[0, "n_outlier"] + out.loc[0, "n_normal"] == 90


def test_repair_rate_validates_input(df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing"):
        repair_rate_by_outlier(df.drop(columns=BINARY_COL), ["x"])
    with pytest.raises(ValueError, match="0 and 1"):
        repair_rate_by_outlier(df.assign(**{BINARY_COL: 2}), ["x"])
    with pytest.raises(ValueError, match="leakage"):
        repair_rate_by_outlier(df, [BINARY_COL])
    assert repair_rate_by_outlier(df.iloc[:0], ["x"]).empty


def test_analysis_does_not_modify_the_input(df: pd.DataFrame) -> None:
    before = df.copy()
    summarize_features(df, ["x"])
    repair_rate_by_outlier(df, ["x"], min_outliers=10)
    pd.testing.assert_frame_equal(df, before)  # nothing removed, nothing clipped


def test_markdown_and_figures(df: pd.DataFrame, tmp_path: Path) -> None:
    summary = summarize_features(df, ["x", "const", "zeros"])
    repair = repair_rate_by_outlier(df, ["x"], min_outliers=10)
    text = render_markdown(summary, repair, base_rate=0.24, n_rows=100, k_label="5")
    assert "| x |" in text and "Nothing was removed or clipped" in text
    for path in (
        plot_skew_vs_share(summary, tmp_path / "a" / "skew.png"),
        plot_repair_rate(repair, 0.24, tmp_path / "b" / "rate.png"),
    ):
        assert path.exists() and path.stat().st_size > 0
