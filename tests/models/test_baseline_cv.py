"""Tests for failsense.models.baseline_cv (helpers only, no real data)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from failsense.data.labels import BINARY_COL
from failsense.models.baseline_cv import build_training_frame, parse_n_cutoffs, read_table

BOUNDS = (48.0, 24.0, 12.0, 6.0)


@pytest.fixture
def readouts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "vehicle_id": [1, 1, 1, 2, 2, 2],
            "time_step": [40.0, 60.0, 95.0, 40.0, 60.0, 95.0],
            "171_0": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        }
    )


@pytest.fixture
def tte() -> pd.DataFrame:
    # vehicle 1 repaired at 100, vehicle 2 censored at 100
    return pd.DataFrame(
        {
            "vehicle_id": [1, 2],
            "length_of_study_time_step": [100.0, 100.0],
            "in_study_repair": [1, 0],
        }
    )


@pytest.mark.leakage
def test_training_frame_keeps_only_known_labels(readouts, tte) -> None:
    frame = build_training_frame(readouts, tte, BOUNDS, n_per_vehicle=None, seed=0)
    # vehicle 2 at t=60 (d=40) and t=95 (d=5) are unknown -> dropped
    assert len(frame) == 4
    assert frame[BINARY_COL].tolist() == [0, 1, 1, 0]


@pytest.mark.leakage
def test_training_frame_has_features_and_only_binary_label(readouts, tte) -> None:
    frame = build_training_frame(readouts, tte, BOUNDS, n_per_vehicle=1, seed=0)
    assert list(frame.columns) == ["vehicle_id", "time_step", "171_0", BINARY_COL]
    assert frame["vehicle_id"].value_counts().to_dict() == {1: 1, 2: 1}


def test_training_frame_rows_match_readouts(readouts, tte) -> None:
    frame = build_training_frame(readouts, tte, BOUNDS, n_per_vehicle=None, seed=0)
    pd.testing.assert_frame_equal(frame.drop(columns=[BINARY_COL]), readouts.loc[frame.index])


@pytest.mark.parametrize(("text", "expected"), [("5", 5), ("1", 1), ("all", None), ("ALL", None)])
def test_parse_n_cutoffs(text: str, expected: int | None) -> None:
    assert parse_n_cutoffs(text) == expected


def test_read_table_csv_and_parquet(tmp_path: Path, readouts) -> None:
    pytest.importorskip("pyarrow")  # Parquet engine; installed in the project env
    csv_path = tmp_path / "t.csv"
    parquet_path = tmp_path / "t.parquet"
    readouts.to_csv(csv_path, index=False)
    readouts.to_parquet(parquet_path, index=False)
    pd.testing.assert_frame_equal(read_table(csv_path), readouts)
    pd.testing.assert_frame_equal(read_table(parquet_path), readouts)


@pytest.mark.leakage
def test_training_frame_engineered_features_use_previous_readout(readouts, tte) -> None:
    frame = build_training_frame(
        readouts, tte, BOUNDS, n_per_vehicle=None, seed=0, counter_cols=["171_0"]
    )
    # Deltas come from the FULL readouts: the readout at t=60 (cut-off) has the
    # delta to t=40 although t=40 itself is not a cut-off of its own right.
    row = frame[(frame["vehicle_id"] == 1) & (frame["time_step"] == 60.0)]
    assert row["171_0__delta"].iloc[0] == 1.0
    assert row["171_0__rate"].iloc[0] == pytest.approx(1.0 / 20.0)
    assert BINARY_COL in frame.columns
    assert not {"in_study_repair", "length_of_study_time_step"} & set(frame.columns)


@pytest.mark.leakage
def test_training_frame_delta_does_not_depend_on_sampling(readouts, tte) -> None:
    full = build_training_frame(readouts, tte, BOUNDS, None, 0, counter_cols=["171_0"])
    sub = build_training_frame(readouts, tte, BOUNDS, 1, 0, counter_cols=["171_0"])
    common = sub.index
    pd.testing.assert_series_equal(full.loc[common, "171_0__delta"], sub["171_0__delta"])


def test_training_frame_histogram_features(readouts, tte) -> None:
    readouts = readouts.assign(h_0=1.0, h_1=3.0)
    frame = build_training_frame(
        readouts, tte, BOUNDS, None, 0, hist_groups={"h": ["h_0", "h_1"]}, top_fraction=0.5
    )
    assert (frame["h_1__share"] == 0.75).all()
    assert (frame["h__mean_bin"] == 0.75).all()


def test_mean_proba_scorer_reports_the_average_probability() -> None:
    import numpy as np

    from failsense.models.baseline_cv import SCORING, SCORING_WITH_PROBA
    from failsense.models.pipeline import build_dummy_classifier

    x = pd.DataFrame({"a": range(10)})
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    dummy = build_dummy_classifier().fit(x, y)
    # The prior classifier predicts the share of positives (0.1) for every row.
    assert SCORING_WITH_PROBA["mean_proba"](dummy, x, y) == pytest.approx(0.1)
    assert set(SCORING) < set(SCORING_WITH_PROBA)  # compare_cutoffs' SCORING is unchanged


def test_imbalance_param_names_exist_in_comparisons() -> None:
    from failsense.models.baseline_cv import COMPARISONS, IMBALANCE_OF_MODEL
    from failsense.models.pipeline import IMBALANCE_STRATEGIES

    assert {"logreg_specs_none", "logreg_specs_smote"} <= {n for pair in COMPARISONS for n in pair}
    assert set(IMBALANCE_OF_MODEL.values()) <= {*IMBALANCE_STRATEGIES, "n/a"}


def test_read_table_selects_columns(tmp_path: Path, readouts) -> None:
    parquet = tmp_path / "r.parquet"
    csv = tmp_path / "r.csv"
    readouts.to_parquet(parquet)
    readouts.to_csv(csv, index=False)
    for path in (parquet, csv):
        out = read_table(path, columns=["vehicle_id", "time_step"])
        assert list(out.columns) == ["vehicle_id", "time_step"]
        assert len(out) == len(readouts)
    assert list(read_table(csv).columns) == list(readouts.columns)  # None = all
