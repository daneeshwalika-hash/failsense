"""Tests for failsense.models.eval_set (rank matching, samplings, reports)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from failsense.data.labels import BINARY_COL, CLASS_COL, build_labels
from failsense.data.split import make_group_folds
from failsense.models import baseline_cv, eval_set
from failsense.models.compare_cutoffs import assign_vehicle_folds, build_fixed_eval_folds
from failsense.models.eval_set import (
    build_eval_cutoffs,
    cutoffs_from_time_step,
    eval_set_report,
    known_labels,
    match_cutoffs,
    matched_training_cutoffs,
    plot_eval_set,
    positive_rate_report,
    validation_time_steps,
    vehicle_fold_from_folds,
    wilson_interval,
)

BOUNDS = (48.0, 24.0, 12.0, 6.0)


def make_labels(reach: dict[int, int], step: float = 10.0) -> pd.DataFrame:
    """Label table: truck -> readouts at step, 2 * step, ..., reach (all known, binary 0)."""
    rows = []
    for vehicle, top in reach.items():
        for t in np.arange(step, top + step / 2, step):
            rows.append({"vehicle_id": vehicle, "time_step": float(t)})
    df = pd.DataFrame(rows)
    df[CLASS_COL] = pd.array([0] * len(df), dtype="Int8")
    df[BINARY_COL] = pd.array([0] * len(df), dtype="Int8")
    return df


@pytest.fixture
def labels() -> pd.DataFrame:
    """200 trucks that all reach time step 200 (readouts every 10 steps), shuffled order."""
    return make_labels({v: 200 for v in range(200)}).sample(frac=1.0, random_state=0)


def ks(a, b) -> float:
    return float(stats.ks_2samp(np.asarray(a), np.asarray(b)).statistic)


def test_match_gives_one_row_per_truck_in_original_order(labels: pd.DataFrame) -> None:
    chosen = match_cutoffs(known_labels(labels), np.linspace(50, 200, 100), seed=1)
    rows = labels.loc[chosen]
    assert rows["vehicle_id"].is_unique and len(rows) == labels["vehicle_id"].nunique()
    assert chosen.tolist() == [i for i in labels.index if i in set(chosen)]  # original order


def test_matching_moves_the_distribution_toward_the_target(labels: pd.DataFrame) -> None:
    targets = np.random.default_rng(0).uniform(120, 200, size=500)  # late readouts
    matched = labels.loc[match_cutoffs(known_labels(labels), targets, seed=1)]
    uniform = labels.groupby("vehicle_id").sample(1, random_state=1)  # a random readout per truck
    # readouts lie on a 10-step grid, so the matched values are discrete: KS stays below 0.1
    assert ks(matched["time_step"], targets) < 0.1
    assert ks(matched["time_step"], targets) < ks(uniform["time_step"], targets) - 0.3


def test_a_truck_that_cannot_reach_its_target_takes_its_latest_readout() -> None:
    labels = pd.concat([make_labels({1: 50}), make_labels({2: 200})], ignore_index=True)
    chosen = labels.loc[match_cutoffs(known_labels(labels), [150.0] * 10, seed=0)]
    by_truck = chosen.set_index("vehicle_id")["time_step"]
    assert by_truck[1] == 50.0  # latest it has, never overshoots
    assert by_truck[2] == 150.0  # nearest readout to the target


def test_short_reach_trucks_get_the_small_targets(labels: pd.DataFrame) -> None:
    # rank matching: the truck that cannot go late must not be handed a late target
    mixed = pd.concat(
        [make_labels({v: 60 for v in range(100)}), make_labels({v: 200 for v in range(100, 200)})],
        ignore_index=True,
    )
    targets = np.r_[np.full(100, 40.0), np.full(100, 180.0)]
    rows = mixed.loc[match_cutoffs(known_labels(mixed), targets, seed=0)]
    by_truck = rows.groupby("vehicle_id")["time_step"].first()
    assert (by_truck.loc[:99] == 40.0).all() and (by_truck.loc[100:] == 180.0).all()


def test_deterministic_with_seed_and_independent_of_row_order(labels: pd.DataFrame) -> None:
    targets = np.linspace(50, 200, 300)
    a = match_cutoffs(known_labels(labels), targets, seed=3)
    b = match_cutoffs(known_labels(labels), targets, seed=3)
    shuffled = known_labels(labels).sample(frac=1.0, random_state=9)
    c = match_cutoffs(shuffled, targets, seed=3)
    assert a.tolist() == b.tolist()
    assert set(a) == set(c)  # the same rows although the table was reordered


def test_noise_changes_the_pairing_with_the_seed(labels: pd.DataFrame) -> None:
    mixed = make_labels({v: 40 + 3 * v for v in range(100)})
    targets = np.linspace(30, 300, 200)
    k = known_labels(mixed)
    a = set(match_cutoffs(k, targets, seed=1, noise=0.5))
    b = set(match_cutoffs(k, targets, seed=2, noise=0.5))
    assert a != b
    assert set(match_cutoffs(k, targets, seed=1, noise=0.5)) == a  # still reproducible


@pytest.mark.parametrize(
    ("targets", "noise", "match"),
    [([], 0.0, "non-empty"), ([1.0, np.nan], 0.0, "NaN"), ([1.0], -0.1, "noise")],
)
def test_match_validates_its_input(labels, targets, noise, match) -> None:
    with pytest.raises(ValueError, match=match):
        match_cutoffs(known_labels(labels), targets, seed=0, noise=noise)
    duplicated = pd.concat([labels, labels])  # the same index label twice
    with pytest.raises(ValueError, match="unique index"):
        match_cutoffs(known_labels(duplicated), [1.0], seed=0)
    with pytest.raises(ValueError, match="no known-label rows"):
        match_cutoffs(labels.iloc[:0], [1.0], seed=0)


def test_eval_set_never_uses_unknown_labels(labels: pd.DataFrame) -> None:
    labels = labels.copy()
    late = labels["time_step"] > 150
    labels[CLASS_COL] = labels[CLASS_COL].mask(late)  # late readouts become unknown
    rows = labels.loc[build_eval_cutoffs(labels, np.full(50, 190.0), seed=0).index]
    assert rows[CLASS_COL].notna().all() and rows["time_step"].max() <= 150.0
    assert len(rows) == 200


def test_matched_training_gives_at_most_k_unique_rows_and_moves_toward_target(labels) -> None:
    targets = np.random.default_rng(1).uniform(120, 200, size=500)
    rows = matched_training_cutoffs(labels, targets, k=5, seed=1)
    per_truck = rows.groupby("vehicle_id").size()
    assert per_truck.max() <= 5 and per_truck.min() >= 1
    uniform = labels.groupby("vehicle_id").sample(5, random_state=1)
    assert ks(rows["time_step"], targets) < ks(uniform["time_step"], targets) - 0.2
    with pytest.raises(ValueError, match="k must be"):
        matched_training_cutoffs(labels, targets, k=0, seed=1)


def test_cutoffs_from_time_step_respects_the_minimum(labels: pd.DataFrame) -> None:
    rows = cutoffs_from_time_step(labels, 70, 5, seed=0)
    assert (rows["time_step"] >= 70).all() and rows.groupby("vehicle_id").size().max() <= 5
    assert cutoffs_from_time_step(labels, 1000, 5, seed=0).empty  # no truck reaches that far


def test_vehicle_fold_from_folds_is_a_partition() -> None:
    positive = np.repeat((np.arange(40) % 5 == 0).astype(int), 3)
    frame = pd.DataFrame({"vehicle_id": np.repeat(np.arange(40), 3), BINARY_COL: positive})
    folds = make_group_folds(frame, n_splits=5, seed=42)
    fold_of = vehicle_fold_from_folds(frame, folds)
    assert len(fold_of) == 40 and set(fold_of) == {0, 1, 2, 3, 4}
    for fold_no, (_, val_idx) in enumerate(folds):
        assert set(frame["vehicle_id"].iloc[val_idx]) == set(fold_of[fold_of == fold_no].index)
    bad = [(folds[0][0], folds[0][1]), (folds[1][0], folds[0][1])]  # same truck validated twice
    with pytest.raises(ValueError, match="more than one fold"):
        vehicle_fold_from_folds(frame, bad)
    with pytest.raises(ValueError, match="no validation part"):
        vehicle_fold_from_folds(frame, [folds[0]])


def test_wilson_interval() -> None:
    low, high = wilson_interval(136, 5046)
    assert low == pytest.approx(0.0228, abs=5e-4) and high == pytest.approx(0.0319, abs=5e-4)
    assert wilson_interval(0, 10)[0] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="positive"):
        wilson_interval(1, 0)


def _tte(n: int, repaired_every: int = 4) -> pd.DataFrame:
    repaired = (np.arange(n) % repaired_every == 0).astype(int)
    return pd.DataFrame({"vehicle_id": np.arange(n), "in_study_repair": repaired})


def test_positive_rate_report_flags_a_rate_far_from_the_official_one() -> None:
    n = 1000
    rows = pd.DataFrame({"vehicle_id": np.arange(n), "time_step": np.linspace(10, 400, n),
                         BINARY_COL: (np.arange(n) % 4 == 0).astype(int)})  # fmt: skip
    report = positive_rate_report(rows, rows.iloc[:300], _tte(n))
    assert report["overall"]["rate"] == 0.25 and report["overall"]["far_from_official"]
    assert report["overall"]["ratio_to_official"] > 5
    by_q = report["by_quartile"]
    assert list(by_q.index) == ["Q1 (earliest)", "Q2", "Q3", "Q4 (latest)"]
    assert by_q["n"].sum() == n and by_q["time_step_min"].is_monotonic_increasing
    shares = report["repaired_share"]
    assert shares["eval_set_trucks"] == pytest.approx(0.25)
    assert shares["positive_rate_in_repaired_trucks_eval"] == 1.0


def test_positive_rate_report_accepts_a_realistic_rate() -> None:
    n = 5000
    positive = np.zeros(n, dtype=int)
    positive[:136] = 1  # exactly the official rate
    rows = pd.DataFrame({"vehicle_id": np.arange(n), "time_step": np.linspace(10, 400, n),
                         BINARY_COL: positive})  # fmt: skip
    report = positive_rate_report(rows, rows, _tte(n))
    assert not report["overall"]["far_from_official"]
    assert report["overall"]["ratio_to_official"] == pytest.approx(
        (136 / 5000) / (136 / 5046), rel=1e-6
    )


def test_eval_set_report_and_plot(tmp_path: Path) -> None:
    reference = pd.Series(np.linspace(50, 300, 400))
    table = eval_set_report({"same": reference.copy(), "early": reference - 40}, reference)
    ks_by = dict(zip(table["sample"], table["ks_vs_validation"], strict=True))
    assert ks_by["same"] == 0.0 and ks_by["early"] > 0.1
    assert ks_by["official validation (last readout)"] == 0.0
    path = plot_eval_set({"early": reference - 40}, reference, tmp_path / "f" / "eval.png")
    assert path.exists() and path.stat().st_size > 0


# --- on the tiny SCANIA-shaped dataset -------------------------------------------------------
@pytest.mark.leakage
def test_target_uses_only_validation_time_steps_never_labels(make_mini_scania, monkeypatch) -> None:
    requested: list[tuple[str, str]] = []
    original = eval_set.table_path

    def spy(config, split, table):
        requested.append((split, table))
        return original(config, split, table)

    monkeypatch.setattr(eval_set, "table_path", spy)
    config = make_mini_scania()  # has no validation labels entry: any read would fail
    assert "labels" not in config["datasets"]["scania"]["validation"]
    targets = validation_time_steps(config)
    assert requested == [("validation", "readouts")]
    assert len(targets) == 4 and set(targets.index) == {100, 101, 102, 103}


@pytest.fixture
def mini_labels(make_mini_scania):
    config = make_mini_scania()
    readouts = baseline_cv.read_table(baseline_cv.table_path(config, "train", "readouts"))
    tte = baseline_cv.read_table(baseline_cv.table_path(config, "train", "tte"))
    return config, build_labels(readouts, tte, BOUNDS)


def test_eval_set_has_one_known_row_per_truck_and_is_deterministic(mini_labels) -> None:
    config, labels = mini_labels
    targets = validation_time_steps(config)
    a = build_eval_cutoffs(labels, targets, seed=42)
    b = build_eval_cutoffs(labels, targets, seed=42)
    assert a.index.tolist() == b.index.tolist()  # deterministic with the seed
    assert a["vehicle_id"].is_unique and a["vehicle_id"].nunique() == 40  # one row per truck
    assert a[CLASS_COL].notna().all()  # only known labels


@pytest.mark.leakage
def test_no_truck_is_in_the_train_and_eval_part_of_any_fold(mini_labels) -> None:
    config, labels = mini_labels
    eval_rows = build_eval_cutoffs(labels, validation_time_steps(config), seed=42)
    k5 = labels.loc[labels[CLASS_COL].notna()].groupby("vehicle_id").sample(5, random_state=0)
    k5 = k5.assign(**{BINARY_COL: k5[BINARY_COL].astype(int)})
    vehicle_fold = vehicle_fold_from_folds(k5, make_group_folds(k5, n_splits=3, seed=42))
    folds = build_fixed_eval_folds(
        k5["vehicle_id"].reset_index(drop=True),
        eval_rows["vehicle_id"].reset_index(drop=True),
        vehicle_fold,
    )
    train_ids = k5["vehicle_id"].to_numpy()
    eval_ids = eval_rows["vehicle_id"].to_numpy()
    offset = len(k5)
    assert len(folds) == 3
    for train_pos, val_pos in folds:
        assert set(train_ids[train_pos]).isdisjoint(eval_ids[val_pos - offset])
    # every truck is evaluated exactly once over all folds
    assert sorted(np.concatenate([v - offset for _, v in folds]).tolist()) == list(range(40))
    assert assign_vehicle_folds is not None  # the old random-eval helper is still there


# --- calibrated evaluation set (the final protocol) and the literal procedure ------------------
from failsense.models.eval_set import (  # noqa: E402
    EVAL_METHODS,
    build_calibrated_eval_cutoffs,
    build_eval_cutoffs_by_method,
    build_literal_eval_cutoffs,
    with_censored_tail,
)


def synthetic_study(n: int = 600, repaired_every: int = 8):
    """Trucks with readouts every 10 steps up to their study length; some repaired at the end."""
    from failsense.data.labels import REPAIR_COL, STUDY_LENGTH_COL

    lengths = 100 + (np.arange(n) % 25) * 10  # 100 .. 340
    rows = [
        (v, float(t)) for v, length in enumerate(lengths) for t in range(10, int(length) + 1, 10)
    ]
    readouts = pd.DataFrame(rows, columns=["vehicle_id", "time_step"])
    repaired = (np.arange(n) % repaired_every == 0).astype(int)
    tte = pd.DataFrame(
        {"vehicle_id": np.arange(n), STUDY_LENGTH_COL: lengths.astype(float), REPAIR_COL: repaired}
    )
    return build_labels(readouts, tte, BOUNDS), pd.Series(repaired, index=np.arange(n))


def test_with_censored_tail_labels_the_unknown_rows_as_class_zero() -> None:
    labels, _ = synthetic_study()
    assert labels[CLASS_COL].isna().any()  # censored near-end readouts are unknown for training
    evaluation = with_censored_tail(labels)
    assert evaluation[CLASS_COL].notna().all() and evaluation[BINARY_COL].notna().all()
    unknown = labels[CLASS_COL].isna()
    assert (evaluation.loc[unknown, BINARY_COL] == 0).all()
    pd.testing.assert_series_equal(  # known rows are unchanged
        evaluation.loc[~unknown, CLASS_COL], labels.loc[~unknown, CLASS_COL], check_names=False
    )


def test_literal_procedure_is_one_uniform_readout_per_truck_and_deterministic() -> None:
    labels, _ = synthetic_study()
    a = build_literal_eval_cutoffs(labels, seed=3)
    assert a["vehicle_id"].is_unique and len(a) == 600
    assert a.index.tolist() == build_literal_eval_cutoffs(labels, seed=3).index.tolist()
    assert a[CLASS_COL].notna().all()  # censored tail included as class 0
    known_only = build_literal_eval_cutoffs(labels, seed=3, include_censored_tail=False)
    assert known_only[CLASS_COL].notna().all() and len(known_only) <= 600


def test_calibrated_set_hits_the_target_rate_and_the_time_step_distribution() -> None:
    labels, repaired = synthetic_study()
    targets = np.random.default_rng(0).uniform(60, 330, size=500)
    rows, info = build_calibrated_eval_cutoffs(labels, targets, repaired, 0.04, seed=1)
    assert rows["vehicle_id"].is_unique and len(rows) == 600
    assert rows[BINARY_COL].mean() == pytest.approx(0.04, abs=0.006)
    assert info["rate"] == pytest.approx(rows[BINARY_COL].mean())
    assert 0.0 < info["end_near_share"] < 1.0 and info["early_cut_trucks"] > 0
    uncalibrated = labels.loc[match_cutoffs(known_labels(labels), targets, seed=1)]
    assert (
        rows[BINARY_COL].mean() < uncalibrated[BINARY_COL].mean()
    )  # rank matching alone is too high
    assert ks(rows["time_step"], targets) < 0.12


def test_early_cut_repaired_trucks_are_only_evaluated_on_negative_rows() -> None:
    labels, repaired = synthetic_study()
    targets = np.linspace(60, 330, 500)
    rows, info = build_calibrated_eval_cutoffs(labels, targets, repaired, 0.02, seed=1)
    repaired_rows = rows[rows["vehicle_id"].map(repaired) == 1]
    n_positive = int(repaired_rows[BINARY_COL].sum())
    # the trucks that are not end-near sit on a negative row, so positives < repaired trucks
    assert n_positive < len(repaired_rows) and info["early_cut_trucks"] >= 1
    assert (
        rows.loc[rows["vehicle_id"].map(repaired) == 0, BINARY_COL].eq(0).all()
    )  # censored: class 0


def test_calibration_is_monotone_in_the_target_rate_and_deterministic() -> None:
    labels, repaired = synthetic_study()
    targets = np.linspace(60, 330, 500)
    low, _ = build_calibrated_eval_cutoffs(labels, targets, repaired, 0.02, seed=1)
    high, _ = build_calibrated_eval_cutoffs(labels, targets, repaired, 0.06, seed=1)
    again, _ = build_calibrated_eval_cutoffs(labels, targets, repaired, 0.02, seed=1)
    assert low[BINARY_COL].mean() < high[BINARY_COL].mean()
    assert low.index.tolist() == again.index.tolist()


def test_calibrated_warns_if_the_target_rate_cannot_be_reached(caplog) -> None:
    labels, repaired = synthetic_study()
    with caplog.at_level("WARNING"):
        rows, info = build_calibrated_eval_cutoffs(
            labels, np.linspace(60, 330, 500), repaired, 0.9, 1
        )
    assert info["end_near_share"] == 1.0 and "above the maximum reachable" in caplog.text
    assert rows[BINARY_COL].mean() < 0.9


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_calibrated_rejects_an_invalid_target_rate(bad: float) -> None:
    labels, repaired = synthetic_study(n=40)
    with pytest.raises(ValueError, match="target_rate"):
        build_calibrated_eval_cutoffs(labels, [100.0] * 5, repaired, bad, seed=1)


def test_dispatch_by_method() -> None:
    labels, repaired = synthetic_study()
    targets = np.linspace(60, 330, 500)
    sizes = {}
    for method in EVAL_METHODS:
        rows, info = build_eval_cutoffs_by_method(method, labels, targets, repaired, seed=1)
        assert rows["vehicle_id"].is_unique
        sizes[method] = len(rows)
        assert bool(info) == (method == "calibrated")
    rank, _ = build_eval_cutoffs_by_method("rank", labels, targets, repaired, seed=1)
    assert rank.index.tolist() == build_eval_cutoffs(labels, targets, seed=1).index.tolist()
    with pytest.raises(ValueError, match="unknown eval method"):
        build_eval_cutoffs_by_method("magic", labels, targets, repaired, seed=1)


def test_trucks_without_any_known_readout_are_dropped_from_the_evaluation_set() -> None:
    from failsense.data.labels import REPAIR_COL, STUDY_LENGTH_COL

    # truck 0: censored, only 3 readouts, all within 48 steps of the end -> every label unknown
    readouts = pd.DataFrame(
        {
            "vehicle_id": [0, 0, 0] + [1] * 10,
            "time_step": [10.0, 20.0, 30.0] + list(range(10, 110, 10)),
        }
    )
    tte = pd.DataFrame({"vehicle_id": [0, 1], STUDY_LENGTH_COL: [40.0, 150.0], REPAIR_COL: [0, 1]})
    labels = build_labels(readouts, tte, BOUNDS)
    assert labels.loc[labels["vehicle_id"] == 0, CLASS_COL].isna().all()
    repaired = pd.Series([0, 1], index=[0, 1])
    for method in EVAL_METHODS:
        rows, _ = build_eval_cutoffs_by_method(method, labels, [60.0] * 4, repaired, seed=1)
        assert set(rows["vehicle_id"]) == {1}, method
