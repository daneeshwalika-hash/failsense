"""Tests for failsense.models.compare_cutoffs (fixed-evaluation folds for k).

Small hand-made frames only; the full SCANIA data is never loaded here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from failsense.data.labels import BINARY_COL, VEHICLE_COL
from failsense.models.compare_cutoffs import assign_vehicle_folds, build_fixed_eval_folds

N_SPLITS = 2


@pytest.fixture
def eval_frame() -> pd.DataFrame:
    """One row per vehicle: 10 vehicles, 4 positives (enough for 2 stratified folds)."""
    return pd.DataFrame(
        {
            VEHICLE_COL: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            BINARY_COL: [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        }
    )


@pytest.fixture
def train_vehicles(eval_frame: pd.DataFrame) -> pd.Series:
    """Training sample with 3 cut-offs per vehicle (like k = 3)."""
    return pd.Series(np.repeat(eval_frame[VEHICLE_COL].to_numpy(), 3), name=VEHICLE_COL)


@pytest.mark.leakage
def test_assign_vehicle_folds_one_fold_per_vehicle(eval_frame: pd.DataFrame) -> None:
    vehicle_fold = assign_vehicle_folds(eval_frame, N_SPLITS, seed=42)

    assert sorted(vehicle_fold.index) == sorted(eval_frame[VEHICLE_COL])
    assert set(vehicle_fold) == {0, 1}  # every fold is used, no -1 left


def test_assign_vehicle_folds_is_reproducible(eval_frame: pd.DataFrame) -> None:
    first = assign_vehicle_folds(eval_frame, N_SPLITS, seed=42)
    second = assign_vehicle_folds(eval_frame, N_SPLITS, seed=42)
    pd.testing.assert_series_equal(first, second)


def test_assign_vehicle_folds_rejects_duplicate_vehicles(eval_frame: pd.DataFrame) -> None:
    duplicated = pd.concat([eval_frame, eval_frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="one row per vehicle"):
        assign_vehicle_folds(duplicated, N_SPLITS, seed=42)


@pytest.mark.leakage
def test_fixed_eval_folds_split_train_and_eval_parts(
    eval_frame: pd.DataFrame, train_vehicles: pd.Series
) -> None:
    vehicle_fold = assign_vehicle_folds(eval_frame, N_SPLITS, seed=42)
    eval_vehicles = eval_frame[VEHICLE_COL]
    offset = len(train_vehicles)

    folds = build_fixed_eval_folds(train_vehicles, eval_vehicles, vehicle_fold)

    assert len(folds) == N_SPLITS
    all_val_pos: list[int] = []
    for train_pos, val_pos in folds:
        # Training rows only from the training sample, validation rows only
        # from the evaluation sample.
        assert (train_pos < offset).all()
        assert (val_pos >= offset).all()
        # No vehicle on both sides of a fold.
        train_ids = set(train_vehicles.to_numpy()[train_pos])
        val_ids = set(eval_vehicles.to_numpy()[val_pos - offset])
        assert train_ids.isdisjoint(val_ids)
        # All cut-offs of every training vehicle are used (3 per vehicle).
        assert len(train_pos) == 3 * len(train_ids)
        all_val_pos.extend(val_pos.tolist())

    # Every evaluation row is validated exactly once over all folds.
    assert sorted(all_val_pos) == list(range(offset, offset + len(eval_vehicles)))


def test_fixed_eval_folds_vehicle_without_fold_raises(
    eval_frame: pd.DataFrame, train_vehicles: pd.Series
) -> None:
    vehicle_fold = assign_vehicle_folds(eval_frame, N_SPLITS, seed=42)
    unknown_vehicle = pd.concat([train_vehicles, pd.Series([999])], ignore_index=True)

    with pytest.raises(ValueError, match="no fold"):
        build_fixed_eval_folds(unknown_vehicle, eval_frame[VEHICLE_COL], vehicle_fold)


# --- validation-like evaluation (05.11) -----------------------------------------------------
def test_choose_sampling_rule_picks_the_cheapest_not_clearly_worse() -> None:
    from failsense.models.compare_cutoffs import choose_sampling

    scores = {
        "k5": np.array([0.130, 0.140, 0.120, 0.135, 0.125]),
        "kall": np.array([0.140, 0.145, 0.131, 0.139, 0.133]),  # best by a little
        "matched": np.array([0.120, 0.130, 0.110, 0.125, 0.115]),  # clearly worse
    }
    rows = {"k5": 100, "kall": 1000, "matched": 100}
    # k5 is behind kall by ~0.008 with a difference spread of ~0.004 -> not qualifying
    assert choose_sampling(scores, rows) == "kall"
    close = {**scores, "k5": scores["kall"] + np.array([-0.002, 0.003, -0.003, 0.002, -0.001])}
    assert choose_sampling(close, rows) == "k5"  # within one std of the best, and cheaper
    assert choose_sampling({"only": scores["k5"]}, {"only": 5}) == "only"
    with pytest.raises(ValueError, match="no sampling"):
        choose_sampling({}, {})


def test_sampling_cutoffs_names(mini_vl_labels) -> None:
    from failsense.models.compare_cutoffs import sampling_cutoffs

    labels, targets = mini_vl_labels
    sizes = {
        n: len(sampling_cutoffs(n, labels, targets, 42))
        for n in ("k5", "k10", "kall", "ts70", "matched")
    }
    assert sizes["kall"] >= sizes["k10"] >= sizes["k5"] > 0
    assert (sampling_cutoffs("ts70", labels, targets, 42)["time_step"] >= 70).all()
    assert len(sampling_cutoffs("matched", labels, targets, 42)) > 0
    with pytest.raises(ValueError, match="unknown sampling"):
        sampling_cutoffs("k7", labels, targets, 42)


@pytest.fixture
def mini_vl_labels(make_mini_scania):
    from failsense.data.labels import build_labels
    from failsense.models.baseline_cv import read_table, table_path
    from failsense.models.eval_set import validation_time_steps

    config = make_mini_scania()
    readouts = read_table(table_path(config, "train", "readouts"))
    tte = read_table(table_path(config, "train", "tte"))
    return build_labels(readouts, tte, BOUNDS_VL), validation_time_steps(config)


BOUNDS_VL = (48.0, 24.0, 12.0, 6.0)


def test_validation_like_run_on_tiny_dataset(make_mini_scania, tmp_path) -> None:
    from failsense.models.compare_cutoffs import run_validation_like

    config = make_mini_scania()
    # (ts70 is left out here: on this tiny dataset only repaired trucks reach time_step 70,
    # so a training part would hold a single class; it is covered by test_sampling_cutoffs_names)
    out = run_validation_like(config, "xgboost", ["k5", "matched"], track=False)
    assert set(out["samplings"]) >= {"k5", "matched"}
    best = out["best_sampling"]
    assert best in {"k5", "matched"}
    assert set(out["rechecks"]) == {"no_time_step", "engineered"}
    assert f"{best}__engineered" in out["samplings"]
    assert set(out["old_vs_new"]) == {"xgboost", "logreg"}
    positives = out["eval_set"]["positives"]
    assert positives["overall"]["n"] == 40 and "repaired_share" in positives
    reports = tmp_path / "reports"
    for name in (
        "eval_set_q1_calibrated.csv",
        "eval_set_q1_positive_by_quartile_calibrated.csv",
        "eval_set_q1_positive_rate_calibrated.csv",
        "compare_sampling_q1_calibrated.csv",
        "eval_old_vs_new_q1_calibrated.csv",
    ):
        assert (reports / name).exists(), name
    assert (reports / "figures" / "eval_set_time_step_calibrated.png").exists()
    table = pd.read_csv(reports / "compare_sampling_q1_calibrated.csv")
    assert {"k5", "matched"} <= set(table["name"])


def test_validation_like_time_one_fold_projects_without_writing(make_mini_scania, tmp_path) -> None:
    from failsense.models.compare_cutoffs import run_validation_like

    out = run_validation_like(make_mini_scania(), "xgboost", ["k5", "kall"], time_one_fold=True)
    assert set(out) == {"k5", "kall"} and out["kall"]["rows"] >= out["k5"]["rows"]
    assert out["k5"]["projected_total_s"] >= out["k5"]["fold_s"]
    assert not list((tmp_path / "reports").glob("compare_sampling_q1*.csv"))


def test_validation_like_rejects_unknown_sampling(make_mini_scania) -> None:
    from failsense.models.compare_cutoffs import run_validation_like

    with pytest.raises(ValueError, match="unknown sampling"):
        run_validation_like(make_mini_scania(), "xgboost", ["k7"])


def test_default_mode_is_still_the_random_evaluation(monkeypatch) -> None:
    """``--eval`` defaults to the old behaviour: logistic regression on the random eval row."""
    from failsense.models import compare_cutoffs

    calls = {}
    monkeypatch.setattr(compare_cutoffs, "run", lambda config, ks, track=False: calls.update(ks=ks))
    monkeypatch.setattr(
        compare_cutoffs, "run_validation_like", lambda *a, **k: calls.update(vl=True)
    )
    compare_cutoffs.main(["--no-mlflow", "--ks", "1", "5"])
    assert calls == {"ks": [1, 5]}  # the validation-like path was not taken
    compare_cutoffs.main(["--no-mlflow", "--eval", "validation-like"])
    assert calls["vl"] is True


def test_eval_set_only_builds_every_method_and_writes_the_comparison(
    make_mini_scania, tmp_path
) -> None:
    from failsense.models.compare_cutoffs import eval_set_only

    config = make_mini_scania()  # the factory can be used once per test
    out = eval_set_only(config)
    assert set(out) == {"calibrated", "literal", "rank"}
    table = pd.read_csv(tmp_path / "reports" / "eval_set_methods_q1.csv")
    assert set(table["method"]) == {"calibrated", "literal", "rank"}
    assert {"ks_vs_validation", "positive_rate", "far_from_official"} <= set(table.columns)
    assert (tmp_path / "reports" / "figures" / "eval_set_time_step.png").exists()
    assert out["calibrated"]["info"]  # the calibration result is reported
    with pytest.raises(ValueError, match="unknown eval method"):
        eval_set_only(config, ["magic"])


def test_cli_eval_set_only_trains_nothing(monkeypatch) -> None:
    from failsense.models import compare_cutoffs

    calls = {}
    monkeypatch.setattr(
        compare_cutoffs, "eval_set_only", lambda config, methods: calls.update(m=methods)
    )
    monkeypatch.setattr(
        compare_cutoffs, "run_validation_like", lambda *a, **k: calls.update(run=True)
    )
    compare_cutoffs.main(["--eval", "validation-like", "--eval-set-only", "--eval-method", "all"])
    assert calls == {"m": compare_cutoffs.EVAL_METHODS}
    calls.clear()
    compare_cutoffs.main(["--eval", "validation-like", "--eval-set-only"])
    assert calls == {"m": ("calibrated",)}
    with pytest.raises(SystemExit):  # 'all' without --eval-set-only is refused
        compare_cutoffs.main(["--eval", "validation-like", "--eval-method", "all"])


def test_a_second_method_does_not_overwrite_the_first(make_mini_scania, tmp_path) -> None:
    from failsense.models.compare_cutoffs import run_validation_like

    config = make_mini_scania()
    run_validation_like(config, "xgboost", ["k5"], eval_method="calibrated")
    run_validation_like(config, "xgboost", ["k5"], eval_method="literal")
    reports = tmp_path / "reports"
    assert (reports / "compare_sampling_q1_calibrated.csv").exists()
    assert (reports / "compare_sampling_q1_literal.csv").exists()
