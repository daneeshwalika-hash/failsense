"""Tests for failsense.models.evaluate (hand-made examples and the tiny dataset)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from failsense.models.boosters import STEP6_PARAMS
from failsense.models.evaluate import (
    PROVISIONAL,
    UNTUNED,
    WINNER_FILE,
    choose_winner,
    fold_metrics,
    run,
    top_k_metrics,
)
from failsense.models.tune import BEST_PARAMS_FILE

# The tiny fixture has ~1 positive evaluation row, so some folds have none and scikit-learn warns
# that average precision is undefined there; on the real data every fold has positives.
pytestmark = pytest.mark.filterwarnings("ignore:No positive class found:UserWarning")

Y = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
SCORE = [0.9, 0.8, 0.1, 0.7, 0.2, 0.3, 0.05, 0.04, 0.03, 0.02]


def test_top_k_metrics_on_a_hand_made_example() -> None:
    top20 = top_k_metrics(Y, SCORE, fraction=0.2)  # alerts the 2 riskiest: rows 0 and 1
    assert (top20["alerts"], top20["tp"], top20["fp"], top20["fn"], top20["tn"]) == (2, 1, 1, 1, 7)
    assert top20["precision"] == 0.5 and top20["recall"] == 0.5 and top20["f1"] == 0.5
    top30 = top_k_metrics(Y, SCORE, fraction=0.3)  # rows 0, 1 and 3
    assert (top30["tp"], top30["fp"], top30["fn"], top30["tn"]) == (2, 1, 0, 7)
    assert top30["precision"] == pytest.approx(2 / 3) and top30["recall"] == 1.0
    assert top30["f1"] == pytest.approx(0.8)


def test_top_k_rounds_up_and_handles_edge_cases() -> None:
    assert top_k_metrics(Y, SCORE, fraction=0.05)["alerts"] == 1  # ceil(0.5) = 1, never 0
    everyone = top_k_metrics(Y, SCORE, fraction=1.0)
    assert everyone["tp"] == 2 and everyone["fn"] == 0 and everyone["recall"] == 1.0
    none_positive = top_k_metrics([0] * 10, SCORE, 0.2)
    assert none_positive["recall"] == 0.0 and none_positive["f1"] == 0.0
    ties = top_k_metrics([1, 0, 0, 0], [0.5, 0.5, 0.5, 0.5], 0.25)  # stable: the first row wins
    assert ties["tp"] == 1
    for bad in (0.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="fraction"):
            top_k_metrics(Y, SCORE, bad)
    with pytest.raises(ValueError, match="equal length"):
        top_k_metrics([1, 0], [0.1], 0.5)
    with pytest.raises(ValueError, match="non-empty"):
        top_k_metrics([], [], 0.5)


def test_fold_metrics_keys_and_ranges() -> None:
    m = fold_metrics(np.array(Y), np.array(SCORE), 0.2)
    assert {
        "pr_auc",
        "roc_auc",
        "brier",
        "mean_proba",
        "top_precision",
        "top_recall",
        "top_f1",
    } <= set(m)
    assert m["mean_proba"] == pytest.approx(np.mean(SCORE))
    assert 0 <= m["brier"] <= 1 and m["tp"] + m["fp"] + m["fn"] + m["tn"] == 10


def cmp(mean: float, std: float, wins: int) -> dict[str, float]:
    return {"mean_diff": mean, "std_diff": std, "wins": wins, "n_folds": 5}


def test_winner_rule_needs_both_conditions() -> None:
    means = {"xgboost_tuned": 0.07, "lightgbm_tuned": 0.065, UNTUNED: 0.06}
    clear = choose_winner(
        {"xgboost_tuned": cmp(0.010, 0.004, 5), "lightgbm_tuned": cmp(0.005, 0.004, 4)}, means
    )
    assert clear["winner"] == "xgboost_tuned" and clear["qualifying"] == [
        "xgboost_tuned",
        "lightgbm_tuned",
    ]
    # mean difference not larger than its std -> not clearly better
    noisy = choose_winner(
        {"xgboost_tuned": cmp(0.004, 0.006, 5), "lightgbm_tuned": cmp(0.003, 0.005, 5)}, means
    )
    assert noisy["winner"] == UNTUNED and noisy["qualifying"] == []
    # better in only 3 of 5 folds -> not clearly better
    few = choose_winner(
        {"xgboost_tuned": cmp(0.010, 0.002, 3), "lightgbm_tuned": cmp(0.010, 0.002, 3)}, means
    )
    assert few["winner"] == UNTUNED and "simpler untuned" in few["reason"]
    # of two qualifying models the one with the higher mean score wins
    both = choose_winner(
        {"xgboost_tuned": cmp(0.010, 0.002, 5), "lightgbm_tuned": cmp(0.011, 0.002, 5)},
        {"xgboost_tuned": 0.07, "lightgbm_tuned": 0.071, UNTUNED: 0.06},
    )
    assert both["winner"] == "lightgbm_tuned"


def test_evaluate_on_the_tiny_dataset(make_mini_scania, tmp_path: Path) -> None:
    config = make_mini_scania()
    reports = tmp_path / "reports"
    reports.mkdir()
    entry = {
        "params": STEP6_PARAMS["xgboost"], "engineered": True, "n_rounds": 10, "best_value": 0.1,
        "n_trials": 3, "n_complete": 3, "n_pruned": 0, "n_trials_target": 3, "timed_out": False,
        "timeout_minutes": 1.0, "seed": 42, "best_rounds_per_fold": [10], "fold_scores": [0.1],
        "anchor_value": 0.1,
    }  # fmt: skip
    (reports / BEST_PARAMS_FILE).write_text(
        json.dumps({"xgboost": entry, "lightgbm": {**entry, "params": STEP6_PARAMS["lightgbm"]}})
    )
    out = run(config, track=False)
    summary = out["summary"]
    assert set(summary["eval_set"]) == {"calibrated", "literal"}
    assert set(summary["model"]) == {"xgboost_tuned", "lightgbm_tuned", UNTUNED}
    assert (summary["brier"].between(0, 1)).all() and (summary["pr_auc"].between(0, 1)).all()
    confusion = pd.read_csv(reports / "evaluate_q1_confusion.csv")
    assert (confusion["status"] == PROVISIONAL).all()
    assert (confusion[["tp", "fp", "fn", "tn"]].sum(axis=1) == 40).all()  # every truck once
    paired = pd.read_csv(reports / "evaluate_q1_paired.csv")
    assert set(paired["reference"]) == {UNTUNED} and len(paired) == 4  # 2 tuned models x 2 sets
    winner = json.loads((reports / WINNER_FILE).read_text())
    assert winner["winner"] in {"xgboost_tuned", "lightgbm_tuned", UNTUNED}
    assert {"family", "params", "engineered", "n_rounds", "reason"} <= set(winner)


def test_evaluate_needs_both_tuning_results(make_mini_scania, tmp_path: Path) -> None:
    config = make_mini_scania()
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / BEST_PARAMS_FILE).write_text(json.dumps({"xgboost": {}}))
    with pytest.raises(FileNotFoundError, match="make tune"):
        run(config)
