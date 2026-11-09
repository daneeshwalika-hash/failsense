"""Anomaly detection helpers: detectors, cross-fitting (no truck in fit and score), metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from failsense.models import anomaly as an


def cloud(n: int = 400, n_out: int = 10, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Normal rows around 0 (+ a category) and a few far-away rows; returns (x, is_outlier)."""
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.normal(0, 1, size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    x["Spec_0"] = rng.choice(["Cat0", "Cat1"], n)
    outlier = np.zeros(n, dtype=bool)
    outlier[:n_out] = True
    x.loc[outlier, [f"f{i}" for i in range(6)]] += 8.0
    return x, outlier


# --- detectors ----------------------------------------------------------------------------
@pytest.mark.parametrize("kind", an.DETECTORS)
def test_every_detector_ranks_far_away_rows_as_more_anomalous(kind: str) -> None:
    x, outlier = cloud()
    healthy = x[~outlier]
    detector = an.make_detector(kind, ["Spec_0"], [], seed=0, n_estimators=50, n_components=4,
                                n_neighbors=10).fit(healthy)  # fmt: skip
    scores = an.anomaly_score(detector, x)
    assert scores[outlier].min() > np.percentile(scores[~outlier], 95)  # clearly separated
    assert isinstance(detector, Pipeline) and list(detector.named_steps)[0] == "preprocess"


def test_higher_score_means_more_anomalous_not_less() -> None:
    x, outlier = cloud()
    detector = an.make_detector("iforest_full", ["Spec_0"], [], 0, 50).fit(x[~outlier])
    raw = detector.score_samples(x)  # sklearn: higher = more normal
    np.testing.assert_allclose(an.anomaly_score(detector, x), -raw)
    assert (
        an.anomaly_score(detector, x)[outlier].mean()
        > an.anomaly_score(detector, x)[~outlier].mean()
    )


def test_detector_is_reproducible_and_rejects_unknown_kinds() -> None:
    x, outlier = cloud()
    a = an.make_detector("iforest_full", ["Spec_0"], [], 3, 30).fit(x[~outlier])
    b = an.make_detector("iforest_full", ["Spec_0"], [], 3, 30).fit(x[~outlier])
    np.testing.assert_array_equal(an.anomaly_score(a, x), an.anomaly_score(b, x))
    with pytest.raises(ValueError, match="kind must be"):
        an.make_detector("dbscan", [], [], 0)


def test_unseen_category_does_not_break_scoring() -> None:
    x, outlier = cloud()
    detector = an.make_detector("iforest_full", ["Spec_0"], [], 0, 30).fit(x[~outlier])
    new = x.iloc[:5].copy()
    new["Spec_0"] = "Cat999"
    assert np.isfinite(an.anomaly_score(detector, new)).all()


def test_healthy_mask() -> None:
    ids = pd.Series([1, 1, 2, 3, 3])
    assert an.healthy_mask(ids, [2]).tolist() == [True, True, False, True, True]
    assert an.healthy_mask(ids, []).all()


# --- cross-fitting: the leakage rule --------------------------------------------------------
class RecordingDetector:
    """Stands in for a Pipeline and enforces the rule: scoring rows of a truck it was fitted on."""

    groups: np.ndarray = np.array([])
    n_fits = 0
    scored: list[np.ndarray] = []

    def fit(self, x: pd.DataFrame) -> RecordingDetector:
        self.trucks_seen = set(self.groups[x.index.to_numpy()])
        self.fit_index = x.index.to_numpy()
        RecordingDetector.n_fits += 1
        return self

    def score_samples(self, x: pd.DataFrame) -> np.ndarray:
        trucks = set(self.groups[x.index.to_numpy()])
        assert trucks.isdisjoint(self.trucks_seen), "a truck was scored by a detector that saw it"
        RecordingDetector.scored.append(x.index.to_numpy())
        return -np.asarray(x["f0"], dtype=float)


def test_cross_fit_never_scores_a_row_with_a_detector_that_saw_its_truck() -> None:
    rng = np.random.default_rng(0)
    n_trucks, per_truck = 60, 5
    groups = np.repeat(np.arange(n_trucks), per_truck)
    x = pd.DataFrame({"f0": rng.normal(size=len(groups))})
    healthy = np.repeat(rng.random(n_trucks) > 0.2, per_truck)
    RecordingDetector.groups, RecordingDetector.n_fits, RecordingDetector.scored = groups, 0, []

    scores = an.cross_fit_anomaly(lambda: RecordingDetector(), x, healthy, groups, 5, seed=1)  # type: ignore[arg-type,return-value]

    assert np.isfinite(scores).all()  # every row is scored
    assert RecordingDetector.n_fits == 5
    scored_rows = np.concatenate(RecordingDetector.scored)
    assert sorted(scored_rows) == list(range(len(x)))  # exactly once each, incl. unhealthy trucks


def test_the_recording_detector_really_catches_a_violation() -> None:
    """Guards the guard: scoring a seen truck must fail the assertion."""
    groups = np.array([0, 0, 1, 1])
    RecordingDetector.groups = groups
    x = pd.DataFrame({"f0": [1.0, 2.0, 3.0, 4.0]})
    detector = RecordingDetector().fit(x.iloc[:2])
    with pytest.raises(AssertionError, match="saw it"):
        detector.score_samples(x.iloc[[1, 2]])  # row 1 belongs to a fitted truck


def test_cross_fit_validates_inputs() -> None:
    x = pd.DataFrame({"f0": np.arange(10.0)})
    with pytest.raises(ValueError, match="same length"):
        an.cross_fit_anomaly(lambda: RecordingDetector(), x, np.ones(9, bool), np.arange(10), 2, 0)  # type: ignore[arg-type,return-value]


def test_cross_fit_with_a_real_detector_scores_outliers_higher() -> None:
    x, outlier = cloud(n=600)
    healthy = ~outlier
    groups = np.arange(len(x)) // 3  # 3 rows per truck; outliers sit in the first trucks
    scores = an.cross_fit_anomaly(
        lambda: an.make_detector("iforest_full", ["Spec_0"], [], 0, 40), x, healthy, groups, 4, 0
    )
    assert np.isfinite(scores).all() and scores[outlier].mean() > scores[healthy].mean()


# --- metrics ----------------------------------------------------------------------------------
def test_evaluate_scores_perfect_random_and_reversed() -> None:
    y = np.array([0] * 90 + [1] * 10)
    perfect = np.r_[np.zeros(90), np.ones(10)]
    metrics = an.evaluate_scores(y, perfect, 0.1, 100, 1)
    assert metrics["pr_auc"] == 1.0 and metrics["roc_auc"] == 1.0
    assert metrics["recall_top_10%"] == 1.0 and metrics["lift"] == pytest.approx(10.0)
    assert metrics["pr_auc_ci_low"] <= metrics["pr_auc"] <= metrics["pr_auc_ci_high"]
    reversed_ = an.evaluate_scores(y, -perfect, 0.1, 100, 1)
    assert reversed_["roc_auc"] == 0.0 and reversed_["recall_top_10%"] == 0.0
    rng = np.random.default_rng(0)
    noise = an.evaluate_scores(np.tile(y, 20), rng.random(2000), 0.05, 100, 1)
    assert noise["roc_auc"] == pytest.approx(0.5, abs=0.06) and 0.5 < noise["lift"] < 1.6


@pytest.mark.parametrize("y", [[0, 0, 0], [1, 1, 1]])
def test_evaluate_scores_needs_both_classes(y: list[int]) -> None:
    with pytest.raises(ValueError, match="both positive and negative"):
        an.evaluate_scores(y, [0.1, 0.2, 0.3], 0.5, 10, 0)


def test_evaluate_scores_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="equal length"):
        an.evaluate_scores([0, 1], [0.1], 0.5, 10, 0)


def test_paired_bootstrap_detects_a_better_score_and_adoption_rule() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(1500) < 0.1).astype(int)
    good = y + rng.normal(0, 0.6, 1500)
    bad = rng.normal(size=1500)
    better = an.paired_bootstrap_pr_auc(y, good, bad, 200, 1)
    assert better["diff"] > 0.2 and better["share_a_better"] > 0.99
    assert an.adoption_decision(better) is True
    worse = an.paired_bootstrap_pr_auc(y, bad, good, 200, 1)
    assert worse["diff"] < 0 and an.adoption_decision(worse) is False
    same = an.paired_bootstrap_pr_auc(y, good, good, 100, 1)
    assert same["diff"] == 0 and an.adoption_decision(same) is False  # not strictly positive


@pytest.mark.parametrize(
    "paired, expected",
    [({"diff": 0.01, "std": 0.02}, False),  # positive but inside the noise
     ({"diff": 0.03, "std": 0.02}, True),
     ({"diff": -0.03, "std": 0.02}, False),
     ({"diff": 0.0, "std": 0.0}, False)],
)  # fmt: skip
def test_adoption_rule_follows_adr_0005(paired: dict[str, Any], expected: bool) -> None:
    assert an.adoption_decision(paired) is expected


def test_paired_bootstrap_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="equal length"):
        an.paired_bootstrap_pr_auc([0, 1], [0.1, 0.2], [0.1], 10, 0)
