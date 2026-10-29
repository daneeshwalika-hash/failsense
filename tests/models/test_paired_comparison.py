"""Tests for failsense.models.baseline_cv.paired_comparison."""

from __future__ import annotations

import numpy as np
import pytest

from failsense.models.baseline_cv import COMPARISONS, paired_comparison


def test_paired_difference_and_wins() -> None:
    candidate = np.array([0.12, 0.10, 0.11, 0.09, 0.13])
    reference = np.array([0.10, 0.11, 0.10, 0.08, 0.12])
    result = paired_comparison(candidate, reference)
    assert result["mean_diff"] == pytest.approx(0.008)
    assert result["wins"] == 4
    assert result["n_folds"] == 5


def test_constant_fold_difficulty_cancels_out() -> None:
    # Folds of very different difficulty, candidate always +0.01 better:
    # the means differ only by 0.01, and the paired std is 0 (a clear signal).
    reference = np.array([0.05, 0.20, 0.10, 0.30, 0.15])
    result = paired_comparison(reference + 0.01, reference)
    assert result["mean_diff"] == pytest.approx(0.01)
    assert result["std_diff"] == pytest.approx(0.0, abs=1e-12)
    assert result["wins"] == 5


def test_ties_are_not_wins() -> None:
    scores = np.array([0.1, 0.2, 0.3])
    assert paired_comparison(scores, scores)["wins"] == 0


@pytest.mark.parametrize(
    ("candidate", "reference"),
    [([], []), ([0.1, 0.2], [0.1]), ([0.1], [0.1, 0.2])],
    ids=["empty", "candidate_longer", "reference_longer"],
)
def test_invalid_inputs_raise(candidate: list[float], reference: list[float]) -> None:
    with pytest.raises(ValueError, match="equal length"):
        paired_comparison(np.array(candidate), np.array(reference))


def test_comparisons_reference_existing_model_names() -> None:
    names = {"dummy_prior", "logreg_balanced", "logreg_specs_standard", "logreg_specs_minmax"}
    for candidate, reference in COMPARISONS:
        assert {candidate, reference} <= names
