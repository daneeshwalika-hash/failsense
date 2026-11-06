"""Tests for failsense.models.cost (matrix from config, binary mappings, total cost)."""

from __future__ import annotations

import numpy as np
import pytest

from failsense.config import load_config
from failsense.models import cost
from failsense.models.cost import (
    binary_costs,
    class_mix,
    cost_matrix_from_config,
    per_class_thresholds,
    theoretical_threshold,
    total_cost,
)

# Table 1 of the IDA 2024 challenge description (2024_IDA_challenge_v2.pdf), typed in again
# on purpose: an independent copy to compare the config against.
PDF_MATRIX = np.array(
    [
        [0, 7, 8, 9, 10],
        [200, 0, 7, 8, 9],
        [300, 200, 0, 7, 8],
        [400, 300, 200, 0, 7],
        [500, 400, 300, 200, 0],
    ],
    dtype=float,
)


def config_with(matrix: object, alert_class: object = 1) -> dict:
    return {"cost": {"matrix": matrix, "binary": {"alert_class": alert_class}}}


def test_config_matrix_matches_the_challenge_pdf() -> None:
    matrix = cost_matrix_from_config(load_config())

    assert matrix.shape == (5, 5)
    assert not np.diagonal(matrix).any()
    np.testing.assert_array_equal(matrix, PDF_MATRIX)


def test_config_alert_class_is_one() -> None:
    assert cost.alert_class_from_config(load_config()) == 1


@pytest.mark.parametrize(
    "bad",
    [
        [[0, 1], [1, 0]],  # wrong shape
        [[0, 1, 2, 3, 4]] * 5,  # non-zero diagonal in rows 1..4
        [
            [0, 7, 8, 9, 10],
            [200, 0, 7, 8, 9],
            [300, 200, 0, 7, 8],
            [400, 300, 200, 0, 7],
            [500, 400, 300, 200, -1],
        ],  # fmt: skip
        [[0, "a", 8, 9, 10]] + [[0] * 5] * 4,  # not numeric
    ],
)
def test_matrix_validation_rejects_bad_matrices(bad: object) -> None:
    with pytest.raises(ValueError):
        cost_matrix_from_config(config_with(bad))


def test_matrix_validation_rejects_nan() -> None:
    bad = PDF_MATRIX.copy()
    bad[1, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        cost_matrix_from_config(config_with(bad.tolist()))


@pytest.mark.parametrize("alert_class", [0, 5, 1.5, True, "1"])
def test_alert_class_validation(alert_class: object) -> None:
    with pytest.raises(ValueError):
        cost.alert_class_from_config(config_with(PDF_MATRIX.tolist(), alert_class))


def test_primary_binary_costs() -> None:
    costs = binary_costs(PDF_MATRIX, alert_class=1, mapping="primary")

    assert costs.fp == 7
    np.testing.assert_array_equal(costs.fn, [200, 300, 400, 500])
    np.testing.assert_array_equal(costs.tp, [0, 0, 0, 0])
    np.testing.assert_array_equal(costs.benefit, [200, 300, 400, 500])


def test_literal_binary_costs() -> None:
    costs = binary_costs(PDF_MATRIX, alert_class=1, mapping="literal")

    np.testing.assert_array_equal(costs.tp, [0, 200, 300, 400])  # Cost_k_1
    np.testing.assert_array_equal(costs.benefit, [200, 100, 100, 100])


def test_binary_costs_other_alert_class_changes_fp() -> None:
    assert binary_costs(PDF_MATRIX, alert_class=3).fp == 9


def test_binary_costs_reject_bad_arguments() -> None:
    with pytest.raises(ValueError, match="mapping"):
        binary_costs(PDF_MATRIX, 1, mapping="other")
    with pytest.raises(ValueError, match="alert_class"):
        binary_costs(PDF_MATRIX, 0)


# Hand-computed example: classes [0, 0, 1, 2, 4, 4], alert [T, F, T, F, F, T].
CLASSES = np.array([0, 0, 1, 2, 4, 4])
ALERT = np.array([True, False, True, False, False, True])


def test_total_cost_primary_hand_computed() -> None:
    # false alarm 7 + (class 1 caught 0) + (class 2 missed 300) + (class 4 missed 500)
    # + (class 4 caught 0) = 807
    costs = binary_costs(PDF_MATRIX, 1, "primary")

    assert total_cost(CLASSES, ALERT, costs) == 807


def test_total_cost_literal_hand_computed() -> None:
    # 807 + class 4 alerted as "class 1" costs Cost_4_1 = 400; class 1 alerted costs 0 -> 1207
    costs = binary_costs(PDF_MATRIX, 1, "literal")

    assert total_cost(CLASSES, ALERT, costs) == 1207


def test_total_cost_never_and_always_alert() -> None:
    costs = binary_costs(PDF_MATRIX, 1, "primary")

    assert total_cost(CLASSES, np.zeros(6, dtype=bool), costs) == 200 + 300 + 500 + 500
    assert total_cost(CLASSES, np.ones(6, dtype=bool), costs) == 14


def test_total_cost_validation_baselines_from_the_official_class_counts() -> None:
    # Class counts of the official validation split (documentation numbers of 06.11).
    classes = np.repeat([0, 1, 2, 3, 4], [4910, 16, 14, 30, 76])
    costs = binary_costs(PDF_MATRIX, 1, "primary")

    assert total_cost(classes, np.ones(len(classes), dtype=bool), costs) == 34_370
    assert total_cost(classes, np.zeros(len(classes), dtype=bool), costs) == 57_400


def test_total_cost_empty_is_zero() -> None:
    costs = binary_costs(PDF_MATRIX, 1)

    assert total_cost(np.array([], dtype=int), np.array([], dtype=bool), costs) == 0.0


def test_total_cost_rejects_bad_input() -> None:
    costs = binary_costs(PDF_MATRIX, 1)
    with pytest.raises(ValueError, match="same length"):
        total_cost(np.array([0, 1]), np.array([True]), costs)
    with pytest.raises(ValueError, match="boolean"):
        total_cost(np.array([0, 1]), np.array([0, 1]), costs)
    with pytest.raises(ValueError, match="0..4"):
        total_cost(np.array([0, 5]), np.array([True, False]), costs)


def test_class_mix() -> None:
    np.testing.assert_allclose(class_mix([0, 0, 1, 1, 2, 4]), [0.5, 0.25, 0.0, 0.25])


def test_class_mix_without_positives_raises() -> None:
    with pytest.raises(ValueError, match="no positive"):
        class_mix([0, 0, 0])
    with pytest.raises(ValueError, match="no positive"):
        class_mix([])


def test_class_mix_rejects_unknown_class() -> None:
    with pytest.raises(ValueError, match="0..4"):
        class_mix([0, 7])


def test_theoretical_threshold_primary() -> None:
    costs = binary_costs(PDF_MATRIX, 1, "primary")

    assert theoretical_threshold(costs, [0, 0, 0, 1]) == pytest.approx(7 / 507)
    assert theoretical_threshold(costs, [0.25] * 4) == pytest.approx(7 / 357)
    np.testing.assert_allclose(per_class_thresholds(costs), [7 / 207, 7 / 307, 7 / 407, 7 / 507])


def test_theoretical_threshold_literal_is_higher() -> None:
    primary = binary_costs(PDF_MATRIX, 1, "primary")
    literal = binary_costs(PDF_MATRIX, 1, "literal")

    assert theoretical_threshold(literal, [0, 0, 0, 1]) == pytest.approx(7 / 107)
    assert theoretical_threshold(literal, [0.25] * 4) > theoretical_threshold(primary, [0.25] * 4)


def test_theoretical_threshold_rejects_bad_mix() -> None:
    costs = binary_costs(PDF_MATRIX, 1)
    with pytest.raises(ValueError, match="mix"):
        theoretical_threshold(costs, [0.5, 0.5])
    with pytest.raises(ValueError, match="mix"):
        theoretical_threshold(costs, [0.5, 0.5, 0.5, 0.5])
