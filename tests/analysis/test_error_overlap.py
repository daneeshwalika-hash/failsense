"""Tests for failsense.analysis.error_overlap (pure functions, small hand-made data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from failsense.analysis.error_overlap import (
    cost_optimal_alerts,
    error_sets,
    found_by_table,
    pair_overlap,
)

# 8 trucks: classes 0,0,0,1,2,3,4,4
CLASSES = np.array([0, 0, 0, 1, 2, 3, 4, 4])
A = np.array([1, 0, 0, 1, 0, 1, 0, 0], dtype=bool)  # FP: 0; FN: 4,6,7 (idx of class 2,4,4)
B = np.array([0, 1, 0, 0, 1, 1, 1, 0], dtype=bool)  # FP: 1; FN: 3,7


def test_error_sets() -> None:
    sets = error_sets(CLASSES, A)

    np.testing.assert_array_equal(sets["fn"], [4, 6, 7])
    np.testing.assert_array_equal(sets["fp"], [0])


def test_pair_overlap_counts_and_jaccard() -> None:
    table = pair_overlap(CLASSES, {"a": A, "b": B}).set_index(["error", "group"])

    assert table.loc[("fn", "a"), "n"] == 3
    assert table.loc[("fn", "b"), "n"] == 2
    assert table.loc[("fn", "a & b"), "n"] == 1  # only truck 7 is missed by both
    assert table.loc[("fn", "a & b"), "jaccard"] == pytest.approx(1 / 4)  # union {3,4,6,7}
    assert table.loc[("fp", "a & b"), "n"] == 0
    assert ("fn", "all") not in table.index  # "all" only for more than two models


def test_found_by_table_partitions_every_failure() -> None:
    c = {"a": A, "b": B}
    table = found_by_table(CLASSES, c).set_index("class")

    assert table.loc["total", "n"] == 5
    # class 1 (idx 3): a only; class 2 (idx 4): b only; class 3 (idx 5): both; class 4: b/none
    assert table.loc["1", "only_a"] == 1
    assert table.loc["2", "only_b"] == 1
    assert table.loc["3", "all"] == 1
    assert table.loc["4", "none"] == 1 and table.loc["4", "only_b"] == 1
    columns = ["all", "only_a", "only_b", "none", "other_combination"]
    np.testing.assert_array_equal(table[columns].sum(axis=1), table["n"])


def test_found_by_table_three_models_has_other_combination() -> None:
    c = {"a": A, "b": B, "c": A | B}
    table = found_by_table(CLASSES, c).set_index("class")

    assert table.loc["3", "all"] == 1
    assert table.loc["1", "other_combination"] == 1  # idx 3 is found by a and c


def test_length_mismatch_and_empty_input_raise() -> None:
    with pytest.raises(ValueError):
        error_sets(CLASSES, A[:3])
    with pytest.raises(ValueError):
        pair_overlap(CLASSES, {"a": A[:3]})
    with pytest.raises(ValueError):
        found_by_table(CLASSES, {})


def test_empty_data() -> None:
    empty = np.array([], dtype=int)

    table = found_by_table(empty, {"a": empty.astype(bool)})

    assert table.loc[table["class"] == "total", "n"].item() == 0
    assert isinstance(pair_overlap(empty, {"a": empty.astype(bool)}), pd.DataFrame)


def test_cost_optimal_alerts_uses_own_threshold_per_model() -> None:
    config = {"cost": {"matrix": [[0, 7, 8, 9, 10], [200, 0, 7, 8, 9], [300, 200, 0, 7, 8],
                                  [400, 300, 200, 0, 7], [500, 400, 300, 200, 0]],
                       "binary": {"alert_class": 1}}}  # fmt: skip
    scores = {"good": np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9, 0.95]), "bad": np.zeros(8) + 0.5}

    alerts, info = cost_optimal_alerts(CLASSES, scores, config)

    assert alerts["good"][3:].all() and not alerts["good"][:3].any()
    assert info["good"]["cost"] == 0
    assert alerts["bad"].all() or not alerts["bad"].any()  # a constant score cannot separate
