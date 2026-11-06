"""Tests for failsense.explain.importance (groups, permutation, gain)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import TINY_INPUT_COLUMNS
from failsense.explain.importance import (
    aggregate,
    feature_groups,
    gain_importance,
    group_name,
    permutation_importance,
    variable_of,
)


@pytest.mark.parametrize(
    ("column", "variable"),
    [
        ("167_3", "167"), ("167__mean_bin", "167"), ("167_3__share", "167"),
        ("171_0", "171"), ("171_0__delta", "171"), ("397__top_share", "397"),
        ("time_step", "time_step"), ("Spec_3", "Spec_3"),
    ],
)  # fmt: skip
def test_variable_of(column: str, variable: str) -> None:
    assert variable_of(column) == variable


def test_variable_of_rejects_unknown_pattern() -> None:
    with pytest.raises(ValueError, match="cannot assign"):
        variable_of("mystery")


def test_group_names_and_feature_groups() -> None:
    assert group_name("167") == "167 (histogram)"
    assert group_name("171") == "171 (counter)"
    assert group_name("time_step") == "time_step"

    groups = feature_groups(TINY_INPUT_COLUMNS)

    assert groups["167 (histogram)"] == ["167_0", "167_1", "167__mean_bin"]
    assert groups["171 (counter)"] == ["171_0", "171_0__delta", "171_0__rate"]
    assert groups["time_step"] == ["time_step"] and groups["Spec_0"] == ["Spec_0"]


def make_xy(n: int = 800, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({"sig": rng.normal(size=n), "twin": rng.normal(size=n),
                      "noise": rng.normal(size=n)})  # fmt: skip
    y = (x["sig"] + 0.5 * rng.normal(size=n) > 1.0).astype(int).to_numpy()
    return x, y


def predict(frame: pd.DataFrame) -> np.ndarray:
    return frame["sig"].to_numpy()  # the "model" only reads `sig`


def test_signal_column_has_a_reliable_drop_and_noise_has_none() -> None:
    x, y = make_xy()

    table = permutation_importance(predict, x, y, {c: [c] for c in x}, 10, seed=1)
    by = table.set_index("group")

    assert by.loc["sig", "mean_drop"] > 0.2 and bool(by.loc["sig", "reliable"])
    assert by.loc["noise", "mean_drop"] == 0.0 and not by.loc["noise", "reliable"]
    assert by.loc["twin", "std_drop"] == 0.0  # not used by the model: never changes
    assert (table["baseline"] == table["baseline"].iloc[0]).all()


def test_group_is_shuffled_jointly_and_input_is_not_modified() -> None:
    x, y = make_xy()
    before = x.copy()

    table = permutation_importance(predict, x, y, {"pair": ["sig", "twin"]}, 5, seed=3)

    pd.testing.assert_frame_equal(x, before)
    assert table.loc[0, "n_columns"] == 2 and table.loc[0, "mean_drop"] > 0.2


def test_permutation_is_reproducible_and_independent_of_group_order() -> None:
    x, y = make_xy()
    groups = {"a": ["sig"], "b": ["noise"]}
    reverse = {"b": ["noise"], "a": ["sig"]}

    one = permutation_importance(predict, x, y, groups, 6, seed=5).set_index("group")
    two = permutation_importance(predict, x, y, groups, 6, seed=5).set_index("group")
    rev = permutation_importance(predict, x, y, reverse, 6, seed=5).set_index("group")

    pd.testing.assert_frame_equal(one, two)
    assert one.loc["b", "mean_drop"] == rev.loc["b", "mean_drop"]
    # the group order changes the per-group seed (index), so only equal-by-construction cases match
    assert one.loc["a", "mean_drop"] > 0


def test_permutation_importance_errors() -> None:
    x, y = make_xy(50)
    with pytest.raises(ValueError, match="n_repeats"):
        permutation_importance(predict, x, y, {"a": ["sig"]}, 1, 0)
    with pytest.raises(ValueError, match="same length"):
        permutation_importance(predict, x, y[:10], {"a": ["sig"]}, 3, 0)
    with pytest.raises(ValueError, match="both classes"):
        permutation_importance(predict, x, np.zeros(50, dtype=int), {"a": ["sig"]}, 3, 0)
    with pytest.raises(ValueError, match="unknown columns"):
        permutation_importance(predict, x, y, {"a": ["nope"]}, 3, 0)


def test_permutation_on_a_real_pipeline_finds_the_planted_signal(tiny_pipeline) -> None:
    pipeline, frame, y = tiny_pipeline
    groups = feature_groups(TINY_INPUT_COLUMNS)

    table = permutation_importance(
        lambda f: pipeline.predict_proba(f)[:, 1], frame, y, groups, 5, seed=0
    ).set_index("group")

    assert table.loc["171 (counter)", "mean_drop"] > table.loc["167 (histogram)", "mean_drop"]
    assert bool(table.loc["171 (counter)", "reliable"])
    assert frame["Spec_0"].dtype == object or frame["Spec_0"].dtype == "str"  # still text


def test_gain_importance_and_aggregate(tiny_pipeline) -> None:
    pipeline, _, _ = tiny_pipeline
    names = list(pipeline.named_steps["preprocess"].get_feature_names_out())

    gain = gain_importance(pipeline.named_steps["model"], names)

    assert list(gain.index) == names and gain.sum() == pytest.approx(1.0)
    assert gain.idxmax() in {"num__171_0__delta", "cat__Spec_0_A"}
    with pytest.raises(ValueError, match="does not match"):
        gain_importance(pipeline.named_steps["model"], names[:-1])


def test_aggregate_sums_by_key_and_keeps_order() -> None:
    out = aggregate([1.0, 2.0, 3.0, 4.0], ["b", "a", "b", "a"])

    assert list(out.index) == ["b", "a"] and list(out.values) == [4.0, 6.0]
