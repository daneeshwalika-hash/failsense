"""Tests for failsense.analysis.unsupervised (mini fixtures, no raw data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from failsense.analysis import unsupervised as uns
from failsense.data.labels import (
    CLASS_COL,
    LABEL_COLUMNS,
    REPAIR_COL,
    TIME_COL,
    VEHICLE_COL,
)
from failsense.data.schema import SPEC_COLUMNS, counter_columns, histogram_groups

ARGS = ((48, 24, 12, 6), counter_columns(), histogram_groups(), 0.2)


def table(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return uns.truck_table(frames["readouts"], frames["specs"], frames["tte"], *ARGS)


def blobs(n: int = 60, k: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centres = np.array([[0, 0], [10, 0], [0, 10]])[:k]
    return np.vstack([rng.normal(c, 0.5, size=(n, 2)) for c in centres])


# --- one row per truck ----------------------------------------------------------------------
def test_one_row_per_truck_at_its_last_readout(mini_frames: dict[str, pd.DataFrame]) -> None:
    features, truth = table(mini_frames)
    assert list(features.index) == list(range(40)) == list(truth.index)
    assert (features[TIME_COL] == 110.0).all()  # the last of the 12 readouts
    assert features["171_0__delta"].notna().all()  # deltas come from the previous readout


def test_last_readout_is_found_in_shuffled_readouts(mini_frames: dict[str, pd.DataFrame]) -> None:
    shuffled = {**mini_frames, "readouts": mini_frames["readouts"].sample(frac=1, random_state=1)}
    features, _ = table(shuffled)
    expected, _ = table(mini_frames)
    pd.testing.assert_frame_equal(features, expected)


def test_labels_are_separate_and_never_in_the_features(
    mini_frames: dict[str, pd.DataFrame],
) -> None:
    features, truth = table(mini_frames)
    assert not set(features.columns) & (LABEL_COLUMNS | {VEHICLE_COL})
    assert list(truth.columns) == [REPAIR_COL, CLASS_COL]
    assert truth[REPAIR_COL].to_dict() == {v: int(v % 4 == 0) for v in range(40)}
    assert not set(uns.select_inputs(features, None).columns) & LABEL_COLUMNS


def test_truck_table_rejects_bad_input(mini_frames: dict[str, pd.DataFrame]) -> None:
    with pytest.raises(ValueError, match="empty"):
        uns.truck_table(mini_frames["readouts"].iloc[0:0], mini_frames["specs"],
                        mini_frames["tte"], *ARGS)  # fmt: skip
    with pytest.raises(ValueError, match="no tte row"):
        uns.truck_table(mini_frames["readouts"], mini_frames["specs"],
                        mini_frames["tte"].iloc[1:], *ARGS)  # fmt: skip
    with pytest.raises(ValueError, match="no specification row"):
        uns.truck_table(mini_frames["readouts"], mini_frames["specs"].iloc[1:],
                        mini_frames["tte"], *ARGS)  # fmt: skip


def test_select_inputs_keeps_order_and_refuses_labels_and_missing(
    mini_frames: dict[str, pd.DataFrame],
) -> None:
    features, _ = table(mini_frames)
    assert list(uns.select_inputs(features, ["167_0", "171_0"]).columns) == ["167_0", "171_0"]
    with pytest.raises(ValueError, match="forbidden|not allowed|label"):
        uns.select_inputs(features.assign(**{REPAIR_COL: 1}), ["167_0", REPAIR_COL])
    with pytest.raises(ValueError, match="missing"):
        uns.select_inputs(features, ["no_such_column"])


# --- PCA ------------------------------------------------------------------------------------
@pytest.mark.parametrize("log1p", [True, False])
def test_embedding_pipeline_projects_every_truck(
    mini_frames: dict[str, pd.DataFrame], log1p: bool
) -> None:
    features, _ = table(mini_frames)
    pipe = uns.build_embedding_pipeline(features, n_components=3, seed=0, log1p=log1p)
    z = pipe.fit_transform(features)
    assert z.shape == (40, 3) and np.isfinite(z).all()


def test_variance_table_is_cumulative_and_sorted(mini_frames: dict[str, pd.DataFrame]) -> None:
    features, _ = table(mini_frames)
    pipe = uns.build_embedding_pipeline(features, n_components=None, seed=0).fit(features)
    table_ = uns.variance_table(pipe.named_steps["pca"])
    assert table_["explained"].is_monotonic_decreasing
    assert table_["cumulative"].iloc[-1] == pytest.approx(1.0)
    assert table_["component"].iloc[0] == 1


def test_top_loadings_returns_the_largest_weights(mini_frames: dict[str, pd.DataFrame]) -> None:
    features, _ = table(mini_frames)
    pipe = uns.build_embedding_pipeline(features, n_components=2, seed=0).fit(features)
    top = uns.top_loadings(pipe, component=0, n=5)
    weights = pipe.named_steps["pca"].components_[0]
    assert len(top) == 5
    assert top["weight"].abs().max() == pytest.approx(np.abs(weights).max())
    assert top["weight"].abs().is_monotonic_decreasing


# --- KMeans ---------------------------------------------------------------------------------
def test_kmeans_scan_finds_the_three_blobs() -> None:
    scan = uns.kmeans_scan(blobs(), [2, 3, 4, 5], sample_size=100, seed=0)
    assert list(scan["k"]) == [2, 3, 4, 5]
    assert uns.best_k(scan) == 3
    assert scan["silhouette"].between(-1, 1).all()
    assert scan["inertia"].is_monotonic_decreasing


def test_sample_size_larger_than_data_is_capped() -> None:
    scan = uns.kmeans_scan(blobs(n=10), [3], sample_size=10_000, seed=0)
    assert scan.loc[0, "silhouette"] > 0.5


def test_kmeans_scan_is_reproducible() -> None:
    a = uns.kmeans_scan(blobs(), [3], 50, seed=7)
    b = uns.kmeans_scan(blobs(), [3], 50, seed=7)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("ks", [[1], [2, 0], [500]])
def test_kmeans_scan_rejects_invalid_k(ks: list[int]) -> None:
    with pytest.raises(ValueError, match="every k"):
        uns.kmeans_scan(blobs(n=5), ks, 10, seed=0)


def test_kmeans_scan_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="2-D"):
        uns.kmeans_scan(np.arange(10.0), [2], 10, seed=0)
    with pytest.raises(ValueError, match="2-D"):
        uns.kmeans_scan(np.empty((0, 2)), [2], 10, seed=0)


def test_best_k_prefers_the_smaller_k_on_ties() -> None:
    scan = pd.DataFrame({"k": [2, 3], "inertia": [2.0, 1.0], "silhouette": [0.3, 0.3]})
    assert uns.best_k(scan) == 2


# --- cluster summary ------------------------------------------------------------------------
def test_cluster_summary_matches_a_hand_computation() -> None:
    clusters = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    truth = pd.DataFrame({REPAIR_COL: [1, 1, 0, 0, 0, 0, 0, 1]})
    summary = uns.cluster_summary(clusters, truth).set_index("cluster")
    assert summary.loc[0, ["n_trucks", "n_repaired"]].tolist() == [4, 2]
    assert summary.loc[0, "repair_rate"] == 0.5 and summary.loc[1, "repair_rate"] == 0.25
    assert summary.loc[0, "lift"] == pytest.approx(0.5 / 0.375)
    assert (summary["ci_low"] <= summary["repair_rate"]).all()
    assert (summary["repair_rate"] <= summary["ci_high"]).all()


def test_cluster_summary_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        uns.cluster_summary(np.array([0, 1]), pd.DataFrame({REPAIR_COL: [1]}))


def test_agreement_is_one_for_identical_and_near_zero_for_independent() -> None:
    truth = pd.DataFrame({REPAIR_COL: [0, 0, 1, 1] * 50})
    assert uns.cluster_agreement(truth[REPAIR_COL].to_numpy() * 3, truth) == pytest.approx(1.0)
    independent = np.tile([0, 1], 100)  # alternates, unrelated to the 0,0,1,1 pattern
    assert uns.cluster_agreement(independent, truth) == pytest.approx(0.0, abs=1e-9)


def test_spec_profile_shares_sum_to_one_per_cluster(mini_frames: dict[str, pd.DataFrame]) -> None:
    features, _ = table(mini_frames)
    clusters = np.arange(40) % 2
    profile = uns.spec_profile(clusters, features, top=10)  # more than the 3 categories
    assert set(profile["spec"]) == set(SPEC_COLUMNS)
    sums = profile.groupby(["spec", "cluster"])["share"].sum()
    assert np.allclose(sums, 1.0)
    with pytest.raises(ValueError, match="same length"):
        uns.spec_profile(np.array([0]), features)


# --- figures --------------------------------------------------------------------------------
def test_figures_are_written(tmp_path: Path) -> None:
    z = blobs()
    repaired = (np.arange(len(z)) % 5 == 0).astype(int)
    scan = uns.kmeans_scan(z, [2, 3], 50, seed=0)
    truth = pd.DataFrame({REPAIR_COL: repaired})
    summary = uns.cluster_summary(np.arange(len(z)) % 3, truth)
    paths = [
        uns.plot_pca_map(z, repaired, np.array([0.5, 0.2]), tmp_path / "map.png"),
        uns.plot_k_scan(scan, tmp_path / "scan.png"),
        uns.plot_cluster_repair_rate(summary, repaired.mean(), tmp_path / "bars.png"),
    ]
    assert all(p.is_file() and p.stat().st_size > 1000 for p in paths)
