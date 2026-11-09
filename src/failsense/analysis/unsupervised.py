"""PCA map and KMeans clusters of the training trucks (unsupervised, labels only for colour).

One row per truck: its LAST readout with the same engineered features as the supervised model
(counter deltas from the previous readout, histogram shares, specifications). The labels
(``in_study_repair``, ``proximity_class``) are kept in a separate table and never enter the
feature matrix; they are only used afterwards to colour the map and to cross-tabulate clusters
against repairs.

Questions (course topic 09.11, dimensionality reduction and clustering):

1. How many principal components carry the variance of the 239 inputs, and does the repaired /
   not-repaired split show up in the first two?
2. Does KMeans find a number of groups that means something (silhouette with ``sample_size``)
   and do the clusters differ in their repair rate?

SQL analogy: PCA is like replacing 239 columns by a few computed columns that keep as much of
the row-to-row differences as possible; KMeans then groups rows that are close in that space.

Leakage / honesty notes
-----------------------
* Everything is fitted on **training trucks only**; validation and test are not read.
* Fitting an unsupervised model on all training trucks and colouring by their own labels is not
  a prediction experiment, so no train/test split is needed; but the numbers must not be read
  as predictive performance.
* The last readout of a repaired truck lies just before its repair, the last readout of a
  non-repaired truck at the end of its observation. That is a different situation from the
  random cut-offs of the validation, so "repaired trucks separate" here is a descriptive finding.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.pipeline import Pipeline

from failsense.config import load_config, resolve_path
from failsense.data.labels import (
    CLASS_COL,
    REPAIR_COL,
    TIME_COL,
    VEHICLE_COL,
    build_labels,
    class_bounds_from_config,
)
from failsense.data.schema import (
    SPEC_COLUMNS,
    counter_columns,
    feature_columns,
    histogram_groups,
)
from failsense.features.build import (
    attach_specifications,
    build_model_features,
    features_params_from_config,
)
from failsense.models.baseline_cv import read_table, table_path
from failsense.models.eval_set import wilson_interval
from failsense.models.pipeline import assert_no_forbidden_features, build_preprocessor
from failsense.models.validation_data import registry_input_columns
from failsense.plotting import (
    COLOR_TRAIN,
    COLOR_VAL,
    GRID,
    INK_PRIMARY,
    INK_SECONDARY,
    SURFACE,
)

logger = logging.getLogger(__name__)

REPORT_PREFIX = "unsupervised"


# --- data: one row per truck ----------------------------------------------------------------
def truck_table(
    readouts: pd.DataFrame,
    specs: pd.DataFrame,
    tte: pd.DataFrame,
    class_bounds: Sequence[float],
    counter_cols: Sequence[str],
    hist_groups: Mapping[str, Sequence[str]],
    top_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features and labels of the last readout of every truck.

    Returns:
        ``(features, truth)``, both indexed by ``vehicle_id`` (sorted). ``features`` holds the
        engineered model inputs and the specifications; ``truth`` has ``in_study_repair`` (0/1)
        and ``proximity_class`` (Int8, ``<NA>`` for a censored truck near its end).

    Raises:
        ValueError: On empty readouts or a truck without tte / specification row.
    """
    if readouts.empty:
        raise ValueError("readouts is empty")
    # Deltas need the full, sorted history; the last rows are selected afterwards.
    ordered = readouts.sort_values([VEHICLE_COL, TIME_COL], kind="stable").reset_index(drop=True)
    last_idx = ordered.groupby(VEHICLE_COL)[TIME_COL].idxmax()
    labels = build_labels(ordered[[VEHICLE_COL, TIME_COL]], tte, class_bounds)

    features = build_model_features(
        ordered, pd.Index(last_idx.to_numpy()), counter_cols, hist_groups, top_fraction
    )
    features = attach_specifications(features, specs)
    features = features.set_index(VEHICLE_COL).sort_index()

    truth = labels.loc[last_idx.to_numpy(), [VEHICLE_COL, CLASS_COL]].set_index(VEHICLE_COL)
    repaired = tte.set_index(VEHICLE_COL)[REPAIR_COL].astype(int)
    truth[REPAIR_COL] = repaired.loc[truth.index].to_numpy()
    return features, truth.sort_index()[[REPAIR_COL, CLASS_COL]]


def select_inputs(features: pd.DataFrame, input_columns: Sequence[str] | None) -> pd.DataFrame:
    """The feature matrix: the given columns (e.g. the registry's) or all columns of ``features``.

    Label and identifier columns are refused (``assert_no_forbidden_features``). The columns
    are checked to exist; the order is kept.
    """
    columns = list(input_columns) if input_columns is not None else list(features.columns)
    assert_no_forbidden_features(columns)
    missing = [c for c in columns if c not in features.columns]
    if missing:
        raise ValueError(f"{len(missing)} input columns missing, e.g. {missing[:5]}")
    return features[columns]


# --- PCA ------------------------------------------------------------------------------------
def build_embedding_pipeline(
    matrix: pd.DataFrame, n_components: int | float, seed: int, log1p: bool = True
) -> Pipeline:
    """Impute -> (log1p of raw counts) -> standardise -> one-hot specs -> PCA, as one Pipeline.

    ``log1p`` is applied to the raw counters and histogram bins only (non-negative by the data
    contract; the deltas can be negative). The counters are cumulative and strongly right
    skewed; without log1p a few extreme trucks would own the first components.
    """
    categorical = [c for c in SPEC_COLUMNS if c in matrix.columns]
    log_columns = [c for c in feature_columns() if c in matrix.columns] if log1p else []
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(categorical, "standard", log_columns)),
            ("pca", PCA(n_components=n_components, random_state=seed)),
        ]
    )


def variance_table(pca: PCA) -> pd.DataFrame:
    """Explained variance per component and cumulative (``component`` counts from 1)."""
    ratio = np.asarray(pca.explained_variance_ratio_)
    return pd.DataFrame(
        {
            "component": np.arange(1, len(ratio) + 1),
            "explained": ratio,
            "cumulative": ratio.cumsum(),
        }
    )


def top_loadings(pipeline: Pipeline, component: int = 0, n: int = 8) -> pd.DataFrame:
    """The ``n`` input features with the largest absolute weight in one principal component."""
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    weights = pipeline.named_steps["pca"].components_[component]
    order = np.argsort(-np.abs(weights))[:n]
    return pd.DataFrame({"feature": names[order], "weight": weights[order]})


# --- KMeans ---------------------------------------------------------------------------------
def kmeans_scan(
    z: np.ndarray, ks: Sequence[int], sample_size: int, seed: int, n_init: int = 10
) -> pd.DataFrame:
    """Inertia and silhouette for every k.

    The silhouette needs all pairwise distances (O(n^2)); ``sample_size`` evaluates it on a
    random subsample of rows (fixed ``random_state``), which is close to the full value and
    fast. ``sample_size`` larger than the number of rows is capped.

    Raises:
        ValueError: If a k is < 2 or >= the number of rows, or ``z`` is not a 2-D array.
    """
    z = np.asarray(z, dtype=float)
    if z.ndim != 2 or len(z) == 0:
        raise ValueError("z must be a non-empty 2-D array")
    if any(k < 2 or k >= len(z) for k in ks):
        raise ValueError(f"every k must be in [2, {len(z) - 1}], got {list(ks)}")
    size = min(int(sample_size), len(z))
    rows = []
    for k in ks:
        model = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(z)
        # sample_size == n means "all rows"; sklearn then needs sample_size=None
        score = silhouette_score(
            z, model.labels_, sample_size=None if size == len(z) else size, random_state=seed
        )
        rows.append({"k": k, "inertia": float(model.inertia_), "silhouette": float(score)})
    return pd.DataFrame(rows)


def best_k(scan: pd.DataFrame) -> int:
    """The k with the highest silhouette (ties: the smaller k)."""
    return int(scan.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])


def cluster_summary(clusters: np.ndarray, truth: pd.DataFrame) -> pd.DataFrame:
    """Per cluster: size, repaired trucks, repair rate with a 95 % Wilson interval, lift.

    ``lift`` = cluster repair rate / overall repair rate. ``truth`` must be in the same row
    order as ``clusters``.
    """
    if len(clusters) != len(truth):
        raise ValueError("clusters and truth must have the same length")
    repaired = truth[REPAIR_COL].to_numpy(dtype=int)
    base = repaired.mean()
    rows = []
    for cluster in sorted(set(clusters.tolist())):
        mask = clusters == cluster
        n, n_rep = int(mask.sum()), int(repaired[mask].sum())
        low, high = wilson_interval(n_rep, n)
        rows.append(
            {"cluster": int(cluster), "n_trucks": n, "n_repaired": n_rep,
             "repair_rate": n_rep / n, "ci_low": low, "ci_high": high,
             "lift": (n_rep / n) / base if base > 0 else float("nan")}
        )  # fmt: skip
    return pd.DataFrame(rows)


def spec_profile(clusters: np.ndarray, features: pd.DataFrame, top: int = 3) -> pd.DataFrame:
    """Share of each specification category inside every cluster (what do the clusters mean?).

    Long table ``spec, category, cluster, share`` with the ``top`` most frequent categories of
    each specification column (over all trucks). ``features`` must be in the row order of
    ``clusters``.
    """
    if len(clusters) != len(features):
        raise ValueError("clusters and features must have the same length")
    rows = []
    for column in (c for c in SPEC_COLUMNS if c in features.columns):
        keep = features[column].value_counts().index[:top]
        shares = pd.crosstab(features[column], np.asarray(clusters), normalize="columns")
        for category in keep:
            for cluster in shares.columns:
                rows.append({"spec": column, "category": category, "cluster": int(cluster),
                             "share": float(shares.loc[category, cluster])})  # fmt: skip
    return pd.DataFrame(rows, columns=["spec", "category", "cluster", "share"])


def cluster_agreement(clusters: np.ndarray, truth: pd.DataFrame) -> float:
    """Normalised mutual information between clusters and the repaired flag (0 = unrelated)."""
    return float(normalized_mutual_info_score(truth[REPAIR_COL].to_numpy(dtype=int), clusters))


# --- figures --------------------------------------------------------------------------------
def _style(ax: Any) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def plot_pca_map(z: np.ndarray, repaired: np.ndarray, variance: np.ndarray, path: Path) -> Path:
    """PC1 vs PC2, trucks coloured by repaired / not repaired (the minority is drawn on top)."""
    fig = Figure(figsize=(7.5, 5.5), dpi=150, facecolor=SURFACE)
    ax = fig.subplots()
    for flag, label, color in ((0, "not repaired", COLOR_TRAIN), (1, "repaired", COLOR_VAL)):
        mask = repaired == flag
        ax.scatter(z[mask, 0], z[mask, 1], s=6, alpha=0.45, color=color, linewidths=0,
                   label=f"{label} ({int(mask.sum()):,})")  # fmt: skip
    ax.set_xlabel(f"PC1 ({variance[0]:.1%} of variance)", color=INK_SECONDARY)
    ax.set_ylabel(f"PC2 ({variance[1]:.1%} of variance)", color=INK_SECONDARY)
    ax.set_title("Training trucks (last readout) in the first two principal components",
                 color=INK_PRIMARY, fontsize=11, loc="left")  # fmt: skip
    _style(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_PRIMARY, markerscale=3)
    fig.tight_layout()
    return _save(fig, path)


def plot_k_scan(scan: pd.DataFrame, path: Path) -> Path:
    """Silhouette per k (left) and inertia per k (right); two panels, one axis each."""
    fig = Figure(figsize=(8.5, 3.6), dpi=150, facecolor=SURFACE)
    left, right = fig.subplots(1, 2)
    left.plot(scan["k"], scan["silhouette"], marker="o", color=COLOR_TRAIN, linewidth=2)
    left.set_ylabel("silhouette (sampled)", color=INK_SECONDARY)
    left.set_title("Silhouette: higher is better", color=INK_PRIMARY, fontsize=10, loc="left")
    right.plot(scan["k"], scan["inertia"], marker="o", color=COLOR_VAL, linewidth=2)
    right.set_ylabel("inertia (within-cluster sum of squares)", color=INK_SECONDARY)
    right.set_title("Inertia: look for the elbow", color=INK_PRIMARY, fontsize=10, loc="left")
    for ax in (left, right):
        ax.set_xlabel("number of clusters k", color=INK_SECONDARY)
        ax.set_xticks(list(scan["k"]))
        _style(ax)
    fig.tight_layout()
    return _save(fig, path)


def plot_cluster_repair_rate(summary: pd.DataFrame, base_rate: float, path: Path) -> Path:
    """Repair rate per cluster with its 95 % interval against the overall rate."""
    fig = Figure(figsize=(7, 3.8), dpi=150, facecolor=SURFACE)
    ax = fig.subplots()
    x = np.arange(len(summary))
    err = [summary["repair_rate"] - summary["ci_low"], summary["ci_high"] - summary["repair_rate"]]
    ax.bar(x, summary["repair_rate"] * 100, width=0.6, color=COLOR_VAL,
           yerr=np.array(err) * 100, ecolor=INK_SECONDARY, capsize=3)  # fmt: skip
    ax.axhline(base_rate * 100, color=INK_SECONDARY, linewidth=1, linestyle="--",
               label=f"all trucks ({base_rate:.1%})")  # fmt: skip
    names = [
        f"cluster {c}\n(n={n:,})"
        for c, n in zip(summary["cluster"], summary["n_trucks"], strict=True)
    ]
    ax.set_xticks(x, names)
    ax.set_ylabel("repaired trucks (%)", color=INK_SECONDARY)
    ax.set_title("Repair rate per KMeans cluster", color=INK_PRIMARY, fontsize=11, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_PRIMARY)
    fig.tight_layout()
    return _save(fig, path)


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    logger.info("Saved figure to %s", path)
    return path


# --- orchestration --------------------------------------------------------------------------
def load_truck_data(config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training trucks only: ``(features, truth)`` from the config's training tables."""
    readouts = read_table(table_path(config, "train", "readouts"))
    specs = read_table(table_path(config, "train", "specifications"))
    tte = read_table(table_path(config, "train", "tte"))
    return truck_table(
        readouts, specs, tte, class_bounds_from_config(config), counter_columns(),
        histogram_groups(), features_params_from_config(config),
    )  # fmt: skip


def run(config: Mapping[str, Any]) -> dict[str, Any]:
    """PCA + KMeans on the training trucks; writes CSV and PNG files to ``reports/``."""
    section = config["unsupervised"]
    seed = int(config["seed"])
    reports = resolve_path(config["paths"]["reports"])
    features, truth = load_truck_data(config)
    matrix = select_inputs(features, registry_input_columns(config))
    logger.info("%d training trucks, %d input columns", len(matrix), matrix.shape[1])

    # Full PCA first (all components) for the variance table, then the 2-D map.
    full = build_embedding_pipeline(matrix, n_components=None, seed=seed).fit(matrix)
    variance = variance_table(full.named_steps["pca"])
    n_keep = int(np.searchsorted(variance["cumulative"].to_numpy(), section["variance_kept"]) + 1)
    pipe = build_embedding_pipeline(matrix, n_components=n_keep, seed=seed).fit(matrix)
    z = pipe.transform(matrix)
    loadings = {f"PC{i + 1}": top_loadings(pipe, i) for i in range(2)}

    scan = kmeans_scan(z, section["ks"], section["silhouette_sample_size"], seed)
    k = best_k(scan)
    clusters = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(z)
    summary = cluster_summary(clusters, truth)
    nmi = cluster_agreement(clusters, truth)
    specs = spec_profile(clusters, features)
    base = float(truth[REPAIR_COL].mean())

    variance.to_csv(reports / f"{REPORT_PREFIX}_pca_variance.csv", index=False, float_format="%.5f")
    scan.to_csv(reports / f"{REPORT_PREFIX}_k_scan.csv", index=False, float_format="%.5f")
    summary.to_csv(reports / f"{REPORT_PREFIX}_clusters.csv", index=False, float_format="%.5f")
    specs.to_csv(reports / f"{REPORT_PREFIX}_cluster_specs.csv", index=False, float_format="%.4f")
    repaired = truth[REPAIR_COL].to_numpy(dtype=int)
    explained = variance["explained"].to_numpy()
    plot_pca_map(z, repaired, explained, reports / f"{REPORT_PREFIX}_pca_map.png")
    plot_k_scan(scan, reports / f"{REPORT_PREFIX}_k_scan.png")
    plot_cluster_repair_rate(summary, base, reports / f"{REPORT_PREFIX}_clusters.png")

    _log_to_mlflow(config, matrix.shape, n_keep, variance, scan, k, nmi)
    for name, table in (("variance (first 10)", variance.head(10)), ("k scan", scan),
                        ("clusters", summary), *loadings.items()):  # fmt: skip
        logger.info("%s:\n%s", name, table.to_string(index=False))
    logger.info("%d components keep >= %.0f %% variance", n_keep, 100 * section["variance_kept"])
    logger.info("best k=%d, NMI(cluster, repaired)=%.4f", k, nmi)
    return {"variance": variance, "scan": scan, "summary": summary, "k": k, "nmi": nmi,
            "n_components": n_keep, "loadings": loadings, "specs": specs, "z": z, "truth": truth,
            "clusters": clusters}  # fmt: skip


def _log_to_mlflow(
    config: Mapping[str, Any], shape: tuple[int, int], n_keep: int, variance: pd.DataFrame,
    scan: pd.DataFrame, k: int, nmi: float,
) -> None:  # fmt: skip
    """One MLflow run with the parameters and the headline numbers."""
    import mlflow

    from failsense.tracking import setup_mlflow

    setup_mlflow(config, experiment=config["unsupervised"]["mlflow_experiment"])
    with mlflow.start_run(run_name="pca_kmeans_train_trucks"):
        mlflow.log_params({"n_trucks": shape[0], "n_inputs": shape[1], "n_components": n_keep,
                           "variance_kept": config["unsupervised"]["variance_kept"],
                           "best_k": k, "seed": config["seed"]})  # fmt: skip
        mlflow.log_metrics({"pc1_explained": float(variance["explained"].iloc[0]),
                            "pc2_explained": float(variance["explained"].iloc[1]),
                            "best_silhouette": float(scan.set_index("k").loc[k, "silhouette"]),
                            "nmi_cluster_vs_repaired": nmi})  # fmt: skip


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run(load_config())


if __name__ == "__main__":
    main()
