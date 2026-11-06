"""Why does the calibrated evaluation set have 607 / 26 / 2 / 0 positives in classes 1-4?

The official validation has 16 / 14 / 30 / 76 (dataset paper, documentation numbers used here
only as a *reference*, never to choose anything). This diagnostic uses **training data only**
(training labels, training readouts, plus the ``time_step`` of the validation readouts that the
evaluation protocol already uses). Questions:

1. Are the readouts of repaired trucks denser close to the repair? (class densities, gaps)
2. Could the validation classes come from another reference point than
   ``d = length_of_study_time_step - time_step``? (shift test)
3. Where does calibrated rank matching put the repaired trucks, and why almost never in
   classes 3-4? (reach vs. target quantiles, distance of the chosen readouts)

Run it with ``uv run python -m failsense.analysis.class_mix`` (``make class-mix``). The protocol
is not changed and no calibration is re-run.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from failsense.config import load_config, resolve_path
from failsense.data.labels import (
    CLASS_COL,
    DISTANCE_COL,
    TIME_COL,
    VEHICLE_COL,
    build_labels,
    class_bounds_from_config,
)
from failsense.models.baseline_cv import read_table, table_path
from failsense.models.eval_set import (
    EVAL_METHODS,
    build_eval_cutoffs_by_method,
    validation_time_steps,
    with_censored_tail,
)

logger = logging.getLogger(__name__)

# Class counts of the official validation split as printed in the dataset paper. Reference only.
OFFICIAL_VALIDATION_CLASS_COUNTS = {1: 16, 2: 14, 3: 30, 4: 76}
REPORT_FILE = "class_mix_check_q1.csv"
SUMMARY_FILE = "class_mix_check_q1.md"


def class_widths(bounds: Sequence[float]) -> dict[int, float]:
    """Width of the windows of classes 1..n, e.g. ``(48, 24, 12, 6)`` -> 24, 12, 6, 6."""
    edges = [*bounds, 0.0]
    return {k + 1: float(edges[k] - edges[k + 1]) for k in range(len(bounds))}


def repaired_known(labels: pd.DataFrame, repaired: pd.Series) -> pd.DataFrame:
    """Known-label readouts of the repaired trucks, sorted by truck and time.

    Raises:
        ValueError: If a required column is missing or no repaired truck is left.
    """
    missing = [c for c in (VEHICLE_COL, TIME_COL, DISTANCE_COL, CLASS_COL) if c not in labels]
    if missing:
        raise ValueError(f"labels is missing columns: {missing}")
    flags = labels[VEHICLE_COL].map(repaired)
    rows = labels[(flags == 1) & labels[CLASS_COL].notna()]
    if rows.empty:
        raise ValueError("no known readouts of repaired trucks")
    return rows.sort_values([VEHICLE_COL, TIME_COL], kind="stable")


def class_density_table(rows: pd.DataFrame, bounds: Sequence[float]) -> pd.DataFrame:
    """Readouts of the repaired trucks per class window: count, share and density per time unit.

    ``readouts_per_unit`` = readouts / window width / number of repaired trucks. A flat value
    means the readouts are evenly spread in time; a larger value near the repair means denser.
    """
    n_trucks = rows[VEHICLE_COL].nunique()
    widths = class_widths(bounds)
    counts = rows[CLASS_COL].astype(int).value_counts()
    trucks = rows.groupby(CLASS_COL)[VEHICLE_COL].nunique()
    table = []
    for k, width in widths.items():
        n = int(counts.get(k, 0))
        table.append(
            {
                "class": k,
                "window": width,
                "n_readouts": n,
                "readouts_per_unit": n / width / n_trucks,
                "share_of_trucks_with_readout": trucks.get(k, 0) / n_trucks,
            }
        )
    return pd.DataFrame(table)


def gap_by_distance(rows: pd.DataFrame, edges: Sequence[float]) -> pd.DataFrame:
    """Time between a readout and the previous one of the same truck, by distance to the repair."""
    work = rows.sort_values([VEHICLE_COL, TIME_COL])
    work = work.assign(gap=work.groupby(VEHICLE_COL)[TIME_COL].diff())
    bins = [-np.inf, *edges, np.inf]
    work["distance_bin"] = pd.cut(work[DISTANCE_COL], bins)
    out = work.groupby("distance_bin", observed=True)["gap"].agg(["size", "median", "mean"])
    return out.rename(columns={"size": "n_readouts", "median": "median_gap", "mean": "mean_gap"})


def expected_positive_counts(
    rows: pd.DataFrame, bounds: Sequence[float], shift: float = 0.0
) -> np.ndarray:
    """Expected positives per class if every repaired truck gets ONE uniformly random readout.

    Per truck the share of its readouts in each class is summed over the repaired trucks
    (censored trucks contribute class 0). ``shift`` moves the reference point of the distance:
    ``d' = d + shift`` (positive = the failure is assumed ``shift`` steps earlier); ``d' < 0``
    counts as the last class. The result is the expected number of positives per class among
    the training trucks (one readout each).
    """
    distance = rows[DISTANCE_COL].to_numpy(dtype=float) + shift
    upper = list(bounds)  # 48, 24, 12, 6: class k if d' <= upper[k-1]
    cls = np.zeros(len(rows), dtype=int)
    for k, bound in enumerate(upper, start=1):
        cls[distance <= bound] = k
    shares = pd.get_dummies(pd.Series(cls, index=rows.index)).reindex(
        columns=range(len(upper) + 1), fill_value=0
    )
    per_truck = shares.astype(float).groupby(rows[VEHICLE_COL].to_numpy()).mean()
    return per_truck.sum().to_numpy()[1:]  # classes 1..n


def chi_square(observed: Sequence[float], expected: Sequence[float]) -> float:
    """Pearson chi-square of the observed class counts vs. the expected *shape* (same total).

    Returns ``inf`` if an expected share is 0 where something was observed.
    """
    obs = np.asarray(observed, dtype=float)
    exp = np.asarray(expected, dtype=float)
    if obs.shape != exp.shape or obs.sum() <= 0 or exp.sum() <= 0:
        raise ValueError("observed and expected need the same length and a positive total")
    exp = exp / exp.sum() * obs.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(exp > 0, (obs - exp) ** 2 / exp, np.where(obs > 0, np.inf, 0.0))
    return float(terms.sum())


def positive_class_counts(rows: pd.DataFrame, n_classes: int = 4) -> dict[int, int]:
    """Number of evaluation rows per class 1..n."""
    counts = rows[CLASS_COL].astype(int).value_counts()
    return {k: int(counts.get(k, 0)) for k in range(1, n_classes + 1)}


def offset_table(
    reach: pd.Series,
    targets: Sequence[float],
    quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
) -> pd.DataFrame:
    """Training reach (last known readout per truck) vs. validation last-readout targets.

    Rank matching pairs the truck at rank ``q`` with the target at quantile ``q``; the column
    ``offset`` = reach - target is therefore the distance (in time steps) at which a truck is
    evaluated before the end of its history.
    """
    reach_q = reach.quantile(list(quantiles)).to_numpy()
    target_q = np.quantile(np.asarray(targets, dtype=float), list(quantiles))
    return pd.DataFrame(
        {
            "quantile": list(quantiles),
            "reach": reach_q,
            "target": target_q,
            "offset": reach_q - target_q,
        }  # fmt: skip
    )


def distance_bins_of_chosen(
    rows: pd.DataFrame, repaired: pd.Series, edges: Sequence[float]
) -> pd.Series:
    """Number of chosen readouts of repaired trucks per distance-to-repair bin."""
    chosen = rows[rows[VEHICLE_COL].map(repaired) == 1]
    return pd.cut(chosen[DISTANCE_COL], [-np.inf, *edges, np.inf]).value_counts().sort_index()


def run(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the three checks on the training data and write the report files."""
    reports = resolve_path(config["paths"]["reports"])
    bounds = class_bounds_from_config(config)
    seed = int(config["seed"])
    readouts = read_table(table_path(config, "train", "readouts"), columns=[VEHICLE_COL, TIME_COL])
    tte = read_table(table_path(config, "train", "tte"))
    labels = build_labels(readouts, tte, bounds)
    repaired = tte.set_index(VEHICLE_COL)["in_study_repair"]
    n_trucks = labels[VEHICLE_COL].nunique()
    rows = repaired_known(labels, repaired)

    density = class_density_table(rows, bounds)
    gaps = gap_by_distance(rows, (6, 12, 24, 48, 96, 200))
    reference = np.array([OFFICIAL_VALIDATION_CLASS_COUNTS[k] for k in range(1, 5)])

    shifts = []
    for shift in (-12.0, -6.0, 0.0, 6.0, 12.0):
        expected = expected_positive_counts(rows, bounds, shift)
        shifts.append({"shift": shift, "positive_rate": expected.sum() / n_trucks,
                       "chi2_vs_reference": chi_square(reference, expected),
                       **{f"expected_c{k + 1}": v for k, v in enumerate(expected)}})  # fmt: skip

    targets = validation_time_steps(config)
    mixes = []
    chosen_bins = None
    for method in EVAL_METHODS:
        cut, info = build_eval_cutoffs_by_method(method, labels, targets, repaired, seed)
        counts = positive_class_counts(cut)
        total = sum(counts.values())
        chi2 = chi_square(reference, list(counts.values())) if total else float("inf")
        mixes.append({"set": method, "n_positive": total, "positive_rate": total / len(cut),
                      "chi2_vs_reference": chi2,
                      **{f"c{k}": v for k, v in counts.items()}})  # fmt: skip
        if method == "calibrated":
            chosen_bins = distance_bins_of_chosen(cut, repaired, (6, 12, 24, 48, 60, 96, 200))
            positives = cut[cut[CLASS_COL].astype(int) >= 1]
    official = {f"c{k}": int(v) for k, v in enumerate(reference, 1)}
    mixes.append({"set": "official validation (reference)", "n_positive": int(reference.sum()),
                  "positive_rate": reference.sum() / 5046, "chi2_vs_reference": 0.0,
                  **official})  # fmt: skip

    reach = with_censored_tail(labels).groupby(VEHICLE_COL)[TIME_COL].max()
    offsets = offset_table(reach, targets)
    result = {
        "density": density, "gaps": gaps, "shifts": pd.DataFrame(shifts),
        "mixes": pd.DataFrame(mixes), "offsets": offsets, "chosen_bins": chosen_bins,
        "calibrated_positive_distance": positives[DISTANCE_COL].quantile([0, 0.25, 0.5, 0.75, 1]),
    }  # fmt: skip
    result["mixes"].to_csv(reports / REPORT_FILE, index=False, float_format="%.4f")
    for name, frame in result.items():
        logger.info("%s:\n%s", name, frame.to_string() if hasattr(frame, "to_string") else frame)
    return result


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run(load_config())


if __name__ == "__main__":
    main()
