"""Champion / challenger: promote a new model recipe only if it is clearly cheaper (ADR 0007).

Run: ``uv run python -m failsense.models.retrain`` (``make retrain``, a dry run). With ``--promote``
a winning challenger is registered and takes the alias (``tuning.alias``).

Protocol (the same fixed split every time, so two retrains are comparable):

1. **Holdout** = the trucks of one fold of the project's CV (``retrain.holdout_fold``, 20 % of the
   training trucks, the same truck-to-fold assignment as ``make cv``). Neither recipe is fitted on
   them.
2. **Champion** = the recipe of the registered model (``reports/registry_q1.json``: XGBoost, its
   parameters and rounds). **Challenger** = the best Optuna parameters of ``make tune``. Both are
   refitted on the other four folds' trucks; refitting the champion is necessary because the
   REGISTERED model was fitted on all training trucks, including the holdout.
3. **Score** = total cost (SCANIA matrix, primary mapping: false alarm 7, miss 200-500) of the
   cost-optimal alert decision on the validation-like evaluation rows of the holdout trucks; the
   threshold of each model is chosen by cost on the other trucks of the holdout and applied to the
   held-out ones (``cross_fit_threshold``).
4. **Rule:** promote only if the challenger is cheaper and the 95 % interval of the paired bootstrap
   of the cost difference lies entirely below 0 (500 resamples of the evaluation rows). A cheaper
   challenger inside the noise is NOT promoted; the holdout has only ~100 positive rows, so the
   default outcome is "keep the champion".

Promotion registers the challenger (fitted on ALL training trucks, like ``register``) and moves the
alias. The calibrator and the alert threshold in ``calibration_q1_summary.json`` belong to the old
version: the API refuses to start on the new version until ``make calibrate`` has been run for it.

The validation and test splits are not used here. Real retraining on newer data would replace the
"outside the holdout" data of step 2 by the new training data and keep the rest.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from failsense.config import load_config, resolve_path
from failsense.models.boosters import make_booster
from failsense.models.cost import (
    BinaryCosts,
    alert_class_from_config,
    binary_costs,
    cost_matrix_from_config,
)
from failsense.models.q2_classes import paired_bootstrap_cost
from failsense.models.register import REGISTRY_FILE, register_winner
from failsense.models.threshold import cross_fit_threshold, per_truck_costs, total_cost
from failsense.models.tuning_data import build_tuning_data
from failsense.tracking import setup_mlflow

logger = logging.getLogger(__name__)

SUMMARY_FILE = "retrain_q1.json"
FEATURE_SET = "engineered"
EVAL_METHOD = "calibrated"


@dataclass(frozen=True)
class Recipe:
    """A model recipe: booster family, parameters and the number of rounds."""

    name: str
    family: str
    params: Mapping[str, Any]
    n_rounds: int
    engineered: bool = True


def champion_from_registry(reports_dir: Any) -> Recipe:
    """The recipe of the registered model (``registry_q1.json``)."""
    facts = json.loads((reports_dir / REGISTRY_FILE).read_text(encoding="utf-8"))
    return Recipe(
        f"champion_v{facts['version']}", facts["family"], facts["params"], int(facts["n_rounds"]),
        bool(facts["engineered"]),
    )  # fmt: skip


def challenger_from_file(path: Any, model: str) -> Recipe:
    """The challenger recipe: the Optuna result of ``make tune`` for ``model``.

    Raises:
        FileNotFoundError: If ``make tune`` has not been run.
        KeyError: If the file has no entry for ``model``.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `make tune` first")
    entry = json.loads(path.read_text(encoding="utf-8"))[model]
    return Recipe(f"challenger_{model}", model, entry["params"], int(entry["n_rounds"]),
                  bool(entry.get("engineered", True)))  # fmt: skip


def decision_costs(
    classes: Sequence[int], scores: Sequence[float], groups: Sequence[Any], costs: BinaryCosts,
    n_splits: int, seed: int,
) -> tuple[np.ndarray, float]:  # fmt: skip
    """Cost per truck of the cross-fitted cost-optimal alert decision, and their sum.

    Raises:
        ValueError: If the per-truck costs do not add up to the cross-fitted total (a bug guard).
    """
    result = cross_fit_threshold(classes, scores, groups, costs, n_splits, seed)
    alert_cost, miss_cost = per_truck_costs(classes, costs)
    per_truck = np.where(result.alerts, alert_cost, miss_cost)
    if not np.isclose(per_truck.sum(), result.cost):
        raise ValueError("per-truck costs and the cross-fitted total differ")
    return per_truck, float(result.cost)


def decide_promotion(
    cost_challenger: np.ndarray, cost_champion: np.ndarray, n_boot: int, seed: int
) -> dict[str, Any]:
    """ADR 0007 rule: the 95 % bootstrap interval of (challenger - champion) must be below 0."""
    paired = paired_bootstrap_cost(cost_challenger, cost_champion, n_boot, seed)
    promote = bool(paired["diff"] < 0 and paired["ci_high"] < 0)
    rule = "challenger cheaper AND the 95 % bootstrap interval of the difference is below 0"
    return {**paired, "promote": promote, "rule": rule}


def run(config: Mapping[str, Any], promote: bool = False) -> dict[str, Any]:
    """Compare champion and challenger on the fixed holdout; optionally promote the winner."""
    section = config["retrain"]
    seed, n_boot = int(config["seed"]), int(section["n_bootstrap"])
    n_splits = int(config["cv"]["n_splits"])
    reports = resolve_path(config["paths"]["reports"])
    champion = champion_from_registry(reports)
    challenger = challenger_from_file(
        resolve_path(section["challenger_file"]), section["challenger_model"]
    )
    costs = binary_costs(
        cost_matrix_from_config(config), alert_class_from_config(config), "primary"
    )

    data = build_tuning_data(config, eval_methods=(EVAL_METHOD,))
    fold = data.folds[int(section["holdout_fold"])]
    classes, groups = fold.c_eval[EVAL_METHOD], fold.vehicles_eval[EVAL_METHOD]
    logger.info("holdout fold %d: %d training rows, %d evaluation rows (%d positives)",
                fold.fold, len(fold.y_train), len(classes), int((classes >= 1).sum()))  # fmt: skip

    per_truck: dict[str, np.ndarray] = {}
    totals: dict[str, float] = {}
    for recipe in (champion, challenger):
        # ``engineered`` recipes use the engineered feature set, the others the base columns
        feature_set = FEATURE_SET if recipe.engineered else "base"
        booster = make_booster(recipe.family, recipe.params, seed, os.cpu_count() or 1,
                               recipe.n_rounds)  # fmt: skip
        booster.fit(fold.x_train[feature_set], fold.y_train)
        scores = booster.predict_proba(fold.x_eval[EVAL_METHOD][feature_set])[:, 1]
        per_truck[recipe.name], totals[recipe.name] = decision_costs(
            classes, scores, groups, costs, n_splits, seed
        )
        logger.info("%s: cross-fitted total cost %.0f", recipe.name, totals[recipe.name])

    never = total_cost(classes, np.zeros(len(classes), dtype=bool), costs)
    always = total_cost(classes, np.ones(len(classes), dtype=bool), costs)
    decision = decide_promotion(per_truck[challenger.name], per_truck[champion.name], n_boot, seed)
    summary: dict[str, Any] = {
        "holdout_fold": fold.fold, "n_eval_rows": int(len(classes)),
        "n_eval_positives": int((classes >= 1).sum()),
        "cost_never_alert": never, "cost_always_alert": always,
        "champion": {"name": champion.name, "total_cost": totals[champion.name],
                     "params": dict(champion.params), "n_rounds": champion.n_rounds},
        "challenger": {"name": challenger.name, "total_cost": totals[challenger.name],
                       "params": dict(challenger.params), "n_rounds": challenger.n_rounds},
        "decision": decision, "promoted": False, "promote_requested": promote,
        "note": "holdout of the training trucks; validation and test not used",
    }  # fmt: skip
    if decision["promote"] and promote:
        winner = {
            "winner": challenger.name,
            "family": challenger.family,
            "params": dict(challenger.params),
            "engineered": challenger.engineered,
            "n_rounds": challenger.n_rounds,
            "reason": f"retrain: cheaper than {champion.name} on the holdout "
            f"({decision['diff']:+.0f}, 95 % interval below 0)",
        }
        setup_mlflow(config)  # register_winner needs the project's tracking store to be selected
        facts = register_winner(config, winner)
        summary["promoted"] = True
        summary["new_version"] = facts["version"]
        logger.warning("Promoted %s as version %s. Run `make calibrate` before serving it: the API "
                       "refuses a version the calibrator was not fitted for.",
                       challenger.name, facts["version"])  # fmt: skip
    elif decision["promote"]:
        logger.info("The challenger WOULD be promoted (dry run: use --promote to register it)")
    else:
        logger.info("Keep the champion: %s", decision["rule"])

    reports.mkdir(parents=True, exist_ok=True)
    (reports / SUMMARY_FILE).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _log_to_mlflow(config, summary)
    return summary


def _log_to_mlflow(config: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
    import mlflow

    setup_mlflow(config, experiment=config["retrain"]["mlflow_experiment"])
    with mlflow.start_run(run_name="champion_vs_challenger"):
        mlflow.set_tags({"task": "q1_binary", "stage": "retrain"})
        mlflow.log_metrics({
            "cost_champion": summary["champion"]["total_cost"],
            "cost_challenger": summary["challenger"]["total_cost"],
            "cost_diff": summary["decision"]["diff"],
            "cost_diff_ci_high": summary["decision"]["ci_high"],
            "promoted": float(summary["promoted"]),
        })  # fmt: skip
        mlflow.log_param("promote_requested", summary["promote_requested"])


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--promote", action="store_true",
                        help="register the challenger and move the alias if it wins")  # fmt: skip
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run(load_config(), promote=args.promote)


if __name__ == "__main__":
    main()
