# ADR 0007: Promote a retrained model only if it is clearly cheaper on a fixed holdout

- **Status:** Accepted
- **Date:** 2026-11-09
- **Deciders:** Claude Code under the autonomous working mode of 09.11; Accepted — Daneesh Walika, 2026-11-11

## Context

The plan asks for `make retrain` with a champion / challenger comparison: "promote only if the total cost is
better on a fixed holdout". Three problems have to be solved: (1) the **registered** model was fitted on all
training trucks, so it cannot be judged on a holdout of them; (2) the holdout has only about 100 positive
evaluation rows, so a small cost difference is noise (the lesson of ADR 0005: a one-standard-deviation rule is
too lenient); (3) a new model version needs its **own** calibrator and threshold, otherwise the API would
serve wrong risks without any error.

## Decision

- **Fixed holdout:** the trucks of one fold (`retrain.holdout_fold`, 20 %) of the project's CV. Both the
  champion **recipe** (parameters and rounds of the registered model) and the challenger recipe (the best
  Optuna parameters of `make tune`) are refitted on the other trucks and judged on the validation-like
  evaluation rows of the holdout trucks (the project's standard evaluation set).
- **Score:** the total cost (SCANIA matrix, primary mapping) of the cost-optimal alert decision, with the
  threshold of each model cross-fitted by truck inside the holdout (`cross_fit_threshold`).
- **Rule:** promote only if the challenger is cheaper **and** the 95 % interval of the paired bootstrap of
  the cost difference (500 resamples of the evaluation rows) is entirely below 0. Default: keep the champion.
- **Promotion** (`--promote`; `make retrain` is a dry run) registers the challenger, fitted on all training
  trucks, and moves the alias. `deployment_choice.model_version` in the calibration summary records the
  version the calibrator belongs to; `ModelService.from_config` **refuses to start** on another version and
  names the fix (`make calibrate`). A summary without a version only logs a warning (old files).

## Consequences

- With ~100 positives the rule will rarely promote; that is intended. A real retrain would add newer data and
  could use a larger, more recent holdout.
- A promotion is a two-step process by design: register (retrain), then recalibrate (calibrate); the API
  cannot serve the half-finished state.
- The validation and test splits are not used by `retrain`.
