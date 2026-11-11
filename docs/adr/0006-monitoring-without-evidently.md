# ADR 0006: Drift check with PSI + KS on the risk distribution; Evidently not added

- **Status:** Accepted (reversible: Evidently stays an option)
- **Date:** 2026-11-09
- **Deciders:** Claude Code under the autonomous working mode of 09.11; Daneesh to confirm

## Context

The plan for 14.11 lists "Evidently `drift_check.py`". Adding Evidently was tried (`uv add --group monitoring
evidently`) and measured: **35 additional packages** (a web framework `litestar`, `nltk`, `statsmodels`,
`plotly`, `typer`, `uvloop` ...) and an `iterative-telemetry` client; its API also changed between 0.6 and
0.7 (`evidently.report` / `metric_preset` vs. `evidently.Report` / `presets`). The telemetry opt-out could not
be verified. A heavy new dependency is one of the things the working mode of 09.11 reserves for Daneesh, so
the dependency was **reverted** and not committed.

## Decision

`failsense.monitoring.drift` implements the two statistics that matter for one score distribution with
`scipy` and `numpy` (already installed): the **PSI** (decile bins of the reference, thresholds 0.1 / 0.25)
and the **two-sample KS test**. Status = `alert` if PSI >= 0.25 or (KS p < 0.01 and PSI >= 0.1), `warning` if
PSI >= 0.1 or KS p < 0.01, `insufficient_data` below 100 rows. The reference is the deployed model's risks on
the official validation last readouts (`make drift-reference`); the current data are the newest rows of the
API's prediction log, read through `GET /predictions/latest`, ignoring rows of other model versions.

## Consequences

- Output drift only: the log stores no input columns (a decision of block 6). Input drift, per-feature
  reports and HTML dashboards are what Evidently would add.
- Measured power with 200 rows: a x1 and a x3 scaling of the counters are **not** flagged (PSI 0.03 / 0.07);
  x10 and above are (PSI 0.11-0.15, KS p < 0.004). The model is fairly insensitive to the raw counter scale
  (most of its signal comes from the histogram features, which a scaling of the counters leaves unchanged), so
  "no alert" does not mean "no change in the fleet". *(Wording corrected on 2026-11-11: this line named `log1p`,
  but the deployed XGBoost pipeline has no `log1p` step; only the linear models use it.)*
- To adopt Evidently later: `uv add --group monitoring evidently` (not in `default-groups`), a function that
  renders its `DataDriftPreset` from the same reference CSV and the log rows, and a statement about
  telemetry. Daneesh decides.
