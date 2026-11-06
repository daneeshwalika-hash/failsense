# ADR 0003: Cut Bosch CNC (Should item) from v1.0

- **Status:** Accepted
- **Date:** 2026-11-06
- **Deciders:** Daneesh Walika
- **Amends:** [ADR 0001](0001-dataset-switch.md) (Bosch CNC as the Should item; "if the schedule
  slips, Bosch CNC is cut first")

## Context

ADR 0001 (26.10) made Bosch CNC Machining the Should item for anomaly detection on real vibration
data, with the rule "if the schedule slips, Bosch CNC is cut first". The project plan repeats it:
behind schedule by more than 2 days means a Should item is cut, Bosch CNC first.

The repository was about six days behind the plan on 05.11. The catch-up of plan days 30.10–05.11
is done (PRs #12–#21), and the calibration and cost threshold of 06.11 are merged (PR #22). The
buffer for 07.11–15.11 is thin, and every remaining Must item still has to be built and tested:
FastAPI, Docker, Streamlit, anomaly detection on SCANIA, RUL on C-MAPSS (time series), the MLP and
the local LLM report, plus the release on 15.11. The deadline 15.11.2026 does not move.

## Decision

Bosch CNC is **cut from v1.0**. It moves to "Future work" and is started only **after the
presentation on 20.11.2026**. The Must items keep their scope; anomaly detection stays Must on
SCANIA Component X.

## Consequences

- **09.11 and 10.11 cover SCANIA only.** PCA / KMeans on trucks (09.11) and IsolationForest + LOF
  on healthy trucks (10.11); no Bosch FFT features, no split by machine.
- **No `signals/` module and no `04_bosch_cnc` notebook in v1.0.** No Bosch data is downloaded, no
  data contract and no tests are written for it.
- **Anomaly detection is evaluated on one dataset.** The claim "works across machines" is not made
  in v1.0; the README lists the cross-machine test as future work.
- **Citation:** Tnani et al. (2022) stays out of the README "Data" section until the dataset is
  actually used. ADR 0001 keeps the original plan as history.
- **`CLAUDE.md` still lists Bosch** (sections 4, 5 and 8, and the leakage rule "split by machine
  or time period") and should be updated by Daneesh to match this decision.
- The freed time is a buffer for the Must items, not a reason to add new features
  (week 4 is reading and review only).

## How to resume (after 20.11)

1. Open a GitHub issue "Bosch CNC cross-machine anomaly detection" and a branch `feat/bosch-cnc`.
2. Data: `boschresearch/CNC_Machining` (UCI id 752), CC BY 4.0; vibration signals of three machines
   and good / bad labels. Add the dataset, licence and citation (Tnani et al., *Procedia CIRP* 107,
   2022) to the README only when the code uses them.
3. Leakage rule from ADR 0001: **split by machine or time period, never random.** Train the
   anomaly detector on good parts of some machines and evaluate on another machine.
4. Features: FFT / spectral features per signal window (a `signals/` module), PCA for the 2D map,
   IsolationForest / LOF as for SCANIA; a pandera data contract and small fixtures for tests.
5. Metric: PR-AUC of the anomaly score for the bad class and recall in the top-k, compared with a
   random ranking (same as Q4 on SCANIA).
