# Plan: one frozen evaluation on the test splits (Q1, Q2, RUL)

**Status (17.11.2026): the single test run was done on 16.11.2026**, one day earlier than planned below (see
`docs/results.md` section 15 and `reports/test_evaluation_final.json`). The text below is the plan as written on
11.11.2026, kept as it was; the "State" column of section 0 is updated. Before the run, no model, threshold,
calibrator or feature choice used the test split, and the test labels had never been loaded for an analysis. What
*has* read the test files (format conversion and schema checks only) is listed in section 1a. The run needed
Daneesh's separate approval, which was given.

## 0. Order of events (schedule decided by Daneesh on 10.11.2026; it replaces the plan of 09.11)

| When | Step | State |
|---|---|---|
| 09.11 | Plan, six decisions, freeze code, evaluation script + fixture tests (`failsense.evaluate_test`, `scripts/evaluate_test_once.py`) | done |
| 10.-11.11 | Review fixes: preflight of the frozen models before the lock, rehearsal on validation (`make rehearse`), honest list of the test-file readers (section 1a), README path from a clean clone | done |
| until Sat 14.11 | Remaining review fixes. No test run. | done |
| Sun 15.11 | The repository is published **without** a test result. README: "Final test evaluation pending — scheduled after course review (17.11.2026)." | not done: the repository stayed private; it is published after the test run |
| Mon 16.11 | Meeting with the teacher (course review). **No test run**, only small changes. | changed: the single test run was done on this day |
| **Tue 17.11** (run brought forward to Mon 16.11) | **Test day**, in this order (section 0a): clean-clone check, re-freeze, freeze-verify, rehearse, evaluate-test, commit the results, docs with the numbers as they come out | done |
| 18.-19.11 | Buffer, slides, backup demo video. No code changes after the test run (bugs become GitHub issues), except blockers | open |
| Fri 20.11 | Presentation | open |

*The plan of 09.11 had the re-freeze and the single run on Sat 14.11 and the release steps on Wed 18.11. It was
changed on 10.11.2026 so that the code review fixes and the course review come before the one irreversible step.*

## 0a. Test day (Tue 17.11.2026): the run order

| # | Step | Command | Must be true before the next step |
|---|---|---|---|
| 0 | Clean-clone check following the README ("From zero to a running API"); fixes go into the final commit | - | the README works as written; everything is committed |
| 1 | **freeze** on the FINAL commit (training data only; hashes the five test files, parses none). The manifest also hashes the evaluation script, `compose.yaml`, `Dockerfile`, `.env.example`, the prompt and the bundled model, no exemptions. Commit the new manifest; report the differences to the 09.11 manifest | `uv run python -m failsense.freeze --replace` | `git status` is empty after the manifest commit |
| 2 | **freeze-verify** | `make freeze-verify` | "verified", exit code 0 |
| 3 | **rehearse**: the same code path on the VALIDATION split; no lock, no test file opened or hashed, result in the log and in the temp folder | `make rehearse` | Q1 prints 5,046 trucks, 136 positives, PR-AUC 0.0653, 3,013 alerts, cost 27,565 (the in-sample decision of `reports/calibration_q1_summary.json`); `git status` is still empty |
| 4 | **evaluate-test**: the single run, only with Daneesh's approval | `uv run python scripts/evaluate_test_once.py --approved-single-run` | `reports/test_evaluation.lock` says "finished" |
| 5 | Commit `reports/test_evaluation_final.json` and the lock file; then the docs (`docs/results.md`, model card, README) with the numbers as they come out | - | - |

**Rule: between the freeze (step 1) and the test run (step 4) nothing is written to the repository.** No
`make drift-check` (it writes `reports/drift_check.json`), no `make figures`, no training or calibration target, no
edited file. `git status` must be empty: an untracked or changed file makes `verify_manifest` refuse the run. The
rehearsal respects this by design (it refuses an output path inside the repository). There is deliberately no `make`
target that starts the single run: it needs the explicit flag `--approved-single-run`.

If the rehearsal does not print the expected Q1 numbers, or the preflight of step 4 refuses, **stop**: no lock has
been created, the run is still available, and the cause can be fixed (a fix in a frozen path means a new freeze).

The manifest of 09.11 (`reports/frozen_manifest.json`) no longer verifies on purpose: code and config changed after it
(compose fix, LLM grounding, signature, seed step, the evaluation script, the review fixes of 10.-11.11). The re-freeze
replaces it; the evaluation script refuses to run until `make freeze-verify` is clean.

## 1. Principle

The test data are used **once**. Everything that could change a number is decided and stored **before** the run,
the evaluation script reads **only saved artifacts** (no training, no fitting, no threshold search, no
calibration), refuses to run if any artifact differs from its recorded hash, and refuses a second run. The
result is reported whatever it is; nothing is tuned afterwards.

## 1a. What has read the test files so far (and what has not)

"Used once" is about results: nothing has ever been fitted, tuned, chosen or scored with test data. It does not mean
that no program has opened the files. The complete list:

| Reader | What it reads | Why | Uses the content for a model? |
|---|---|---|---|
| `make parquet` (`failsense.data.convert`) | all nine SCANIA CSV files, including `test_labels.csv` | format conversion CSV -> Parquet; the Parquet copy of the test labels is never read back (the evaluation reads the raw CSV) | no |
| `make validate` (`failsense.data.schema`) | test readouts and specifications (Parquet) | data contract (columns, dtypes, ranges) and "no truck in two splits" | no |
| EDA notebook `notebooks/01_eda_scania.ipynb` | test readouts and specifications, **not** the labels | truck IDs for the split overlap, number and share of the specification categories | no (see below) |
| `make freeze` / `make freeze-verify` (`failsense.freeze`) | the five test files as bytes | SHA-256 only, nothing is parsed | no |
| the single run (`failsense.evaluate_test`) | the five test files | the evaluation of this plan | this is the one use |

`make parquet`, `make validate` and the EDA notebook read test files only for format conversion and schema checks; no
model, threshold, calibrator or feature choice uses them. The one thing the EDA noticed there, that validation and
test contain specification categories that do not occur in training, is handled by
`OneHotEncoder(handle_unknown="ignore")`, a robustness setting the validation split alone already requires. The
C-MAPSS test files (`test_FD001.txt`, `RUL_FD001.txt`) have not been read by anything but the hashing. The rehearsal
(`make rehearse`) reads the validation split and opens no test file.

## 2. What is frozen

| | Q1 (binary, SCANIA) | Q2 (classes 0-4, SCANIA) | RUL (C-MAPSS FD001) |
|---|---|---|---|
| Model | registered `failsense-q1` **version 1** (run `a40dd5b3...`), alias `candidate` | multi-class XGBoost, the Q1 recipe (`multi:softprob`), fitted on the `k5` training rows | LightGBM + rolling mean/std (windows 5, 15), parameters of `rul.lightgbm`, fitted on all 100 training engines; **baselines frozen the same way (training engines only): Ridge on the raw settings + sensors (alpha 1.0) and the mean predictor** |
| Status | **exists** | **does not exist as an artifact** (`q2_classes.run` fits it in memory) | **does not exist as an artifact** (only cross-validation was run) |
| Calibration / correction | sigmoid slope 0.5258518583, intercept -1.9573859341, clip eps 1e-6 (`calibration_q1_summary.json`, `deployment_choice`) | prior shift with `pi_train` (counted on the training rows) and `pi_target` = the class mix **published for the validation** (4,910 / 16 / 14 / 30 / 76). The test class counts are **not** used (ADR 0004) | none; the RUL cap 125 of the training target |
| Decision rule | alert if calibrated risk >= **0.0353238554** | class with the lowest expected cost under the 5 x 5 matrix | point prediction at the last cycle of each test engine |
| Cost / error | SCANIA matrix: primary mapping (false alarm 7, miss 200-500, alert on a failing truck 0) **and** the literal mapping | 5 x 5 matrix, total cost | RMSE in cycles |
| Features | the 239 registered input columns (`registry_q1.json`), built by the same shared function as training and validation (last readout per truck, deltas from the previous readout) | same 239 columns | 3 settings + 21 sensors + rolling features of the informative sensors (the 15 non-constant sensors of the training engines (frozen list)), past cycles of the same engine only |
| Seeds | fixed (42) | fixed (42) | fixed (42) |

## 3. The freeze step (must be approved and run first, it does not touch test data)

`failsense.freeze` (`make freeze`) trains **on training data only** and writes `artifacts/frozen/`
(git-ignored, hashes committed):

0. **Q1 model:** a copy of the registered model (version 1) is stored in `artifacts/frozen/`, so the evaluation
   does not depend on the MLflow registry.
1. **Q2 model:** the multi-class pipeline of `q2_classes.run`, fitted on all `k5` training rows, saved (skops /
   MLflow, same trusted types as `register`); plus `prior_train`, `prior_target`, the cost matrix.
2. **RUL models:** the LightGBM pipeline, the Ridge baseline and the mean predictor, all fitted on all training engines, saved; plus the list of informative
   sensors, the windows and the cap.
3. **Manifest** `reports/frozen_manifest.json` (committed; the frozen files themselves stay in the git-ignored
   `artifacts/frozen/`) with: the git commit (the working tree must be clean), the
   SHA-256 of every frozen file (Q1 registry artifact reference + version, calibration summary, registry
   file, Q2 and RUL models, config sections used, the cost matrix), the SHA-256 of the **five** test files (SCANIA readouts, specifications, labels; C-MAPSS `test_FD001.txt`,
   `RUL_FD001.txt`; hashed only: the bytes are not parsed), the library versions (`uv.lock` hash), and the decisions of section 5.

## 4. The evaluation script (`failsense.evaluate_test`, entry `scripts/evaluate_test_once.py`; written 09.11, preflight and rehearsal added 10.11)

* Refuses to start unless (`failsense.freeze.verify_manifest`): the flag `--approved-single-run` is given, the working tree is clean and no code or
  config file (`src/`, `configs/`, `pyproject.toml`, `uv.lock`, the registry and calibration files) differs from the
  manifest commit (documentation may change), every hash matches, and `reports/test_evaluation.lock` does not exist.
* **Preflight before the lock (10.11):** all frozen models (Q1, Q2, RUL LightGBM and Ridge), the calibrator and the
  threshold are loaded and checked first: the file loads with the recorded trusted types, the model was fitted on
  exactly the recorded input columns, a probe prediction on synthetic rows is finite. A failure names the artifact
  and stops the run **without a lock and without reading a test file**, so a broken file cannot use up the run.
* Only then it creates the lock, **before** reading any test file, so a crash after that point cannot be "retried
  with a fix" silently.
* **Rehearsal (`--rehearse-on-validation`, `make rehearse`, 10.11):** the same evaluation functions on the official
  validation split (C-MAPSS has no validation file: training engines cut before their end, an in-sample plumbing
  check). No approval, no lock, no test file opened or hashed, no MLflow run; the result goes to the log and to a file
  in the system temp folder, and an output path inside the repository is refused.
* **Q1:** test readouts -> last readout per truck -> registered model -> frozen calibrator -> frozen threshold.
  Reported: PR-AUC and ROC-AUC (95 % bootstrap interval over trucks) **with the test prevalence next to them**
  (and the lift PR-AUC / prevalence), total cost at the frozen threshold under both mappings **next to the constant
  baselines: "never alert" and "alert every truck" (the primary baseline) under both mappings**, the saving against
  each, alert rate, recall and precision, recall per class. The test cost is genuinely out-of-sample (nothing is
  cross-fitted).
* **Q2:** test last readouts -> Q2 model -> corrected and uncorrected probabilities -> expected-cost decision.
  Reported: total cost, the five constant predictors and the best constant (frozen from the validation:
  "always class 4"), macro F1, confusion matrices, paired bootstrap of the correction and of the saving vs. the
  best constant.
* **RUL:** the last cycle of each of the 100 test engines (rolling features from its past cycles) -> prediction ->
  RMSE against `RUL_FD001.txt`. **Primary: against the RUL capped at 125** (the training target); also reported
  against the raw RUL (comparable with the literature) and the asymmetric NASA score. **The frozen baselines (mean
  predictor and Ridge on the raw sensors) are scored on the same 100 last cycles and reported next to LightGBM**,
  with a paired bootstrap of the difference to Ridge (RUL compared with a baseline). 100 engines: the interval is
  wide and is reported.
* Writes `reports/test_evaluation_final.json`, one MLflow run tagged `stage = test, single run`, and the sections
  for `docs/results.md`. **No training, tuning, calibration or analysis code can reach the test split.** Two guards in
  `tests/test_evaluate_test.py` enforce it: a static one (the only modules with a test path or a loop over all splits are
  the four of section 1a: `freeze`, `evaluate_test`, `data/convert`, `data/schema`) and a dynamic one (a spy on
  `pd.read_csv`, `pd.read_parquet` and `open` runs the training entry points on fixtures and fails if a file named
  `test_*` or `RUL_FD*` is opened; a further test proves that this guard does fail on a fake reader). Another test
  asserts that the SCANIA test files are read as RAW csv, never from the Parquet copies.
* Script tests (67 in `tests/test_evaluate_test.py`, mini fixtures only, never the real test files; the metrics are checked by hand): refuses without the flag, with a
  manifest that does not verify, with an existing lock; the lock is created before the first read; eight kinds of
  broken artifacts stop the run before the lock and before any read; the rehearsal reproduces hand-computed metrics
  on a validation fixture, opens no test-split file and writes nothing into the repository; the metrics match
  hand-computed values on a tiny example.

## 5. Decisions needed before the freeze

1. **RUL model:** freeze LightGBM + rolling features (best cross-validation, 18.1 RMSE) rather than Ridge. *Proposal: yes.*
2. **RUL reference:** capped at 125 as primary, raw as secondary. *Proposal: as written above.*
3. **Q2 target mix:** the validation mix published in the paper, not the test mix. *Proposal: yes (ADR 0004).*
4. **Q1 headline mapping:** primary mapping as the headline, literal mapping next to it. *Proposal: both, primary first.*
5. **Not evaluated on test:** the MLP, LightGBM for Q1, the anomaly detectors, the anomaly feature, the
   retrain challenger (none was selected). *Proposal: leave them out.*
6. **After the run:** report the numbers as they are, including a worse-than-validation result, in
   `docs/results.md` and the model card; no re-tuning, no second run.

## 6. What is deliberately not frozen

The test labels' class mix and the test `time_step` distribution are unknown to us and stay so; the threshold and
the Q2 correction are therefore applied as learned on the validation, and a shift between validation and test shows
up in the result instead of being corrected away.
