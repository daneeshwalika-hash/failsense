# ADR 0001: Switch from AI4I 2020 to real-world data (SCANIA Component X, Bosch CNC)

- **Status:** Accepted
- **Date:** 2026-10-26
- **Deciders:** Daneesh Walika

## Context

FailSense started with three datasets: AI4I 2020 (main classification data),
NASA C-MAPSS FD001 (RUL) and SECOM (real manufacturing data, Should item).
On 26.10.2026 three problems became clear:

1. **AI4I 2020 is synthetic.** It is designed to look like industrial data,
   but the failure modes follow simple documented rules. A model can only
   rediscover these rules.
2. **AI4I 2020 is one of the most common predictive-maintenance datasets**
   in public portfolios. It does not differentiate a job application.
3. **The course uses the same data.** The teacher's course repository uses
   AI4I in several notebooks and SECOM in the SVM chapter. A project on the
   same data looks like course homework.

The EDA on AI4I (24.–25.10.) also showed the strongest lesson of that dataset:
the failure-mode columns `TWF`, `HDF`, `PWF`, `OSF`, `RNF` define the target.
A simple rule `any(TWF..RNF)` reaches recall 0.973 and precision 0.948
without any ML. This leakage lesson is kept here and in
`docs/problem_framing.md`.

## Options considered

| Option | Real data | Fits the five questions | Job relevance (Stuttgart) | Main drawback |
| --- | --- | --- | --- | --- |
| Keep AI4I 2020 | No | Yes | Low | Synthetic, common, used in the course |
| **SCANIA Component X** | Yes (truck fleet) | Yes: binary, 0–4 classes, time-to-event, anomaly, cost matrix | High (trucks, automotive suppliers) | Anonymised variables, labels must be built from time-to-event data |
| **Bosch CNC Machining** | Yes (Bosch production plant) | Partly: good/bad, anomaly, no RUL | High (Bosch brand) | Small (~1,700 samples), signal processing needed |
| MetroPT-3 | Yes (metro compressor) | Partly: very few failure events | Medium | Weak base for supervised learning |
| CARE to Compare | Yes (wind turbines) | Mainly anomaly detection | Medium | Very large, many features |
| Backblaze Drive Stats | Yes (data center) | Yes | Medium (IT, not manufacturing) | Several GB per quarter |

## Decision

- **SCANIA Component X** replaces AI4I 2020 as the main dataset
  (classification, cost-based threshold, anomaly detection).
- **NASA C-MAPSS FD001** stays for RUL regression. It is not used in the course.
- **Bosch CNC Machining** replaces SECOM as the Should item
  (anomaly detection and transfer between machines).
- AI4I 2020 and SECOM are removed from the repository and the download script.
  The AI4I notebook stays available in the Git history (commits `a9e4b68`, `dbe5b0a`).

## Consequences

**Positive**

- Real fleet data and real factory data instead of synthetic data.
- The cost-based threshold uses the cost matrix published with the SCANIA
  challenge instead of an assumed 10 : 1 ratio.
- Course methods (for example FFT features) are applied to new data, not copied.

**Negative / risks**

- Variables are anonymised and partly scaled: no physical feature engineering
  (power, strain) as with AI4I.
- Training labels must be derived from time-to-event data, including censored
  vehicles. This is new code (`labels.py`) with its own tests.
- Results will be lower than on AI4I. This is expected and will be reported honestly.
- About one evening of rework (EDA, problem framing). The deadline 15.11.2026
  does not move; if the schedule slips, Bosch CNC is cut first.

**Leakage rules that change**

- SCANIA: label columns (`length_of_study_time_step`, `in_study_repair`,
  derived class) and `vehicle_id` are never features; no vehicle in more than
  one split; CV with `StratifiedGroupKFold` by `vehicle_id`; features only from
  past readouts.
- Bosch CNC: split by machine or time period, never random.

## References

- Kharazian, Z., Lindgren, T., Magnússon, S., Steinert, O., Andersson Reyna, O. (2025).
  *SCANIA Component X dataset: a real-world multivariate time series dataset for
  predictive maintenance.* Scientific Data 12, 493. https://doi.org/10.1038/s41597-025-04802-6
- Dataset: Lindgren, T. et al. (2024). SCANIA Component X Dataset (Version 2).
  Scania CV AB. https://doi.org/10.5878/jvb5-d390
- Tnani, M.-A., Feil, M., Diepold, K. (2022). *Smart Data Collection System for
  Brownfield CNC Milling Machines: A New Benchmark Dataset for Data-Driven
  Machine Monitoring.* Procedia CIRP 107. https://github.com/boschresearch/CNC_Machining (CC BY 4.0)
- Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset. UCI. https://doi.org/10.24432/C5HS5C
