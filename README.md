# FailSense

End-to-end predictive maintenance system: predicts machine failures, estimates remaining useful life (RUL), detects anomalies and explains its predictions.

> Work in progress — first release planned for November 2026.

## Quickstart

```bash
make setup   # uv sync + pre-commit hooks
make data    # download AI4I 2020, SECOM, NASA C-MAPSS into data/raw/
make test    # run pytest
```

## Data

| Dataset | Source | Licence |
| --- | --- | --- |
| AI4I 2020 Predictive Maintenance | UCI Machine Learning Repository | CC BY 4.0 |
| SECOM | UCI Machine Learning Repository | see source page |
| Turbofan Engine Degradation Simulation (C-MAPSS) | NASA Prognostics Data Repository | see source page |

Raw data is never committed; `scripts/download_data.sh` fetches it.
