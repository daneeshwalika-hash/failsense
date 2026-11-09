"""Drift check of the served model's risks (PSI + Kolmogorov-Smirnov), no extra dependency.

What is compared: the distribution of the calibrated ``risk`` the API answered recently (read from
the API's prediction log through ``GET /predictions/latest``) against a **reference** made when the
model was deployed: the risks of the same model on the 5,046 official validation last readouts
(``make drift-reference`` -> ``reports/drift_reference_q1.csv`` + ``.json``).

* **PSI** (population stability index) = sum over bins of ``(cur - ref) * ln(cur / ref)``; the bins
  are the deciles of the reference. Rule of thumb from credit scoring: < 0.1 stable, 0.1-0.25
  moderate shift, >= 0.25 large shift. It measures the SIZE of a shift.
* **KS test** (two-sample Kolmogorov-Smirnov) asks whether the two samples can come from one
  distribution. With many rows it flags tiny shifts, so it is only an alert together with a PSI
  of 0.1.
* Also reported: the alert rate and the mean risk.

Limits (honest): this is **output drift** only; the log stores no input columns, so a change of the
inputs that does not move the risk is invisible, and a drift of the risk is not yet a loss of
accuracy (there are no labels in production). The reference is the validation last readouts: a
stream that replays all readouts of a truck is NOT comparable (use ``simulate_stream.py --tail 1``).

Evidently was considered and not added (ADR 0006): 35 extra packages incl. a telemetry client.

Exit codes of ``check``: 0 ok, 1 warning, 2 alert, 3 not enough data (for a cron job).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from failsense.config import load_config, resolve_path

logger = logging.getLogger(__name__)

PSI_WARN = 0.1
PSI_ALERT = 0.25
KS_P_VALUE = 0.01
MIN_ROWS = 100
REFERENCE_CSV = "drift_reference_q1.csv"
REFERENCE_JSON = "drift_reference_q1.json"
REPORT_JSON = "drift_check.json"
EXIT_CODES = {"ok": 0, "warning": 1, "alert": 2, "insufficient_data": 3}


def psi(
    reference: Sequence[float] | np.ndarray,
    current: Sequence[float] | np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-4,
) -> float:
    """Population stability index of ``current`` against ``reference`` (reference deciles as bins).

    Raises:
        ValueError: On empty or non-finite input, or ``n_bins < 2``.
    """
    ref, cur = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    if ref.size == 0 or cur.size == 0:
        raise ValueError("reference and current must not be empty")
    if not (np.isfinite(ref).all() and np.isfinite(cur).all()):
        raise ValueError("reference and current must be finite")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    if len(edges) < 3:  # a constant reference: one bin has no spread to compare
        return 0.0 if np.allclose(cur, ref[0]) else float("inf")
    ref_share = np.histogram(ref, bins=edges)[0] / ref.size
    cur_share = np.histogram(cur, bins=edges)[0] / cur.size
    ref_share, cur_share = np.clip(ref_share, eps, None), np.clip(cur_share, eps, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


@dataclass
class DriftResult:
    """Outcome of one drift check."""

    status: str
    n_reference: int
    n_current: int
    psi: float | None
    ks_statistic: float | None
    ks_p_value: float | None
    mean_risk_reference: float
    mean_risk_current: float | None
    alert_rate_reference: float
    alert_rate_current: float | None
    reasons: list[str] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.status]


def evaluate_drift(
    reference_risks: Sequence[float] | np.ndarray,
    current_risks: Sequence[float] | np.ndarray,
    threshold: float,
    min_rows: int = MIN_ROWS,
) -> DriftResult:
    """Compare the current risks with the reference; ``threshold`` defines the alert rate.

    Status: ``insufficient_data`` below ``min_rows``; ``alert`` if PSI >= 0.25 or (KS p < 0.01 and
    PSI >= 0.1); ``warning`` if PSI >= 0.1 or KS p < 0.01; else ``ok``.
    """
    ref, cur = np.asarray(reference_risks, dtype=float), np.asarray(current_risks, dtype=float)
    if ref.size == 0:
        raise ValueError("the reference is empty")
    base = {
        "n_reference": int(ref.size), "n_current": int(cur.size),
        "mean_risk_reference": float(ref.mean()),
        "alert_rate_reference": float((ref >= threshold).mean()),
    }  # fmt: skip
    if cur.size < min_rows:
        return DriftResult(
            "insufficient_data", **base, psi=None, ks_statistic=None, ks_p_value=None,
            mean_risk_current=float(cur.mean()) if cur.size else None,
            alert_rate_current=float((cur >= threshold).mean()) if cur.size else None,
            reasons=[f"only {cur.size} current rows, need at least {min_rows}"],
        )  # fmt: skip
    value = psi(ref, cur)
    ks = stats.ks_2samp(ref, cur)
    reasons = []
    if value >= PSI_ALERT:
        reasons.append(f"PSI {value:.3f} >= {PSI_ALERT}: large shift of the risk distribution")
    elif value >= PSI_WARN:
        reasons.append(f"PSI {value:.3f} >= {PSI_WARN}: moderate shift of the risk distribution")
    if ks.pvalue < KS_P_VALUE:
        reasons.append(f"KS test p = {ks.pvalue:.2g} < {KS_P_VALUE}: not the same distribution")
    if value >= PSI_ALERT or (ks.pvalue < KS_P_VALUE and value >= PSI_WARN):
        status = "alert"
    elif value >= PSI_WARN or ks.pvalue < KS_P_VALUE:
        status = "warning"
    else:
        status = "ok"
    return DriftResult(
        status, **base, psi=value, ks_statistic=float(ks.statistic), ks_p_value=float(ks.pvalue),
        mean_risk_current=float(cur.mean()), alert_rate_current=float((cur >= threshold).mean()),
        reasons=reasons,
    )  # fmt: skip


# --- reference and the API log --------------------------------------------------------------
def load_reference(reports: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Reference risks and their metadata (model version, threshold).

    Raises:
        FileNotFoundError: If ``make drift-reference`` has not been run.
    """
    csv, meta = reports / REFERENCE_CSV, reports / REFERENCE_JSON
    for path in (csv, meta):
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found; run `make drift-reference` first")
    return pd.read_csv(csv)["risk"].to_numpy(dtype=float), json.loads(meta.read_text("utf-8"))


def fetch_latest(api_url: str, limit: int, timeout: float = 30.0) -> list[dict[str, Any]]:
    """The newest rows of the API's prediction log (standard library HTTP)."""
    if not api_url.startswith(("http://", "https://")):
        raise ValueError("api_url must start with http:// or https://")
    url = f"{api_url.rstrip('/')}/predictions/latest?{urllib.parse.urlencode({'limit': limit})}"
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 (checked scheme)
        return json.loads(response.read().decode("utf-8"))


def current_risks(rows: Sequence[Mapping[str, Any]], model_version: str) -> tuple[np.ndarray, int]:
    """Risks of the rows made by ``model_version``, and how many rows came from other versions."""
    same = [r["risk"] for r in rows if str(r["model_version"]) == str(model_version)]
    return np.asarray(same, dtype=float), len(rows) - len(same)


def build_reference(config: Mapping[str, Any]) -> dict[str, Any]:
    """Risks of the DEPLOYED model on the validation last readouts (``make drift-reference``)."""
    from failsense.api.service import ModelService
    from failsense.models.calibration import apply_sigmoid_parameters
    from failsense.models.validation_data import build_validation_frame, select_model_inputs

    service = ModelService.from_config(config)
    frame = build_validation_frame(config)
    inputs = select_model_inputs(frame, service.input_columns)
    scores = service.pipeline.predict_proba(inputs)[:, 1]
    risks = apply_sigmoid_parameters(service.calibrator, scores)
    reports = resolve_path(config["paths"]["reports"])
    reports.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"risk": risks}).to_csv(reports / REFERENCE_CSV, index=False, float_format="%.6f")
    meta = {
        "model_name": service.info.name, "model_version": service.info.version,
        "threshold": service.threshold, "n": int(len(risks)),
        "mean_risk": float(risks.mean()), "alert_rate": float((risks >= service.threshold).mean()),
        "source": "official validation, last readout per truck (no test data)",
    }  # fmt: skip
    (reports / REFERENCE_JSON).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    logger.info("Reference: %s", meta)
    return meta


def check(config: Mapping[str, Any], api_url: str, limit: int, min_rows: int) -> DriftResult:
    """Fetch the recent log rows, compare with the reference, write ``reports/drift_check.json``."""
    reports = resolve_path(config["paths"]["reports"])
    reference, meta = load_reference(reports)
    rows = fetch_latest(api_url, limit)
    risks, other = current_risks(rows, meta["model_version"])
    if other:
        logger.warning("%d of %d log rows come from another model version and are ignored",
                       other, len(rows))  # fmt: skip
    result = evaluate_drift(reference, risks, float(meta["threshold"]), min_rows)
    (reports / REPORT_JSON).write_text(
        json.dumps({**asdict(result), "ignored_other_model_rows": other}, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("drift status: %s | %s", result.status, "; ".join(result.reasons) or "no findings")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("reference", help="write the reference from the validation (loads the model)")
    chk = sub.add_parser("check", help="compare the recent API log with the reference")
    chk.add_argument("--api-url", help="API address (default: FAILSENSE_API_URL or config)")
    chk.add_argument("--limit", type=int, default=1000, help="log rows to fetch (max 1000)")
    chk.add_argument("--min-rows", type=int, default=MIN_ROWS)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = load_config()
    if args.command == "reference":
        build_reference(config)
        return 0
    from failsense.dashboard.client import api_url_from_config

    return check(config, args.api_url or api_url_from_config(config), args.limit,
                 args.min_rows).exit_code  # fmt: skip


if __name__ == "__main__":
    sys.exit(main())
