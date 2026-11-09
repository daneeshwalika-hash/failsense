"""Drift check: PSI by hand, status rules, reference loading, log rows, exit codes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from failsense.monitoring import drift as dr

RNG = np.random.default_rng(0)


def risks(n: int, mean: float = 0.03, seed: int = 0) -> np.ndarray:
    """Right-skewed risks like the model's (lognormal around ``mean``)."""
    return np.clip(np.random.default_rng(seed).lognormal(np.log(mean), 0.6, n), 0, 1)


# --- PSI ----------------------------------------------------------------------------------------
def test_psi_of_a_sample_with_itself_is_zero() -> None:
    sample = risks(2000)
    assert dr.psi(sample, sample) == pytest.approx(0.0, abs=1e-9)


def test_psi_by_hand_on_two_bins() -> None:
    ref = np.array([0.0] * 50 + [1.0] * 50)
    cur = np.array([0.0] * 20 + [1.0] * 80)
    # bins split at the median 0.5 -> shares ref (0.5, 0.5), cur (0.2, 0.8)
    expected = (0.2 - 0.5) * np.log(0.2 / 0.5) + (0.8 - 0.5) * np.log(0.8 / 0.5)
    assert dr.psi(ref, cur, n_bins=2) == pytest.approx(expected, abs=1e-3)


def test_psi_grows_with_the_shift_and_is_roughly_symmetric_in_size() -> None:
    ref = risks(5000)
    small = dr.psi(ref, risks(5000, 0.033, 1))
    big = dr.psi(ref, risks(5000, 0.06, 2))
    assert small < dr.PSI_WARN < dr.PSI_ALERT < big
    assert dr.psi(ref, risks(5000, 0.015, 3)) > dr.PSI_ALERT  # a drop is a shift as well


def test_psi_handles_values_outside_the_reference_range_and_empty_bins() -> None:
    ref = risks(1000)
    assert np.isfinite(dr.psi(ref, np.full(500, 0.9)))  # everything in the top bin
    assert dr.psi(ref, np.full(500, 0.9)) > dr.PSI_ALERT


def test_psi_constant_reference() -> None:
    assert dr.psi(np.full(100, 0.1), np.full(100, 0.1)) == 0.0
    assert dr.psi(np.full(100, 0.1), np.linspace(0, 1, 100)) == float("inf")


@pytest.mark.parametrize(
    "ref, cur, bins, message",
    [([], [1.0], 10, "empty"), ([1.0], [], 10, "empty"), ([1.0, np.nan], [1.0], 10, "finite"),
     ([1.0, 2.0], [1.0], 1, "n_bins")],
)  # fmt: skip
def test_psi_rejects_bad_input(ref: list, cur: list, bins: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        dr.psi(ref, cur, bins)


# --- status -----------------------------------------------------------------------------------
def test_same_distribution_is_ok() -> None:
    result = dr.evaluate_drift(risks(5046), risks(500, seed=5), threshold=0.035)
    assert result.status == "ok" and result.exit_code == 0 and result.reasons == []
    assert result.psi < dr.PSI_WARN and result.ks_p_value > dr.KS_P_VALUE


def test_a_large_shift_is_an_alert_with_reasons() -> None:
    result = dr.evaluate_drift(risks(5046), risks(500, 0.08, 6), threshold=0.035)
    assert result.status == "alert" and result.exit_code == 2
    assert any("PSI" in r for r in result.reasons) and any("KS" in r for r in result.reasons)
    assert result.alert_rate_current > result.alert_rate_reference
    assert result.mean_risk_current > result.mean_risk_reference


def test_a_moderate_psi_shift_by_hand() -> None:
    """60 % uniform + 40 % in the lower half: decile shares 0.06 / 0.14 -> PSI about 0.17."""
    ref = np.linspace(0.0, 1.0, 1000)
    cur = np.r_[np.linspace(0.0, 1.0, 600), np.linspace(0.0, 0.5, 400)]
    expected = 5 * (0.06 - 0.1) * np.log(0.6) + 5 * (0.14 - 0.1) * np.log(1.4)
    value = dr.psi(ref, cur)
    assert value == pytest.approx(expected, abs=0.01)
    assert dr.PSI_WARN <= value < dr.PSI_ALERT
    # with 1,000 rows the KS test is significant too, so the verdict is an alert (documented rule)
    assert dr.evaluate_drift(ref, cur, 0.5, min_rows=50).status == "alert"


def test_too_few_rows_give_no_verdict() -> None:
    result = dr.evaluate_drift(risks(1000), risks(40), 0.035)
    assert result.status == "insufficient_data" and result.exit_code == 3
    assert result.psi is None and "need at least 100" in result.reasons[0]
    empty = dr.evaluate_drift(risks(1000), np.array([]), 0.035)
    assert empty.status == "insufficient_data" and empty.mean_risk_current is None


def test_empty_reference_is_an_error() -> None:
    with pytest.raises(ValueError, match="reference is empty"):
        dr.evaluate_drift([], [0.1] * 200, 0.03)


def test_a_tiny_but_significant_shift_alone_is_only_a_warning() -> None:
    ref = risks(200_000, 0.03, 1)
    cur = risks(200_000, 0.0306, 2)  # 2 % higher: KS sees it, PSI says "stable"
    result = dr.evaluate_drift(ref, cur, 0.035)
    assert result.psi < dr.PSI_WARN
    assert result.ks_p_value < dr.KS_P_VALUE and result.status == "warning"


# --- log rows and the reference files ---------------------------------------------------------
def test_current_risks_keep_only_the_reference_model_version() -> None:
    rows = [{"risk": 0.1, "model_version": "1"}, {"risk": 0.2, "model_version": 2},
            {"risk": 0.3, "model_version": "1"}]  # fmt: skip
    values, other = dr.current_risks(rows, "1")
    assert values.tolist() == [0.1, 0.3] and other == 1


def test_load_reference_and_a_helpful_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make drift-reference"):
        dr.load_reference(tmp_path)
    pd.DataFrame({"risk": [0.1, 0.2]}).to_csv(tmp_path / dr.REFERENCE_CSV, index=False)
    (tmp_path / dr.REFERENCE_JSON).write_text(json.dumps({"model_version": "1", "threshold": 0.03}))
    reference, meta = dr.load_reference(tmp_path)
    assert reference.tolist() == [0.1, 0.2] and meta["model_version"] == "1"


def test_fetch_latest_builds_the_url_and_refuses_other_schemes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = {}

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def read(self) -> bytes:
            return b'[{"risk": 0.1}]'

    def fake_urlopen(url: str, timeout: float) -> Response:
        seen["url"], seen["timeout"] = url, timeout
        return Response()

    monkeypatch.setattr(dr.urllib.request, "urlopen", fake_urlopen)
    assert dr.fetch_latest("http://127.0.0.1:8000/", 250) == [{"risk": 0.1}]
    assert seen["url"] == "http://127.0.0.1:8000/predictions/latest?limit=250"
    with pytest.raises(ValueError, match="http"):
        dr.fetch_latest("file:///etc/passwd", 10)


def test_check_end_to_end_writes_the_report_and_ignores_other_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = risks(3000)
    pd.DataFrame({"risk": reference}).to_csv(tmp_path / dr.REFERENCE_CSV, index=False)
    (tmp_path / dr.REFERENCE_JSON).write_text(
        json.dumps({"model_version": "1", "threshold": 0.035})
    )
    rows = [{"risk": float(r), "model_version": "1"} for r in risks(300, 0.09, 4)]
    rows += [{"risk": 0.001, "model_version": "2"}] * 10
    monkeypatch.setattr(dr, "fetch_latest", lambda url, limit: rows)

    result = dr.check({"paths": {"reports": str(tmp_path)}}, "http://x", 1000, 100)

    assert result.status == "alert" and result.n_current == 300  # the 10 version-2 rows are ignored
    saved = json.loads((tmp_path / dr.REPORT_JSON).read_text())
    assert saved["status"] == "alert" and saved["ignored_other_model_rows"] == 10
