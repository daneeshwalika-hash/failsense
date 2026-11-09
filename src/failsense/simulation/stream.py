"""Replay validation readouts to ``POST /predict`` like a fleet that reports in real time.

Every tick one truck of the sample reports its next readout (with the readout immediately before
it as ``previous``, as the API contract requires), so the API's DuckDB prediction log and the
dashboard fill up. Optional drift: from a chosen reading on the counters are scaled up gradually,
imitating a fleet that is used harder than the training data; the monitoring added later
(drift check on the prediction log) has something to detect.

Run: ``make simulate`` or ``uv run python scripts/simulate_stream.py --help``.
Only the validation split is used; the test split is never read.
"""

from __future__ import annotations

import argparse
import copy
import logging
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from failsense.config import load_config
from failsense.dashboard import data
from failsense.dashboard.client import ApiClient, ApiError, api_url_from_config
from failsense.dashboard.payloads import build_payload, sorted_readouts
from failsense.data.labels import VEHICLE_COL
from failsense.data.schema import counter_columns

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Drift:
    """Counters are multiplied by a factor that ramps from 1 to ``factor``.

    The factor is 1 before reading ``start``, then grows linearly over ``ramp`` readings and
    stays at ``factor``.
    """

    start: int
    factor: float = 1.5
    ramp: int = 30

    def __post_init__(self) -> None:
        if self.start < 0 or self.ramp < 1 or self.factor <= 0:
            raise ValueError("drift needs start >= 0, ramp >= 1 and factor > 0")


@dataclass
class StreamStats:
    """What the run did."""

    sent: int = 0
    alerts: int = 0
    errors: int = 0


def drift_factor(index: int, drift: Drift | None) -> float:
    """Scaling factor of the ``index``-th reading (0-based); 1.0 without drift."""
    if drift is None or index < drift.start:
        return 1.0
    progress = min(1.0, (index - drift.start + 1) / drift.ramp)
    return 1.0 + (drift.factor - 1.0) * progress


def apply_drift(
    payload: Mapping[str, Any], factor: float, columns: Sequence[str] | None = None
) -> dict[str, Any]:
    """A copy of the request body with the counters of ``current`` and ``previous`` scaled.

    Both readouts are scaled, so the delta between them scales too (a harder-used truck). ``null``
    stays ``null``. The input is not modified.
    """
    scaled = copy.deepcopy(dict(payload))
    names = list(columns) if columns is not None else counter_columns()
    for part in ("current", "previous"):
        readout = scaled.get(part)
        if readout is None:
            continue
        for name in names:
            if readout.get(name) is not None:
                readout[name] = readout[name] * factor
    return scaled


def iter_readings(
    readouts: pd.DataFrame, specs: pd.DataFrame, seed: int, tail: int | None = None
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Request bodies of a fleet in random order: ``(vehicle_id, body)``.

    Each tick a random truck that still has readouts reports its next one (its readouts stay in
    time order, so ``previous`` is always the readout immediately before ``current``). A truck's
    very first readout is sent without ``previous``. Ends when every truck is exhausted.

    Args:
        tail: Replay only the last ``tail`` readouts of every truck (the period before the end of
            observation, where the risk is interesting); ``None`` replays everything from the
            first readout, which is mostly the quiet early life of the trucks.
    """
    if tail is not None and tail < 1:
        raise ValueError("tail must be at least 1")
    rng = np.random.default_rng(seed)
    trucks = {
        int(vid): sorted_readouts(group) for vid, group in readouts.groupby(VEHICLE_COL, sort=True)
    }
    spec_rows = {int(r[VEHICLE_COL]): r.to_dict() for _, r in specs.iterrows()}
    missing = sorted(set(trucks) - set(spec_rows))
    if missing:
        raise ValueError(f"trucks without specification row, e.g. {missing[:5]}")
    next_position = {vid: 0 if tail is None else max(0, len(t) - tail) for vid, t in trucks.items()}
    active = sorted(trucks)
    while active:
        vid = active[int(rng.integers(len(active)))]
        position = next_position[vid]
        body = build_payload(trucks[vid], spec_rows[vid], position, True, vehicle_id=vid)
        yield vid, body
        next_position[vid] += 1
        if next_position[vid] >= len(trucks[vid]):
            active.remove(vid)


def run_stream(
    client: ApiClient,
    readings: Iterator[tuple[int, dict[str, Any]]],
    interval: float,
    max_readings: int,
    drift: Drift | None = None,
    sleep: Callable[[float], None] = time.sleep,
    max_consecutive_errors: int = 5,
) -> StreamStats:
    """Send up to ``max_readings`` readings, one per ``interval`` seconds.

    A failed request is logged and counted; after ``max_consecutive_errors`` in a row the run
    stops with the last error (the API is probably down).
    """
    stats, consecutive = StreamStats(), 0
    for index, (vid, body) in enumerate(readings):
        if index >= max_readings:
            break
        factor = drift_factor(index, drift)
        if factor != 1.0:
            body = apply_drift(body, factor)
        try:
            answer = client.predict(body)
        except ApiError as exc:
            stats.errors += 1
            consecutive += 1
            logger.warning("reading %d (truck %d) failed: %s", index, vid, exc)
            if consecutive >= max_consecutive_errors:
                raise
        else:
            consecutive = 0
            stats.sent += 1
            stats.alerts += int(answer["alert"])
            logger.info(
                "reading %d truck %d time_step %.0f drift x%.2f -> risk %.2f %%%s",
                index, vid, body["current"]["time_step"], factor, 100 * answer["risk"],
                "  ALERT" if answer["alert"] else "",
            )  # fmt: skip
        sleep(interval)
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay validation readouts to POST /predict.")
    parser.add_argument("--api-url", help="API address (default: FAILSENSE_API_URL or config)")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds per reading")
    parser.add_argument("--max-readings", type=int, default=100, help="stop after this many")
    parser.add_argument("--seed", type=int, help="order of the trucks (default: config seed)")
    parser.add_argument(
        "--tail", type=int, default=10,
        help="replay only the last N readouts of each truck (0 = from the first readout)",
    )  # fmt: skip
    parser.add_argument("--drift-start", type=int, help="reading index where the drift starts")
    parser.add_argument("--drift-factor", type=float, default=1.5, help="final counter factor")
    parser.add_argument("--drift-ramp", type=int, default=30, help="readings to reach the factor")
    return parser


def main(argv: Sequence[str] | None = None) -> StreamStats:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args(argv)
    config = load_config()
    url = args.api_url or api_url_from_config(config)
    client = ApiClient(url, timeout=float(config["dashboard"]["timeout_seconds"]))
    client.health()  # fail fast, with a clear message, if the API is not running
    drift = (
        Drift(args.drift_start, args.drift_factor, args.drift_ramp)
        if args.drift_start is not None
        else None
    )
    readouts, specs, _ = data.load_demo_trucks(config)
    seed = int(config["seed"]) if args.seed is None else args.seed
    stats = run_stream(
        client,
        iter_readings(readouts, specs, seed, args.tail or None),
        args.interval,
        args.max_readings,
        drift,
    )
    logger.info("done: %d sent, %d alerts, %d errors", stats.sent, stats.alerts, stats.errors)
    return stats


if __name__ == "__main__":
    main()
