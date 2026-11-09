"""Stream simulator: drift, replay order, error handling (fake client, no sleeping)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from failsense.dashboard.client import ApiError
from failsense.data.schema import counter_columns
from failsense.simulation import stream as sim


class FakeClient:
    def __init__(self, fail_from: int | None = None, always_fail: bool = False) -> None:
        self.bodies: list[dict[str, Any]] = []
        self.fail_from, self.always_fail = fail_from, always_fail

    def predict(self, body: dict[str, Any]) -> dict[str, Any]:
        n = len(self.bodies)
        self.bodies.append(body)
        if self.always_fail or (self.fail_from is not None and n >= self.fail_from):
            raise ApiError("down")
        return {"risk": 0.05, "alert": n % 2 == 0, "threshold": 0.03}


def readings(
    mini_frames: dict[str, pd.DataFrame], seed: int = 1, vehicles: int = 3, tail: int | None = None
):
    keep = mini_frames["readouts"]["vehicle_id"] < vehicles
    return sim.iter_readings(mini_frames["readouts"][keep], mini_frames["specs"], seed, tail)


# --- drift ----------------------------------------------------------------------------------
def test_drift_factor_ramps_linearly_and_stays() -> None:
    drift = sim.Drift(start=10, factor=2.0, ramp=4)
    assert [sim.drift_factor(i, drift) for i in (0, 9)] == [1.0, 1.0]
    assert sim.drift_factor(10, drift) == pytest.approx(1.25)
    assert sim.drift_factor(13, drift) == pytest.approx(2.0)
    assert sim.drift_factor(500, drift) == pytest.approx(2.0)
    assert sim.drift_factor(500, None) == 1.0


@pytest.mark.parametrize(
    "kwargs", [{"start": -1}, {"start": 0, "ramp": 0}, {"start": 0, "factor": 0}]
)
def test_drift_rejects_bad_settings(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="drift needs"):
        sim.Drift(**kwargs)


def test_apply_drift_scales_counters_of_both_readouts_and_keeps_the_rest(
    mini_frames: dict[str, pd.DataFrame],
) -> None:
    _, body = next(readings(mini_frames))
    _, later = list(readings(mini_frames))[5]  # a body that has a previous readout
    later["current"]["171_0"] = None
    before = {"current": dict(later["current"]), "previous": dict(later["previous"])}
    scaled = sim.apply_drift(later, 2.0)
    assert scaled["previous"]["666_0"] == pytest.approx(2 * before["previous"]["666_0"])
    assert scaled["current"]["666_0"] == pytest.approx(2 * before["current"]["666_0"])
    assert scaled["current"]["171_0"] is None  # null stays null
    assert scaled["current"]["167_0"] == before["current"]["167_0"]  # histograms untouched
    assert scaled["current"]["time_step"] == before["current"]["time_step"]
    assert later["current"] == before["current"]  # the input is not modified
    assert "previous" not in sim.apply_drift(body, 2.0)  # the first readout has none


# --- replay ---------------------------------------------------------------------------------
def test_each_truck_reports_its_readouts_in_time_order(
    mini_frames: dict[str, pd.DataFrame],
) -> None:
    seen: dict[int, list[float]] = {}
    for vid, body in readings(mini_frames):
        seen.setdefault(vid, []).append(body["current"]["time_step"])
        if "previous" in body:  # previous = the readout immediately before current
            assert body["previous"]["time_step"] == seen[vid][-2]
        else:
            assert len(seen[vid]) == 1  # only a truck's first readout has no previous
    assert set(seen) == {0, 1, 2}
    assert all(steps == sorted(steps) and len(steps) == 12 for steps in seen.values())


def test_replay_is_reproducible_and_seed_changes_the_order(
    mini_frames: dict[str, pd.DataFrame],
) -> None:
    order = lambda seed: [v for v, _ in readings(mini_frames, seed)]  # noqa: E731
    assert order(1) == order(1) and order(1) != order(2)


def test_replay_rejects_trucks_without_specifications(mini_frames: dict[str, pd.DataFrame]) -> None:
    with pytest.raises(ValueError, match="without specification"):
        list(sim.iter_readings(mini_frames["readouts"], mini_frames["specs"].iloc[5:], 1))


# --- the loop -------------------------------------------------------------------------------
def test_run_stream_sends_max_readings_with_the_interval(
    mini_frames: dict[str, pd.DataFrame],
) -> None:
    client, sleeps = FakeClient(), []
    stats = sim.run_stream(client, readings(mini_frames), 0.5, 7, sleep=sleeps.append)
    assert (stats.sent, stats.errors, stats.alerts) == (7, 0, 4)
    assert len(client.bodies) == 7 and sleeps == [0.5] * 7


def test_run_stream_stops_when_the_data_ends(mini_frames: dict[str, pd.DataFrame]) -> None:
    stats = sim.run_stream(
        FakeClient(), readings(mini_frames, vehicles=1), 0, 1000, sleep=lambda s: None
    )
    assert stats.sent == 12


def test_run_stream_applies_the_drift_only_after_its_start(
    mini_frames: dict[str, pd.DataFrame],
) -> None:
    client = FakeClient()
    drift = sim.Drift(start=3, factor=3.0, ramp=1)
    stats = sim.run_stream(client, readings(mini_frames), 0, 5, drift, sleep=lambda s: None)
    base = [b for _, b in zip(range(5), readings(mini_frames), strict=False)]
    c = counter_columns()[0]
    for i, (sent, (_, original)) in enumerate(zip(client.bodies, base, strict=True)):
        expected = 3.0 if i >= 3 else 1.0
        assert sent["current"][c] == pytest.approx(original["current"][c] * expected)
    assert stats.sent == 5


def test_run_stream_counts_errors_and_gives_up_after_five_in_a_row(
    mini_frames: dict[str, pd.DataFrame],
) -> None:
    flaky = FakeClient(fail_from=2)
    with pytest.raises(ApiError):
        sim.run_stream(flaky, readings(mini_frames), 0, 50, sleep=lambda s: None)
    assert len(flaky.bodies) == 2 + 5  # two good ones, then five failures in a row

    class Recovering(FakeClient):
        def predict(self, body: dict[str, Any]) -> dict[str, Any]:
            if len(self.bodies) in (1, 2):
                self.bodies.append(body)
                raise ApiError("blip")
            return super().predict(body)

    stats = sim.run_stream(Recovering(), readings(mini_frames), 0, 6, sleep=lambda s: None)
    assert (stats.sent, stats.errors) == (4, 2)  # a recovery resets the in-a-row counter


def test_parser_defaults_and_drift_arguments() -> None:
    args = sim.build_parser().parse_args([])
    assert (args.interval, args.max_readings, args.drift_start) == (1.0, 100, None)
    args = sim.build_parser().parse_args(["--drift-start", "20", "--drift-factor", "2"])
    assert (args.drift_start, args.drift_factor, args.drift_ramp) == (20, 2.0, 30)


def test_tail_replays_only_the_last_readouts_each_with_a_previous(
    mini_frames: dict[str, pd.DataFrame],
) -> None:
    items = list(readings(mini_frames, tail=4))
    assert len(items) == 3 * 4
    assert all("previous" in body for _, body in items)  # the 4th last readout has a predecessor
    first_steps = {
        vid: min(b["current"]["time_step"] for v, b in items if v == vid) for vid in range(3)
    }
    assert set(first_steps.values()) == {80.0}  # readouts are at 0, 10, ..., 110
    assert max(b["current"]["time_step"] for _, b in items) == 110.0
    assert len(list(readings(mini_frames, tail=99))) == 3 * 12  # more than exists = everything


def test_tail_must_be_positive(mini_frames: dict[str, pd.DataFrame]) -> None:
    with pytest.raises(ValueError, match="tail"):
        list(readings(mini_frames, tail=0))


def test_parser_tail_default_and_zero_for_everything() -> None:
    assert sim.build_parser().parse_args([]).tail == 10
    assert sim.build_parser().parse_args(["--tail", "0"]).tail == 0
