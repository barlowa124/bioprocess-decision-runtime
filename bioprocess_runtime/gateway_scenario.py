"""Build a Scenario from a lab-instrument-gateway capture database.

The gateway (lablink) records an emulated bioreactor's typed readings
(channel, value, unit, quality) plus fault-injection alarms. This module
maps a capture window onto the policy's observation domain so a real
instrument session exercises the same decision surface as the built-in
synthetic scenarios.

Channel mapping: DO -> dissolved_oxygen_pct (and a least-squares slope),
AGIT -> agitation_rpm, any non-ok reading quality or recorded alarm in
the window -> sensor_agreement=False. Advisory-only scope is unchanged:
this produces input data for the policy evaluator, not a new authority.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .domain import Observation
from .simulator import Scenario

# Slope needs at least two ok DO readings in the window.
_SLOPE_MIN_POINTS = 2
_SECONDS_PER_MINUTE = 60.0
# Default analysis window: newest 2 minutes of the capture.
_DEFAULT_WINDOW_S = 120


def _to_dt(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def scenario_from_capture(db_path: str | Path, name: str = "gateway_capture",
                          expected_status: str = "NO_ACTION",
                          window_s: int = _DEFAULT_WINDOW_S,
                          description: str = "") -> Scenario:
    """Read a lablink capture.sqlite and produce a policy Scenario over
    the newest `window_s` seconds of readings."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT ts, channel, value, unit, quality FROM readings "
            "ORDER BY id").fetchall()
        has_alarms = con.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' "
            "AND name='alarms'").fetchone()[0]
        alarms = (con.execute("SELECT COUNT(*) FROM alarms").fetchone()[0]
                  if has_alarms else 0)
    finally:
        con.close()
    if not rows:
        raise ValueError(f"{db_path}: no readings")

    end = max(_to_dt(r[0]) for r in rows)
    window = [r for r in rows if (end - _to_dt(r[0])).total_seconds() <= window_s]

    def latest(channel):
        vals = [r for r in window if r[1] == channel and r[4] == "ok"]
        return vals[-1] if vals else None

    do = latest("DO")
    agit = latest("AGIT")

    # DO slope: least-squares over ok DO readings in the window,
    # percent per minute.
    do_rows = [(_to_dt(r[0]), r[2]) for r in window
               if r[1] == "DO" and r[4] == "ok"]
    slope = 0.0
    if len(do_rows) >= _SLOPE_MIN_POINTS:
        t0 = do_rows[0][0]
        xs = [(t - t0).total_seconds() / _SECONDS_PER_MINUTE
              for t, _ in do_rows]
        ys = [v for _, v in do_rows]
        xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
        denom = sum((x - xm) ** 2 for x in xs)
        if denom > 0:
            slope = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / denom

    bad_quality = any(r[4] != "ok" for r in window)
    agreement = not (bad_quality or alarms > 0)

    observations = {
        "dissolved_oxygen_pct": Observation(
            do[2] if do else float("nan"), "percent",
            _to_dt(do[0]) if do else end,
            f"lablink.capture:{db_path}:DO",
            "valid" if do else "missing"),
        "dissolved_oxygen_slope": Observation(
            slope, "percent_per_minute",
            do_rows[-1][0] if do_rows else end,
            f"lablink.capture:{db_path}:DO (least-squares)",
            "valid" if len(do_rows) >= _SLOPE_MIN_POINTS else "missing"),
        "agitation_rpm": Observation(
            agit[2] if agit else float("nan"), "rpm",
            _to_dt(agit[0]) if agit else end,
            f"lablink.capture:{db_path}:AGIT",
            "valid" if agit else "missing"),
        "sensor_agreement": Observation(
            agreement, "boolean", end,
            f"lablink.capture:{db_path}:quality+alarms"),
    }
    return Scenario(
        name=name,
        description=description or
            f"Window ending {end.isoformat()} of lablink capture {db_path}.",
        observations=observations,
        evaluated_at=end,
        expected_status=expected_status,
    )
