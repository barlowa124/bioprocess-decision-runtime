from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .domain import Observation


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    observations: dict[str, Observation]
    evaluated_at: datetime
    expected_status: str


_BASE_TIME = datetime(2026, 1, 15, 12, 0, 10, tzinfo=timezone.utc)


def _observations(
    dissolved_oxygen_pct: float = 45.0,
    dissolved_oxygen_slope: float = -0.1,
    agitation_rpm: float = 90.0,
    sensor_agreement: bool = True,
    age_seconds: int = 5,
    oxygen_unit: str = "percent",
) -> dict[str, Observation]:
    observed_at = _BASE_TIME - timedelta(seconds=age_seconds)
    return {
        "dissolved_oxygen_pct": Observation(dissolved_oxygen_pct, oxygen_unit, observed_at, "synthetic.do_sensor.primary"),
        "dissolved_oxygen_slope": Observation(dissolved_oxygen_slope, "percent_per_minute", observed_at, "synthetic.derived.do_slope"),
        "agitation_rpm": Observation(agitation_rpm, "rpm", observed_at, "synthetic.agitator.feedback"),
        "sensor_agreement": Observation(sensor_agreement, "boolean", observed_at, "synthetic.sensor_vote"),
    }


def scenarios() -> dict[str, Scenario]:
    return {
        "normal": Scenario(
            "normal",
            "Nominal synthetic state where the transparent model remains below the action threshold.",
            _observations(),
            _BASE_TIME,
            "NO_ACTION",
        ),
        "low_oxygen": Scenario(
            "low_oxygen",
            "Synthetic low dissolved-oxygen trend with agreeing sensors and room inside the demonstration envelope.",
            _observations(dissolved_oxygen_pct=30.0, dissolved_oxygen_slope=-1.0),
            _BASE_TIME,
            "RECOMMENDATION",
        ),
        "sensor_disagreement": Scenario(
            "sensor_disagreement",
            "The model score is high, but redundant synthetic sensors disagree, requiring abstention.",
            _observations(dissolved_oxygen_pct=30.0, dissolved_oxygen_slope=-1.0, sensor_agreement=False),
            _BASE_TIME,
            "ABSTAIN",
        ),
        "recommendation_limit": Scenario(
            "recommendation_limit",
            "The model score is high, but the proposed adjustment exceeds the configured demonstration maximum.",
            _observations(dissolved_oxygen_pct=30.0, dissolved_oxygen_slope=-1.0, agitation_rpm=118.0),
            _BASE_TIME,
            "ABSTAIN",
        ),
        "stale_input": Scenario(
            "stale_input",
            "All observations are older than the policy permits.",
            _observations(dissolved_oxygen_pct=30.0, dissolved_oxygen_slope=-1.0, age_seconds=120),
            _BASE_TIME,
            "ABSTAIN",
        ),
        "unit_mismatch": Scenario(
            "unit_mismatch",
            "Dissolved oxygen is supplied using a unit that does not match the policy declaration.",
            _observations(oxygen_unit="fraction"),
            _BASE_TIME,
            "ABSTAIN",
        ),
    }
