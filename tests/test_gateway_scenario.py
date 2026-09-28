"""Scenario construction from a lab-instrument-gateway capture."""
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bioprocess_runtime.gateway_scenario import scenario_from_capture


def _capture_db(path: Path, ok: bool = True) -> Path:
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE readings (id INTEGER PRIMARY KEY "
                "AUTOINCREMENT, ts TEXT, channel TEXT, value REAL, "
                "unit TEXT, quality TEXT)")
    con.execute("CREATE TABLE alarms (id INTEGER PRIMARY KEY "
                "AUTOINCREMENT, ts TEXT, channel TEXT, rule TEXT, "
                "value REAL)")
    t0 = datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc)
    rows = []
    for i in range(10):
        ts = (t0 + timedelta(seconds=30 * i)).isoformat()
        rows += [
            (ts, "DO", 30.0 - 0.5 * i, "percent", "ok"),
            (ts, "AGIT", 90.0, "rpm", "ok"),
        ]
    if not ok:
        rows[-1] = (rows[-1][0], "AGIT", None, "rpm", "device-error")
        con.execute("INSERT INTO alarms (ts, channel, rule, value) "
                    "VALUES (?,?,?,?)",
                    (rows[-1][0], "AGIT", "device-error", 0))
    con.executemany(
        "INSERT INTO readings (ts, channel, value, unit, quality) "
        "VALUES (?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return path


class GatewayScenarioTests(unittest.TestCase):
    def test_clean_capture_maps_to_observations(self):
        with tempfile.TemporaryDirectory() as td:
            db = _capture_db(Path(td) / "cap.sqlite")
            s = scenario_from_capture(db, expected_status="RECOMMENDATION")
            o = s.observations
            self.assertEqual(o["dissolved_oxygen_pct"].value, 25.5)
            self.assertAlmostEqual(o["dissolved_oxygen_slope"].value,
                                   -1.0, places=6)  # -0.5 per 30s
            self.assertEqual(o["agitation_rpm"].value, 90.0)
            self.assertTrue(o["sensor_agreement"].value)
            self.assertEqual(s.expected_status, "RECOMMENDATION")

    def test_fault_marks_disagreement(self):
        with tempfile.TemporaryDirectory() as td:
            db = _capture_db(Path(td) / "cap.sqlite", ok=False)
            s = scenario_from_capture(db)
            # last AGIT row is device-error. The prior ok row still
            # provides a value, but the error flips sensor_agreement.
            self.assertFalse(s.observations["sensor_agreement"].value)
            self.assertEqual(s.observations["agitation_rpm"].value, 90.0)

    def test_empty_capture_raises(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "empty.sqlite"
            sqlite3.connect(db).execute(
                "CREATE TABLE readings (id INTEGER PRIMARY KEY, ts TEXT, "
                "channel TEXT, value REAL, unit TEXT, quality TEXT)")
            with self.assertRaises(ValueError):
                scenario_from_capture(db)


if __name__ == "__main__":
    unittest.main()
