from __future__ import annotations

import copy
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bioprocess_runtime import gemma_checkpoint as storage
from bioprocess_runtime import gemma_independent as engine
from bioprocess_runtime import gemma_independent_run as runner
from bioprocess_runtime.gemma_rotary_slice import _seal
from test_gemma_independent import synthetic
from test_gemma_two_layers import synthetic_context
from test_gemma_first_layer import profiles

ROOT = Path(__file__).resolve().parents[1]


class CheckpointStorageTests(unittest.TestCase):
    def test_atomic_generations_and_existing_file_protection(self):
        with tempfile.TemporaryDirectory() as parent:
            directory = Path(parent) / "run"
            with storage.CheckpointStore(directory, {"input": "one"}) as store:
                first = store.save({"completed_instruction_count": 0}, {})
                second = store.save({"completed_instruction_count": 3}, {"live": [1]})
                self.assertEqual(second["parent_checkpoint_sha256"], first["checkpoint_sha256"])
                with self.assertRaises(ValueError):
                    store.save({"completed_instruction_count": 3}, {})
            with storage.CheckpointStore(directory, {"input": "one"}, resume=True) as store:
                self.assertEqual(store.latest, second)
            with self.assertRaises(FileExistsError):
                with storage.CheckpointStore(directory, {"input": "one"}):
                    pass
            original = directory / "existing.json"
            original.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                storage._publish(original, {"changed": True})
            self.assertEqual(original.read_text(), "preserve")
            self.assertFalse(list(directory.glob("*.tmp")))

    def test_incomplete_temporary_file_ignored_but_corrupt_latest_rejected(self):
        with tempfile.TemporaryDirectory() as parent:
            directory = Path(parent) / "run"
            with storage.CheckpointStore(directory, {}) as store:
                store.save({"completed_instruction_count": 0}, {})
                store.save({"completed_instruction_count": 8}, {})
            (directory / ".checkpoint-interrupted.tmp").write_text("{broken", encoding="utf-8")
            with storage.CheckpointStore(directory, {}, resume=True) as store:
                self.assertEqual(store.last_index, 8)
            latest = next(directory.glob("checkpoint-000008-*.json"))
            latest.write_text("{broken", encoding="utf-8")
            with self.assertRaises(ValueError):
                with storage.CheckpointStore(directory, {}, resume=True):
                    pass

    def test_publish_failure_retains_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as parent:
            directory = Path(parent) / "run"
            with storage.CheckpointStore(directory, {}) as store:
                initial = store.save({"completed_instruction_count": 0}, {})
                with patch.object(storage.os, "link", side_effect=OSError("simulated interruption before publish")), self.assertRaises(OSError):
                    store.save({"completed_instruction_count": 1}, {})
                self.assertEqual(store.latest, initial)
            with storage.CheckpointStore(directory, {}, resume=True) as store:
                self.assertEqual(store.last_index, 0)

    def test_writer_lock_released_by_process_death(self):
        with tempfile.TemporaryDirectory() as parent:
            directory = Path(parent) / "run"
            script = "from pathlib import Path; import os,sys; from bioprocess_runtime.gemma_checkpoint import CheckpointStore; store=CheckpointStore(Path(sys.argv[1]),{}); store.__enter__(); store.save({'completed_instruction_count':0},{}); print(os.getpid(),flush=True); sys.stdin.read()"
            process = subprocess.Popen([sys.executable, "-c", script, str(directory)], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                owner_pid = int(process.stdout.readline().strip())
                self.assertGreater(owner_pid, 0)
                with self.assertRaises(OSError):
                    with storage.CheckpointStore(directory, {}, resume=True):
                        pass
                os.kill(owner_pid, signal.SIGTERM if os.name == "nt" else signal.SIGKILL)
                process.wait(timeout=10)
                with storage.CheckpointStore(directory, {}, resume=True) as store:
                    self.assertEqual(store.last_index, 0)
            finally:
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=10)

    def test_changed_binding_and_ambiguous_branch_rejected(self):
        with tempfile.TemporaryDirectory() as parent:
            directory = Path(parent) / "run"
            with storage.CheckpointStore(directory, {"runtime": "one"}) as store:
                saved = store.save({"completed_instruction_count": 0}, {})
            with self.assertRaises(ValueError):
                with storage.CheckpointStore(directory, {"runtime": "two"}, resume=True):
                    pass
            original = next(directory.glob("checkpoint-*.json"))
            alternate = directory / ("checkpoint-000000-" + "0" * 64 + ".json")
            alternate.write_bytes(original.read_bytes())
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                with storage.CheckpointStore(directory, {"runtime": "one"}, resume=True):
                    pass


class EngineCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.program, cls.ids, cls.snapshots, cls.providers = synthetic_context()

    def run_engine(self, **options):
        return engine.execute(self.program, self.ids, self.snapshots, self.providers, profiles(), {"synthetic": True}, target="hidden.2", workers=1, **options)

    def interrupt(self, directory, index, retain=False):
        def progress(event):
            if event["event"] == "instruction_complete" and event["index"] == index:
                raise RuntimeError("test interruption")
        with self.assertRaisesRegex(RuntimeError, "test interruption"):
            self.run_engine(checkpoint_dir=directory, checkpoint_every=1, retain_states=retain, progress=progress)

    def test_resumed_prefix_matches_uninterrupted_states_records_and_auxiliaries(self):
        for retain in (False, True):
            with self.subTest(retain=retain), tempfile.TemporaryDirectory() as parent, synthetic(self.program):
                directory = Path(parent) / "run"
                expected = self.run_engine(retain_states=retain)
                self.interrupt(directory, 18, retain)
                events = []
                actual = self.run_engine(checkpoint_dir=directory, resume=True, checkpoint_every=8, retain_states=retain, progress=events.append)
                self.assertTrue(actual["stored_boundary_predictions_used"])
                self.assertEqual(actual["checkpoint_resume"]["completed_instruction_count"], 18)
                indices = [event["index"] for event in events if event["event"] == "instruction_complete"]
                self.assertEqual(indices, list(range(19, 64)))
                engine.check_execution(self.program, actual)
                comparable = copy.deepcopy(actual)
                comparable["checkpoint_resume"] = None
                comparable["stored_boundary_predictions_used"] = False
                self.assertEqual(comparable, expected)
                complete = self.run_engine(checkpoint_dir=directory, resume=True, retain_states=retain)
                self.assertEqual(complete["checkpoint_resume"]["completed_instruction_count"], 63)
                self.assertEqual(complete["records"], actual["records"])

    def test_resealed_live_state_and_missing_state_reject_before_any_new_operation(self):
        for kind in ("value", "missing"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as parent, synthetic(self.program):
                directory = Path(parent) / "run"
                self.interrupt(directory, 9)
                path = next(directory.glob("checkpoint-000009-*.json"))
                checkpoint = json.loads(path.read_text())
                name = next(iter(checkpoint["live_states"]))
                if kind == "missing":
                    checkpoint["live_states"].pop(name)
                else:
                    value = checkpoint["live_states"][name]
                    while isinstance(value[0], list):
                        value = value[0]
                    value[0] ^= 1
                checkpoint = storage.seal({key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"}, "checkpoint_sha256")
                path.unlink()
                (directory / ("checkpoint-000009-" + checkpoint["checkpoint_sha256"] + ".json")).write_text(json.dumps(checkpoint), encoding="utf-8")
                with patch.object(engine, "_operation", side_effect=AssertionError("must reject before computation")), self.assertRaises(ValueError):
                    self.run_engine(checkpoint_dir=directory, resume=True)

    def test_caller_source_context_and_retention_mismatch_fail_closed(self):
        with tempfile.TemporaryDirectory() as parent, synthetic(self.program):
            directory = Path(parent) / "run"
            self.interrupt(directory, 9)
            with self.assertRaises(ValueError):
                self.run_engine(checkpoint_dir=directory, resume=True, checkpoint_context={"different_runner": True})
            with self.assertRaises(ValueError):
                self.run_engine(checkpoint_dir=directory, resume=True, retain_states=True)
            with self.assertRaises(ValueError):
                self.run_engine(checkpoint_dir=directory, resume=True, vocabulary_candidate=True)

    def test_engine_source_change_rejects_checkpoint_before_any_new_operation(self):
        with tempfile.TemporaryDirectory() as parent, synthetic(self.program):
            directory = Path(parent) / "run"
            self.interrupt(directory, 9)
            with patch.object(engine, "code_sha256", return_value="different-source-version"), patch.object(engine, "_operation", side_effect=AssertionError("no cross-version execution")), self.assertRaises(ValueError):
                self.run_engine(checkpoint_dir=directory, resume=True)

    def test_intermediate_checkpoint_is_not_a_completed_or_abstained_prediction(self):
        with tempfile.TemporaryDirectory() as parent, synthetic(self.program):
            directory = Path(parent) / "run"
            self.interrupt(directory, 9)
            saved = json.loads(next(directory.glob("checkpoint-000009-*.json")).read_text())["execution"]
            self.assertEqual(saved["status"], "checkpoint")
            self.assertIsNone(saved["selected_token_id"])
            engine.check_execution(self.program, saved, allow_checkpoint=True)
            with self.assertRaises(ValueError):
                engine.check_execution(self.program, saved)

    def test_numeric_resume_in_fresh_process_after_writer_termination(self):
        with tempfile.TemporaryDirectory() as parent:
            directory = Path(parent) / "run"
            script = """
from pathlib import Path
import os, sys
sys.path.insert(0, str(Path.cwd() / 'tests'))
from test_gemma_independent import synthetic
from test_gemma_two_layers import synthetic_context
from test_gemma_first_layer import profiles
from bioprocess_runtime import gemma_independent as engine
from bioprocess_runtime.gemma_rotary_slice import _sha
program, ids, snapshots, providers = synthetic_context()
resume = sys.argv[2] == 'resume'
def progress(event):
    if not resume and event['event'] == 'instruction_complete' and event['index'] == 18:
        print(os.getpid(), flush=True)
        sys.stdin.read()
with synthetic(program):
    result = engine.execute(program, ids, snapshots, providers, profiles(), {'synthetic':True}, target='hidden.2', workers=1, checkpoint_dir=Path(sys.argv[1]), checkpoint_every=1, resume=resume, progress=progress)
result['checkpoint_resume'] = None
result['stored_boundary_predictions_used'] = False
print(_sha(result), flush=True)
"""
            process = subprocess.Popen([sys.executable, "-c", script, str(directory), "interrupt"], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                owner = int(process.stdout.readline().strip())
                os.kill(owner, signal.SIGTERM if os.name == "nt" else signal.SIGKILL)
                process.wait(timeout=10)
            finally:
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=10)
            resumed = subprocess.run([sys.executable, "-c", script, str(directory), "resume"], cwd=ROOT, capture_output=True, text=True, timeout=120, check=True)
            with synthetic(self.program):
                expected = self.run_engine()
            from bioprocess_runtime.gemma_rotary_slice import _sha
            self.assertEqual(resumed.stdout.strip(), _sha(expected))

    def test_operational_resume_metadata_is_separate_from_numerical_equality(self):
        with synthetic(self.program):
            fresh = self.run_engine()
        resumed = copy.deepcopy(fresh)
        resumed["checkpoint_resume"] = {"checkpoint_sha256": "1" * 64, "completed_instruction_count": 18}
        resumed["stored_boundary_predictions_used"] = True
        a = _seal({"execution": fresh}, "prediction_sha256")
        b = _seal({"execution": resumed}, "prediction_sha256")
        self.assertTrue(runner.same_numerical_prediction(a, b))
        changed = copy.deepcopy(b)
        changed["execution"]["state_bits"]["hidden.2"][0][0][0] ^= 1
        changed = _seal({key: value for key, value in changed.items() if key != "prediction_sha256"}, "prediction_sha256")
        self.assertFalse(runner.same_numerical_prediction(a, changed))
