from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from bioprocess_runtime.gemma_failure_targets import ROOT, build_report, extract_targets


class FailureTargetTests(unittest.TestCase):
    def setUp(self):
        self.probe_path = ROOT / 'results/gemma3_270m_copy_drift_v1.json'
        self.protocol_path = ROOT / 'results/gemma3_270m_copy_drift_protocol_v1.md'
        self.heldout_path = ROOT / 'results/gemma3_270m_copy_drift_heldout_v1.json'
        self.baseline_path = ROOT / 'results/gemma3_270m_operational_semantics_summary.json'
        self.probe = json.loads(self.probe_path.read_bytes())

    def report(self, probe_path=None):
        return build_report(probe_path or self.probe_path, self.protocol_path, self.heldout_path, self.baseline_path)

    def test_saved_failure_targets_and_original_labels(self):
        original = copy.deepcopy(self.probe)
        targets = extract_targets(self.probe)
        self.assertEqual(self.probe, original)
        self.assertEqual([t['source_case_index'] for t in targets], [3, 12, 16, 17, 20, 21])
        self.assertEqual([t['generated_step_zero_based'] for t in targets], [4, 4, 3, 3, 3, 2])
        self.assertEqual([t['expected_digit'] for t in targets], ['7', '3', '3', '3', '0', '6'])
        self.assertEqual([t['selected_minus_expected_recorded_score'] for t in targets[1:]], [2.1875, 4.0625, 1.75, 6.125, 4.5])
        self.assertTrue(all(t['recorded_score_class'] == 'misattribution' for t in targets))
        self.assertTrue(all(t['expected_digit_in_recorded_top5'] for t in targets[1:]))
        self.assertTrue(all(t['prompt_token_count'] is None for t in targets))
        self.assertTrue(all(not t['mechanism_established'] and not t['independent_engine_coverage_established'] for t in targets))

    def test_numeric_prefix_shortfall_does_not_invent_missing_scores(self):
        target = extract_targets(self.probe)[0]
        self.assertEqual(target['inspection_kind'], 'numeric_prefix_shortfall')
        self.assertEqual(target['recorded_prefix_text'], '0.64')
        self.assertEqual(target['observed_token_id'], 107)
        for key in ('recorded_top5', 'observed_digit', 'expected_digit_in_recorded_top5', 'selected_minus_expected_recorded_score'):
            self.assertIsNone(target[key])

    def test_expected_digit_missing_from_top5_is_unknown_score(self):
        row = self.probe['cases'][12]['generation']['digit_token_top5'][-1]
        row['top5'][1]['token'] = '8'
        target = extract_targets(self.probe)[1]
        self.assertFalse(target['expected_digit_in_recorded_top5'])
        self.assertIsNone(target['selected_minus_expected_recorded_score'])

    def test_bad_steps_ids_scores_and_alignment_reject(self):
        for mode in ('duplicate_step', 'token_id', 'nonfinite', 'text', 'asked', 'nonmaximal'):
            changed = copy.deepcopy(self.probe)
            case = changed['cases'][12]
            rows = case['generation']['digit_token_top5']
            if mode == 'duplicate_step':
                rows.append(copy.deepcopy(rows[-1]))
            elif mode == 'token_id':
                rows[-1]['token_id'] += 1
            elif mode == 'nonfinite':
                rows[-1]['top5'][0]['logit'] = float('nan')
            elif mode == 'text':
                case['generation']['generated_text'] = 'The answer is 0.647'
            elif mode == 'asked':
                case['asked_value'] = 0.647
            else:
                rows[-1]['top5'][0]['logit'] = -100
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                extract_targets(changed)

    def test_source_hashes_and_sealed_prompt_order_reject_tampering(self):
        for mode in ('protocol', 'heldout', 'prompt', 'order'):
            changed = copy.deepcopy(self.probe)
            if mode in ('protocol', 'heldout'):
                changed[mode + '_sha256'] = '0' * 64
            elif mode == 'prompt':
                changed['cases'][12]['prompt'] += ' altered'
            else:
                changed['cases'][12], changed['cases'][13] = changed['cases'][13], changed['cases'][12]
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'probe.json'
                path.write_text(json.dumps(changed), encoding='utf-8')
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    self.report(path)

    def test_report_integrity_and_repeatability(self):
        report = self.report()
        self.assertEqual(report, self.report())
        digest = report.pop('report_sha256')
        payload = json.dumps(report, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
        self.assertEqual(digest, hashlib.sha256(payload).hexdigest())
        self.assertFalse(report['model_inference_performed'])
        self.assertEqual(report['recorded_baseline']['token_count'], 30)

    def test_cli_refuses_existing_output_and_does_not_import_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'inspection.json'
            command = [sys.executable, '-m', 'bioprocess_runtime.gemma_failure_targets', '--output', str(output)]
            first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            saved = output.read_bytes()
            second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(output.read_bytes(), saved)
        check = subprocess.run([sys.executable, '-c', "import sys; import bioprocess_runtime.gemma_failure_targets; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(check.returncode, 0, check.stderr)


if __name__ == '__main__':
    unittest.main()
