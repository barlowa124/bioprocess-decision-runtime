from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

SPEC = importlib.util.spec_from_file_location('package_handoff', Path(__file__).resolve().parents[1] / 'tools/package_handoff.py')
package = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package)


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.git = patch.object(package, 'git', side_effect=lambda root, *args: b'abc123\n' if args[0] == 'rev-parse' else b'')
        self.git.start()
        self.addCleanup(self.git.stop)

    def records(self):
        (self.root / 'results').mkdir()
        (self.root / 'results/evidence.json').write_bytes(b'{"value":1}\r\n')
        (self.root / 'payload.bin').write_bytes(bytes(range(256)))
        return [{'path': name, 'bytes': (self.root/name).stat().st_size} for name in ('results/evidence.json', 'payload.bin')]

    def test_archive_preserves_mixed_bytes_and_verifies_every_member(self):
        records = self.records()
        output = self.root / 'dist/handoff.zip'
        result = package.build_archive(self.root, output, records, True)
        self.assertTrue(result['verified'])
        self.assertEqual(result['files'], 4)
        manifest = package.verify_archive(output)
        self.assertEqual(manifest['kind'], 'local_runtime_handoff')
        with zipfile.ZipFile(output) as archive:
            self.assertEqual(archive.read('results/evidence.json'), b'{"value":1}\r\n')
            self.assertEqual(archive.read('payload.bin'), bytes(range(256)))
        self.assertEqual(result['sha256'], hashlib.sha256(output.read_bytes()).hexdigest())

    def test_generated_launcher_is_lightweight_and_requirements_are_pinned(self):
        environment = {'packages': {'numpy':'2.2.6', 'bioprocess-decision-runtime':'0.1.0', 'pip':'25.0'}}
        output = self.root / 'generated.zip'
        with patch.object(package, 'environment_manifest', return_value=environment):
            package.build_archive(self.root, output, [], True)
        with zipfile.ZipFile(output) as archive:
            self.assertEqual(archive.read('handoff-requirements.txt'), b'numpy==2.2.6\n')
            launcher = archive.read('launch-demo.cmd').decode('utf-8')
            self.assertIn('demo_ui --port 0', launcher)
            self.assertNotIn('gemma_independent', launcher)
            self.assertNotIn('pip install', launcher)
        self.assertEqual(len(package.verify_archive(output)['files']), 2)

    def test_mac_demo_bundle_is_small_stdlib_only_and_executable(self):
        root = Path(__file__).resolve().parents[1]
        log = self.root / 'synthetic-regression.log'
        log.write_text('Ran 625 tests in 1.0s\nOK\nNo broken requirements found.\nVERIFICATION_EXIT_CODE=0\n')
        original_safe_file = package.safe_file
        lookup = patch.object(package, 'safe_file', side_effect=lambda base, name: log if name == 'gemma_independent_holdout_verification_v3.log' else original_safe_file(base, name))
        lookup.start()
        self.addCleanup(lookup.stop)
        with patch.object(package, 'runtime_paths', side_effect=AssertionError('no runtime assets')):
            records = package.inventory(root, False, demo_only=True)
        names = {row['path'] for row in records}
        self.assertIn('gemma_independent_holdout_verification_v3.log', names)
        self.assertNotIn('bioprocess_runtime/gemma_independent_run.py', names)
        self.assertFalse(any(name.startswith(('.models/', 'artifacts/')) for name in names))
        output = self.root / 'mac-demo.zip'
        with self.assertRaises(ValueError):
            package.build_archive(root, self.root / 'incomplete.zip', records[:-1], False, demo_only=True)
        with patch.object(package, 'environment_manifest', side_effect=AssertionError('no Windows dependency pins')):
            result = package.build_archive(root, output, records, False, demo_only=True)
        manifest = package.verify_archive(output)
        self.assertEqual(manifest['kind'], 'mac_demo_handoff')
        self.assertEqual(manifest['environment']['dependencies'], 'Python standard library only')
        self.assertLess(result['bytes'], 32 * 1024 * 1024)
        with zipfile.ZipFile(output) as archive:
            launcher = archive.read('Launch Demo.command')
            self.assertNotIn(b'\r', launcher)
            self.assertIn(b'--open-browser', launcher)
            self.assertIn(b'3, 11', launcher)
            self.assertNotIn('handoff-requirements.txt', archive.namelist())
            self.assertEqual(archive.getinfo('Launch Demo.command').external_attr >> 16 & 0o777, 0o755)
            extracted = self.root / 'extracted demo'
            archive.extractall(extracted)
        script = "import sys; from pathlib import Path; root=Path(sys.argv[1]); sys.path.insert(0,str(root)); from bioprocess_runtime import demo_ui as ui; assert Path(ui.__file__).is_relative_to(root); data=ui.catalog(root); assert data['full_target_replay_recorded']; assert data['verification']['tests']==625 and data['verification']['complete']; assert ui.run_scenario(root,'low_oxygen')['matches_expected']; assert not any(name in sys.modules for name in ('torch','numpy','sklearn')); print('EXTRACTED_DEMO_STDLIB_ONLY=True')"
        completed = subprocess.run([sys.executable, '-I', '-c', script, str(extracted)], cwd=extracted, check=True, capture_output=True, text=True, timeout=30)
        self.assertIn('EXTRACTED_DEMO_STDLIB_ONLY=True', completed.stdout)

    def test_existing_archive_is_never_overwritten(self):
        output = self.root / 'old.zip'
        output.write_bytes(b'preserve')
        with self.assertRaises(ValueError):
            package.build_archive(self.root, output, [], False)
        self.assertEqual(output.read_bytes(), b'preserve')

    def test_unsafe_and_private_paths_are_rejected(self):
        for name in ('../outside', '/absolute', 'C:/outside', 'nested\\file', '.env', 'key.pem', '.venv/module.py'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                package.safe_file(self.root, name)

    def test_missing_runtime_file_is_not_silently_omitted(self):
        with self.assertRaises(ValueError):
            package.build_archive(self.root, self.root/'missing.zip', [{'path':'missing.bin','bytes':1}], True)
        self.assertFalse((self.root/'missing.zip').exists())
        self.assertFalse(list(self.root.glob('*.partial')))

    def test_insufficient_disk_space_fails_before_writing_archive(self):
        with patch.object(package.shutil, 'disk_usage', return_value=type('Usage', (), {'free':0})()), self.assertRaises(ValueError):
            package.build_archive(self.root, self.root/'low.zip', [], False)
        self.assertFalse((self.root/'low.zip').exists())

    def test_corrupt_or_duplicate_manifest_entries_fail(self):
        for duplicate in (False, True):
            path = self.root / ('duplicate.zip' if duplicate else 'corrupt.zip')
            entry = {'path':'data.txt','bytes':3,'sha256':hashlib.sha256(b'abc').hexdigest()}
            with zipfile.ZipFile(path, 'x') as archive:
                archive.writestr('data.txt', b'abc' if duplicate else b'xyz')
                archive.writestr('handoff-manifest.json', json.dumps({'files':[entry,entry] if duplicate else [entry]}))
            with self.assertRaises(ValueError):
                package.verify_archive(path)

    def test_source_inventory_is_allowlisted(self):
        with patch.object(package, 'git', return_value=b'README.md\0bioprocess_runtime/core.py\0results/result.json\0.env\0.venv/bin\0artifacts/raw.json\0notes/private.md\0'):
            self.assertEqual(package.source_paths(self.root), {'README.md','bioprocess_runtime/core.py','results/result.json'})


if __name__ == '__main__':
    unittest.main()
