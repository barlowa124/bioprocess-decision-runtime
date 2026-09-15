from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
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
