from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 1024 * 1024
EXTRA_RUNTIME = (
    'artifacts/gemma_vocab_validation_plan_v1.json', 'artifacts/gemma_vocab_validation_bundle_v1.json',
    'artifacts/gemma_vocab_validation_report_v1.json', 'artifacts/gemma_vocab_replay_v1.json',
    'artifacts/gemma_vocab_native_probe_v1.json', 'artifacts/gemma_vocab_validation_v1.py',
    'artifacts/gemma3_270m_independent_baseline_prediction_v2.json', 'artifacts/gemma3_270m_independent_baseline_report_v2.json',
    'artifacts/gemma3_270m_independent_baseline_prediction_v3.json', 'artifacts/gemma3_270m_independent_baseline_report_v3.json',
    'artifacts/gemma3_270m_independent_baseline_replay_checkpoints_v3/checkpoint-000533-14aa6dabda9d421b7538e7033083c238df8abaf04cd075033663472075a6f24b.json',
    'gemma_independent_baseline_v3_replay.log', 'gemma_independent_baseline_v3_replay_supervisor.log',
    'gemma_independent_holdout_verification_v3.log', 'gemma_independent_holdout_execution_v3.log',
    'gemma_independent_holdout_resume1_v3.log', 'gemma_independent_holdout_replay_resume2_v3.log',
    'demo_full_target_lightweight_verification.log', 'handoff_lightweight_verification.log',
)
SOURCE_ROOTS = {'bioprocess_runtime', 'tests', 'results', 'policies', 'tools', '.github'}
SOURCE_FILES = {'AGENTS.md', 'README.md', 'LICENSE', 'pyproject.toml', '.gitignore', '.gitattributes'}


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args])


def safe_file(root, relative):
    name = PurePosixPath(relative)
    if name.is_absolute() or '..' in name.parts or '\\' in relative or ':' in relative:
        raise ValueError('Unsafe package path: ' + relative)
    path = root / relative
    linked = any(parent.is_symlink() or getattr(parent, 'is_junction', lambda: False)() for parent in (path, *path.parents) if parent != root and parent.is_relative_to(root))
    if linked or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('Missing, external or linked package file: ' + relative)
    if any(part in ('.git', '.venv', '__pycache__', '.cache') for part in name.parts):
        raise ValueError('Excluded package directory: ' + relative)
    if name.name.startswith('.env') or name.suffix in ('.pyc', '.pyo', '.pem', '.key'):
        raise ValueError('Excluded package file: ' + relative)
    return path


def source_paths(root):
    names = git(root, 'ls-files', '-z', '--cached', '--others', '--exclude-standard').decode('utf-8').split('\0')
    return {name for name in names if name and (name in SOURCE_FILES or PurePosixPath(name).parts[0] in SOURCE_ROOTS)}


def runtime_paths(root):
    sys.path.insert(0, str(root))
    from bioprocess_runtime import cli
    args = cli.build_parser().parse_args(['gemma-third-layer-scores-verify', 'results/gemma3_270m_execution_ir.json', '--report', 'artifacts/gemma3_270m_third_layer_scores_report.json'])
    names = set(EXTRA_RUNTIME)
    for key, value in vars(args).items():
        if not isinstance(value, Path):
            continue
        path = value if value.is_absolute() else root / value
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Runtime dependency leaves repository')
        if path == root / '.models/gemma-3-270m-it':
            continue
        if key in ('rsqrt_audit_dir', 'exp_audit_dir', 'gelu_audit_dir') and not path.exists():
            continue
        if path.is_dir():
            for member in path.rglob('*'):
                if member.is_file() and member.suffix in ('.json', '.bin', '.npy', '.npz'):
                    names.add(member.relative_to(root).as_posix())
        else:
            names.add(path.relative_to(root).as_posix())
    model = root / '.models/gemma-3-270m-it'
    if not model.is_dir():
        raise ValueError('Fixed model directory is unavailable')
    for path in model.iterdir():
        if path.is_file() and (path.suffix in ('.json', '.safetensors', '.model', '.md', '.txt') or path.name.startswith('LICENSE')):
            names.add(path.relative_to(root).as_posix())
    batch = root / 'artifacts/gemma3_270m_independent_holdout_v3'
    if not batch.is_dir():
        raise ValueError('Completed holdout artifacts are unavailable')
    names.update(path.relative_to(root).as_posix() for path in batch.glob('*.json'))
    for pattern in ('gemma_vocab*.py', 'finalize_holdout_evidence_v3.py', 'gemma_independent*_sources-*.pack', 'gemma_independent*_sources-*.idx'):
        names.update(path.relative_to(root).as_posix() for path in (root / 'artifacts').glob(pattern))
    return names


def demo_paths(root):
    sys.path.insert(0, str(root))
    from bioprocess_runtime import demo_ui
    modules = ('__init__', 'demo_ui', 'demo_assets', 'domain', 'policy', 'runtime', 'serialization', 'simulator')
    names = {f'bioprocess_runtime/{name}.py' for name in modules}
    names.update(('README.md', 'LICENSE', 'policies/oxygen_advisory.bpr', 'gemma_independent_holdout_verification_v3.log',
                  'results/gemma3_270m_execution_ir.json', 'results/gemma3_270m_independent_baseline_v2_diagnosis.json',
                  'results/gemma_vocab_validation_summary_v1.json'))
    names.update(f'results/gemma3_270m_{spec.stem}_{kind}.json' for spec in demo_ui.EVIDENCE for kind in ('plan', 'summary'))
    for name in names:
        if safe_file(root, name).stat().st_size > demo_ui.MAX_FILE_BYTES:
            raise ValueError('Demo-only asset exceeds the UI size limit: ' + name)
    return names


def inventory(root, include_runtime, *, demo_only=False):
    if include_runtime and demo_only:
        raise ValueError('Demo-only packaging cannot include the numerical runtime')
    names = demo_paths(root) if demo_only else source_paths(root)
    if include_runtime:
        names.update(runtime_paths(root))
    records = []
    for name in sorted(names):
        path = safe_file(root, name)
        size = path.stat().st_size
        record = {'path': name, 'bytes': size}
        if PurePosixPath(name).parts[0] not in ('artifacts', '.models') and size < 20 * 1024 * 1024:
            data = path.read_bytes()
            record['crlf_count'] = data.count(b'\r\n')
            record['nul_bytes'] = b'\0' in data
        records.append(record)
    return records


def environment_manifest():
    packages = {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions() if dist.metadata['Name']}
    return {'python': sys.version, 'platform': platform.platform(), 'packages': dict(sorted(packages.items())),
            'scope': 'Installed version inventory only. Not an environment copy, dependency download, CUDA driver installer, or guarantee of compatibility on another machine.'}


def read_digest(stream):
    digest = hashlib.sha256()
    count = 0
    while chunk := stream.read(CHUNK):
        digest.update(chunk)
        count += len(chunk)
    return count, digest.hexdigest()


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive members')
        manifest = json.loads(archive.read('handoff-manifest.json'))
        expected = {item['path']: item for item in manifest['files']}
        if len(expected) != len(manifest['files']) or set(names) != set(expected) | {'handoff-manifest.json'}:
            raise ValueError('Archive member coverage mismatch')
        for name, entry in expected.items():
            parsed = PurePosixPath(name)
            if parsed.is_absolute() or '..' in parsed.parts or '\\' in name or ':' in name:
                raise ValueError('Unsafe archive member')
            with archive.open(name) as stream:
                size, digest = read_digest(stream)
            if size != entry['bytes'] or digest != entry['sha256']:
                raise ValueError('Archive integrity mismatch: ' + name)
    return manifest


def mac_launcher():
    return '''#!/bin/sh
cd "$(dirname "$0")" || exit 1
printf '%s\\n' 'Starting the saved-evidence demo. No model inference will run.'
for candidate in /opt/homebrew/bin/python3 /usr/local/bin/python3 python3.14 python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 11))' >/dev/null 2>&1; then
        exec "$candidate" -m bioprocess_runtime.demo_ui --port 0 --open-browser
    fi
done
printf '%s\\n' 'Python 3.11 or newer is required. Install Python, then open this launcher again.'
printf '%s\\n' 'No packages, model files, or GPU are required for this demo.'
printf '%s' 'Press Return to close. '
read -r answer
exit 1
'''


def build_archive(root, output, records, include_runtime, *, demo_only=False):
    if include_runtime and demo_only:
        raise ValueError('Demo-only packaging cannot include the numerical runtime')
    if demo_only:
        allowed = demo_paths(root)
        if len(records) != len(allowed) or {row['path'] for row in records} != allowed:
            raise ValueError('Demo-only archive requires exactly the allowlisted assets')
    if output.exists() or output.suffix.lower() != '.zip':
        raise ValueError('Use a new .zip output file')
    output.parent.mkdir(parents=True, exist_ok=True)
    total = sum(record['bytes'] for record in records)
    if shutil.disk_usage(output.parent).free < total + max(1024**3, total // 10):
        raise ValueError('Insufficient free disk space for an uncompressed safe handoff')
    temporary = output.with_name(output.name + '.' + uuid.uuid4().hex + '.partial')
    manifest = {'schema_version': 1, 'kind': 'mac_demo_handoff' if demo_only else 'local_runtime_handoff' if include_runtime else 'source_demo_handoff',
                'git_head': git(root, 'rev-parse', 'HEAD').decode().strip(), 'working_tree_clean': not bool(git(root, 'status', '--porcelain')),
                'files': [], 'environment': {'python_requirement': '>=3.11', 'dependencies': 'Python standard library only', 'model_execution_supported': False, 'tested_on_macos': False} if demo_only else environment_manifest(), 'compression': 'stored',
                'limitations': ['No virtual environment or system CUDA driver is bundled.', 'Fresh numerical execution requires the declared matching runtime; saved UI does not run a model.', 'Completed proof checkpoints are included where required; the full interrupted checkpoint history is not bundled.', 'No upload or numerical re-execution was performed by packaging.']}
    if demo_only:
        manifest['limitations'] = ['Saved numerical evidence and live synthetic policy only; no model inference assets.',
                                   'Python 3.11 or newer must already be installed; no dependency installation is performed.',
                                   'The regression log records the original workstation run, not a fresh Mac test.',
                                   'The launcher is prepared for macOS; actual Mac launch and visual review require user confirmation.']
    try:
        with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for record in records:
                path = safe_file(root, record['path'])
                before = path.stat()
                digest = hashlib.sha256()
                written = 0
                with path.open('rb') as source, archive.open(record['path'], 'w', force_zip64=True) as destination:
                    while chunk := source.read(CHUNK):
                        destination.write(chunk)
                        digest.update(chunk)
                        written += len(chunk)
                after = path.stat()
                if before.st_size != written or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError('Source changed while packaging: ' + record['path'])
                manifest['files'].append({'path': record['path'], 'bytes': written, 'sha256': digest.hexdigest()})
            requirements = ''.join(name + '==' + version + '\n' for name, version in manifest['environment'].get('packages', {}).items() if name.lower().replace('_', '-') not in ('bioprocess-decision-runtime', 'pip', 'setuptools', 'wheel'))
            launcher = '@echo off\r\ncd /d "%~dp0"\r\nif exist ".venv\\Scripts\\python.exe" (\r\n  ".venv\\Scripts\\python.exe" -m bioprocess_runtime.demo_ui --port 0\r\n) else (\r\n  py -3.11 -m bioprocess_runtime.demo_ui --port 0\r\n)\r\npause\r\n'
            generated = {'Launch Demo.command': mac_launcher().encode('utf-8')} if demo_only else {'handoff-requirements.txt': requirements.encode('utf-8'), 'launch-demo.cmd': launcher.encode('utf-8')}
            for name, data in generated.items():
                member = zipfile.ZipInfo(name)
                member.create_system = 3
                member.external_attr = (0o100755 if name.endswith('.command') else 0o100644) << 16
                archive.writestr(member, data)
                manifest['files'].append({'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
            archive.writestr('handoff-manifest.json', json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        with temporary.open('r+b') as stream:
            stream.flush()
            os.fsync(stream.fileno())
        verify_archive(temporary)
        os.link(temporary, output)
        if os.name != 'nt':
            descriptor = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()
    with output.open('rb') as stream:
        size, digest = read_digest(stream)
    return {'archive': str(output), 'bytes': size, 'sha256': digest, 'files': len(manifest['files']), 'git_head': manifest['git_head'], 'verified': True}


def main():
    parser = argparse.ArgumentParser(description='Low-CPU local handoff builder; no inference or downloads')
    parser.add_argument('operation', choices=('inventory', 'build', 'verify', 'check-index'))
    parser.add_argument('--root', type=Path, default=ROOT)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--runtime', action='store_true')
    modes.add_argument('--demo-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.operation == 'verify':
        if args.output is None:
            parser.error('--output is required')
        manifest = verify_archive(args.output)
        print(json.dumps({'verified': True, 'files': len(manifest['files']), 'git_head': manifest['git_head']}))
        return
    records = inventory(root, args.runtime, demo_only=args.demo_only)
    if args.operation == 'inventory':
        print(json.dumps({'files': len(records), 'total_bytes': sum(record['bytes'] for record in records),
                          'crlf_source_files': [record['path'] for record in records if record.get('crlf_count')],
                          'binary_source_files': [record['path'] for record in records if record.get('nul_bytes')],
                          'largest': sorted(records, key=lambda record: record['bytes'], reverse=True)[:25]}, indent=2))
    elif args.operation == 'check-index':
        changed = []
        for record in records:
            actual = hashlib.sha256(safe_file(root, record['path']).read_bytes()).hexdigest()
            staged = hashlib.sha256(git(root, 'show', ':' + record['path'])).hexdigest()
            if actual != staged:
                changed.append(record['path'])
        if changed:
            raise ValueError('Index differs from working bytes: ' + ', '.join(changed))
        print(json.dumps({'index_byte_identity': True, 'files': len(records)}))
    else:
        if args.output is None:
            parser.error('--output is required')
        print(json.dumps(build_archive(root, args.output.resolve(), records, args.runtime, demo_only=args.demo_only), indent=2))


if __name__ == '__main__':
    main()
