from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from .serialization import canonical_json

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
MAX_CHECKPOINT_BYTES = 256 * 1024 * 1024
NAME = re.compile(r"checkpoint-(\d{6})-([0-9a-f]{64})\.json")


def code_sha256() -> str:
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Checkpoint implementation changed after import")
    return SOURCE_SHA256


def seal(value: dict[str, Any], field: str) -> dict[str, Any]:
    return {**value, field: hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()}


def checked(value: dict[str, Any], field: str) -> None:
    if not isinstance(value, dict) or value.get(field) != hashlib.sha256(canonical_json({key: item for key, item in value.items() if key != field}).encode("utf-8")).hexdigest():
        raise ValueError("Checkpoint hash mismatch")


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or path.stat().st_size > MAX_CHECKPOINT_BYTES:
        raise ValueError("Unsafe or oversized checkpoint file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Checkpoint must be a data object")
    return value


def _publish(path: Path, value: dict[str, Any]) -> None:
    data = (canonical_json(value) + "\n").encode("utf-8")
    if len(data) > MAX_CHECKPOINT_BYTES:
        raise ValueError("Checkpoint exceeds declared size limit")
    temporary = path.with_name(".checkpoint-" + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


class CheckpointStore:
    def __init__(self, directory: Path, binding: dict[str, Any], *, resume: bool = False):
        self.directory = directory.resolve()
        self.binding = json.loads(canonical_json(binding))
        self.resume = resume
        self.lock = None
        self.manifest = None
        self.latest = None
        self.last_index = -1

    def __enter__(self):
        if self.resume:
            if not self.directory.is_dir():
                raise ValueError("Resume checkpoint directory is missing")
        else:
            self.directory.mkdir(parents=True, exist_ok=False)
        lock_path = self.directory / ".writer.lock"
        if lock_path.is_symlink():
            raise ValueError("Checkpoint lock cannot be a symbolic link")
        try:
            self.lock = lock_path.open("a+b")
            self.lock.seek(0, os.SEEK_END)
            if self.lock.tell() == 0:
                self.lock.write(b"\0")
                self.lock.flush()
            self.lock.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = self.directory / "binding.json"
            if self.resume:
                self.manifest = _load(path)
                checked(self.manifest, "manifest_sha256")
                if self.manifest.get("kind") != "independent_checkpoint_binding_v1" or self.manifest.get("binding") != self.binding:
                    raise ValueError("Checkpoint belongs to a different source/runtime/input/parameter binding")
                self.latest = self._latest()
                self.last_index = self.latest["completed_instruction_count"]
            else:
                self.manifest = seal({"kind": "independent_checkpoint_binding_v1", "run_id": uuid.uuid4().hex, "binding": self.binding}, "manifest_sha256")
                _publish(path, self.manifest)
            return self
        except BaseException:
            if self.lock is not None:
                self.lock.close()
                self.lock = None
            raise

    def __exit__(self, *args):
        if self.lock is not None:
            self.lock.close()
            self.lock = None

    def _latest(self):
        entries = []
        for path in self.directory.iterdir():
            if path.name.startswith("checkpoint-"):
                match = NAME.fullmatch(path.name)
                if not match:
                    raise ValueError("Malformed committed checkpoint name")
                entries.append((int(match[1]), match[2], path))
        if not entries:
            raise ValueError("No committed checkpoint available to resume")
        index = max(item[0] for item in entries)
        candidates = [item for item in entries if item[0] == index]
        if len({item[0] for item in entries}) != len(entries) or len(candidates) != 1:
            raise ValueError("Ambiguous checkpoint branch; refusing to choose")
        _, digest, path = candidates[0]
        value = _load(path)
        checked(value, "checkpoint_sha256")
        if value.get("kind") != "independent_checkpoint_v1" or value["checkpoint_sha256"] != digest or type(value.get("completed_instruction_count")) is not int or value["completed_instruction_count"] != index or value.get("manifest_sha256") != self.manifest["manifest_sha256"]:
            raise ValueError("Checkpoint file/binding/index mismatch")
        ordered = sorted(entries, key=lambda item: item[0])
        parent = ordered[-2][1] if len(ordered) > 1 else None
        if value.get("parent_checkpoint_sha256") != parent:
            raise ValueError("Checkpoint predecessor link is missing or inconsistent")
        return value

    def save(self, result: dict[str, Any], live_states: dict[str, Any]) -> dict[str, Any]:
        if self.lock is None:
            raise ValueError("Checkpoint writer lease is not held")
        index = result["completed_instruction_count"]
        if type(index) is not int or index <= self.last_index:
            raise ValueError("Checkpoint generations must advance monotonically")
        value = seal({"kind": "independent_checkpoint_v1", "manifest_sha256": self.manifest["manifest_sha256"],
                      "completed_instruction_count": index, "parent_checkpoint_sha256": self.latest["checkpoint_sha256"] if self.latest else None,
                      "execution": result, "live_states": live_states,
                      "scope": "Data-only same-run continuation; hashes bind source and state, not fresh arithmetic recomputation or external authenticity."}, "checkpoint_sha256")
        path = self.directory / f"checkpoint-{index:06d}-{value['checkpoint_sha256']}.json"
        _publish(path, value)
        self.latest, self.last_index = value, index
        return value
