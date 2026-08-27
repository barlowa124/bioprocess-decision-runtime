from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .domain import Decision
from .serialization import canonical_json


class AuditIntegrityError(ValueError):
    pass


def record_hash(previous_hash: str, payload: dict[str, Any]) -> str:
    material = f"{previous_hash}\n{canonical_json(payload)}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def read_records(path: str | Path) -> list[dict[str, Any]]:
    audit_path = Path(path)
    if not audit_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for number, line in enumerate(audit_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise AuditIntegrityError(f"Invalid JSON at audit line {number}") from exc
    return records


def append_decision(path: str | Path, decision: Decision) -> dict[str, Any]:
    audit_path = Path(path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    records = read_records(audit_path)
    verify_records(records)
    previous_hash = records[-1]["record_hash"] if records else "GENESIS"
    payload = decision.to_dict()
    record = {
        "previous_hash": previous_hash,
        "payload": payload,
        "record_hash": record_hash(previous_hash, payload),
    }
    with audit_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(record) + "\n")
    return record


def verify_records(records: Iterable[dict[str, Any]]) -> int:
    expected_previous = "GENESIS"
    count = 0
    for count, record in enumerate(records, start=1):
        if record.get("previous_hash") != expected_previous:
            raise AuditIntegrityError(f"Broken previous-hash link at record {count}")
        payload = record.get("payload")
        if not isinstance(payload, dict):
            raise AuditIntegrityError(f"Missing payload at record {count}")
        expected_hash = record_hash(expected_previous, payload)
        if record.get("record_hash") != expected_hash:
            raise AuditIntegrityError(f"Hash mismatch at record {count}")
        expected_previous = expected_hash
    return count


def verify_file(path: str | Path) -> int:
    return verify_records(read_records(path))


def find_decision(path: str | Path, decision_id: str) -> dict[str, Any]:
    for record in read_records(path):
        payload = record.get("payload", {})
        if payload.get("decision_id") == decision_id:
            return payload
    raise KeyError(f"Decision {decision_id!r} was not found")
