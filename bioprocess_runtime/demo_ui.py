from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .demo_assets import HTML, CSS, JAVASCRIPT
from .policy import load_policy
from .runtime import evaluate
from .serialization import canonical_json
from .simulator import scenarios

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 4 * 1024 * 1024
NOTICE = "Research demonstration only. Saved numerical evidence is not fresh inference. Synthetic advisory decisions are not validated for clinical, GMP, manufacturing, or equipment-control use."
IR_SHA256 = "a32274520915ed7dcbf48634499c9e434270b59116ea6c19fdee8ca8637e4712"


@dataclass(frozen=True)
class EvidenceSpec:
    key: str
    title: str
    stem: str
    match_field: str
    plan_sha256: str
    summary_sha256: str
    report_sha256: str
    instruction_indices: tuple[int, ...]
    description: str
    replay_sha256: str | None = None


PREFIX_NODES = tuple(index for index in range(65) if index not in (3, 5))
EVIDENCE = (
    EvidenceSpec("full-target-holdouts", "Full-stack held-outs and replay", "full_target_holdout", "full_target_holdout_matches",
                 "d4dad7b728207cebbbbc83ce538079bb382de02638051a80fa21447a32d3b78a", "f708f85e23c4b3c38dcd92e1bc0a73337627758b83dfa11bcd6ce2aaa497785f", "83b4c183531e43781777cf6d544ed6ba59533ded84264b0127741fdb37f5c960", tuple(range(533)),
                 "Two predeclared token-to-selection cases. All decoder boundaries and vocabulary logits match native execution and replay; checkpoint continuation is disclosed.",
                 "1086fb2580b9d7b4fca09252cf7cb0d7a2fc8ff0a1bcb745f5d76f4e230c0840"),
    EvidenceSpec("two-layers", "Connected two-layer baseline", "two_layers", "two_layers_match",
                 "49d121756cca33eb1d139061093d844aa851791323cc7f5b34f19a0870b78fc4", "e26a8abc0c110ed80bf43395210983762a7a2053ce46c5b499616fc939d7cef3", "9b177ee28872ef11411307ce015843233562b2bdeeeabedc1bfc6caa9dacf5bc", PREFIX_NODES,
                 "Fresh token-to-hidden.2 computation in the recorded experiment; native coverage uses separate stopped forwards."),
    EvidenceSpec("holdouts", "Two-layer raw-token holdouts", "two_layers_holdout", "two_layer_holdout_matches",
                 "23424f5e22df5ec12b3a680c25a589983301593309cbedda41ffb2724300c7aa", "5efbd6d8ce0c3d1a4c5210540a6c5b49ecfb636be8e20d926bd4e1a8c626e3c1", "e61317b19a462af21c0cb0bf471b883e97231be922811579b28720fa5d87303d", PREFIX_NODES,
                 "Two predeclared 30-token cases: distinct tokens and a repeated motif. Numerical evidence, not language-task performance."),
    EvidenceSpec("layer-2-entry", "Layer 2: attention entry", "third_layer_entry", "entry_matches",
                 "b8b1d5bcfcdfce776f7dba5316abcee864ec9c68f023efd75b68e4dfd3e29a30", "9a72a8a8e491599091ba5bb9cb6f82dff4f45e2da0087b1ceecf55f9426c8f1f", "d56dfe2d2609eccd714a452cf52c07cc6b3b954328c5a4b0dc7c84ab449b6b2a", tuple(range(65, 74)),
                 "Reused hidden.2 boundary. Input normalization, Q/K/V projections and Q/K normalization; V heads are a coordinate reinterpretation."),
    EvidenceSpec("layer-2-scores", "Layer 2: rotary and scores", "third_layer_scores", "rotary_scores_match",
                 "5f8ed5aa75f3b71f883cfd136faf215154ed78bf54c6081d0f57cc33074b5509", "a72a975069c22ea21ef61a6d0ece7b2fdb2a477a41846237a0851bcda719cc67", "52323090f0596f91b0ee0ef3adcab64f70afb19d8ac9d7f7479f7797e7839e24", tuple(range(74, 80)),
                 "Reused Q/K/V and rotary/mask boundaries. Stops before target softmax; the non-dispatch control does not observe intermediate scaled scores."),
)


class DemoDataError(ValueError):
    pass


def _file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise DemoDataError("Data path leaves the configured repository")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise DemoDataError("Data file exceeds the demonstration size limit")
    return path


def _json(root: Path, relative: str) -> dict[str, Any]:
    data = _file(root, relative).read_bytes()
    if len(data) > MAX_FILE_BYTES:
        raise DemoDataError("Data file exceeds the demonstration size limit")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise DemoDataError("Expected an evidence object")
    return value


def _sealed(value: dict[str, Any], field: str, pin: str) -> bool:
    digest = hashlib.sha256(canonical_json({key: item for key, item in value.items() if key != field}).encode("utf-8")).hexdigest()
    return value.get(field) == pin == digest


def _counts(value: Any, prefix: str = "") -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        raise DemoDataError("Invalid mismatch-count structure")
    result = []
    for name, count in value.items():
        label = prefix + name
        if isinstance(count, dict):
            result.extend(_counts(count, label + " / "))
        elif type(count) is int and count >= 0:
            result.append({"name": label, "mismatches": count})
        else:
            raise DemoDataError("Mismatch counts must be nonnegative integers")
    return result


def evidence_detail(root: Path, key: str) -> dict[str, Any]:
    spec = next((item for item in EVIDENCE if item.key == key), None)
    if spec is None:
        raise KeyError(key)
    base = {"id": spec.key, "title": spec.title, "description": spec.description, "mode": "saved_evidence",
            "fresh_execution": False, "raw_evidence_revalidated": False, "recorded_match": False,
            "integrity_scope": "Pinned compact plan and summary hashes only; no raw tensor validation or numerical re-execution."}
    try:
        prefix = "results/gemma3_270m_" + spec.stem
        plan = _json(root, prefix + "_plan.json")
        summary = _json(root, prefix + "_summary.json")
        if not _sealed(plan, "plan_sha256", spec.plan_sha256) or not _sealed(summary, "summary_sha256", spec.summary_sha256):
            raise DemoDataError("Compact evidence does not match its pinned hashes")
        if summary.get("plan_sha256") != plan["plan_sha256"] or summary.get("source_report_sha256") != spec.report_sha256:
            raise DemoDataError("Plan, summary and report references disagree")
        if spec.replay_sha256 is not None and (summary.get("source_replay_sha256") != spec.replay_sha256 or summary.get("fresh_replay_matches") is not True):
            raise DemoDataError("Required saved replay evidence is missing or inconsistent")
        counts = _counts(summary["mismatch_counts"])
        aggregate = summary["aggregate_mismatch_count"]
        if type(aggregate) is not int or aggregate != sum(item["mismatches"] for item in counts):
            raise DemoDataError("Aggregate mismatch count is inconsistent")
        matched = summary.get(spec.match_field) is True and aggregate == 0
        coverage = summary.get("coverage_per_case", summary.get("coverage", plan.get("coverage", {})))
        instructions = []
        instruction_error = None
        try:
            program = _json(root, "results/gemma3_270m_execution_ir.json")
            if not _sealed(program, "program_sha256", IR_SHA256):
                raise DemoDataError("Instruction program does not match its pinned hash")
            selected = {f"i{index:04d}" for index in spec.instruction_indices}
            instructions = [{name: node[name] for name in ("id", "opcode", "layer", "inputs", "outputs")} for node in program["instructions"] if node["id"] in selected]
            if len(instructions) != len(selected):
                raise DemoDataError("Instruction coverage is incomplete")
        except (OSError, ValueError, KeyError, TypeError):
            instructions = []
            instruction_error = "Pinned instruction graph unavailable; evidence summary remains separate."
        traces = {"trace_root": plan["trace_root"]} if "trace_root" in plan else {"trace_" + case["case_id"]: case["trace_root"] for case in plan.get("cases", [])}
        return {**base, "availability": "available", "compact_integrity": "pinned_hashes_match", "recorded_match": matched,
                "scope": summary["scope"], "mismatches": aggregate, "comparison_counts": counts,
                "forward_count": summary.get("original_forward_count"), "coverage": coverage,
                "instructions": instructions, "instruction_notice": instruction_error,
                "hashes": {"plan": spec.plan_sha256, "summary": spec.summary_sha256, "report_reference": spec.report_sha256, **({"replay_reference": spec.replay_sha256} if spec.replay_sha256 else {}), **traces},
                "replay_recorded_match": summary.get("fresh_replay_matches") is True and spec.replay_sha256 is not None,
                "cases": [{"id": case.get("case_id"), "mismatches": case.get("aggregate_mismatch_count"), "recorded_match": case.get("case_matches") is True,
                           "selected_token_id": case.get("selected_token_id"), "prediction_restored_prefix_count": case.get("prediction_restored_prefix_count"),
                           "replay_restored_prefix_count": case.get("replay_restored_prefix_count")} for case in summary.get("cases", [])],
                "prefix_reused": summary.get("prefix_boundary_reused", False),
                "qualification": {name: summary.get(name, False) for name in ("qualified", "full_model_qualified", "hardware_semantics_established", "global_exactness_activation_allowed")}}
    except FileNotFoundError:
        return {**base, "availability": "missing", "compact_integrity": "not_checked", "message": "Required saved plan or summary is missing."}
    except (OSError, ValueError, KeyError, TypeError, RecursionError):
        return {**base, "availability": "invalid", "compact_integrity": "failed", "message": "Saved evidence is unreadable, malformed, or does not match the pinned experiment. No match is asserted."}


def verification_status(root: Path) -> dict[str, Any]:
    result = {"scope": "Saved full holdout regression log, predating this UI update; not current UI verification", "complete": False, "tests": None, "state": "completion_not_recorded"}
    try:
        text = _file(root, "gemma_independent_holdout_verification_v3.log").read_text(encoding="utf-8-sig")
        failed = re.search(r"^(FAILED|ERROR:)", text, re.MULTILINE) is not None
        tests = re.search(r"^Ran (\d+) tests? in ([\d.]+)s", text, re.MULTILINE)
        complete = not failed and tests is not None and re.search(r"^OK\s*$", text, re.MULTILINE) is not None and "VERIFICATION_EXIT_CODE=0" in text and "No broken requirements found." in text
        result.update(complete=complete, tests=int(tests[1]) if tests else None, state="passed" if complete else "failure_recorded" if failed else "completion_not_recorded")
    except (OSError, ValueError):
        result["state"] = "log_unavailable"
    return result


def catalog(root: Path) -> dict[str, Any]:
    entries = [evidence_detail(root, spec.key) for spec in EVIDENCE]
    ready = {item["id"]: item["recorded_match"] for item in entries}
    full = ready["full-target-holdouts"]
    layers = [{"index": index, "attention": "full" if index in (5, 11, 17) else "sliding",
               "coverage": "connected_recorded" if full or index < 2 and ready["two-layers"] else "partial_recorded" if index == 2 and (ready["layer-2-entry"] or ready["layer-2-scores"]) else "not_shown",
               "evidence_id": "full-target-holdouts" if full else "two-layers" if index < 2 and ready["two-layers"] else "layer-2-scores" if index == 2 and ready["layer-2-scores"] else "layer-2-entry" if index == 2 and ready["layer-2-entry"] else None} for index in range(18)]
    return {"notice": NOTICE, "mode": "saved_evidence", "layers": layers, "full_target_replay_recorded": full,
            "experiments": [{key: item[key] for key in ("id", "title", "description", "availability", "compact_integrity", "recorded_match")} for item in entries],
            "verification": verification_status(root)}


def scenario_catalog() -> dict[str, Any]:
    return {"mode": "synthetic_advisory", "items": [{"id": item.name, "description": item.description, "expected_status": item.expected_status,
            "evaluated_at": item.evaluated_at.isoformat(), "inputs": {name: observation.to_dict() for name, observation in item.observations.items()}} for item in scenarios().values()]}


def run_scenario(root: Path, name: str) -> dict[str, Any]:
    scenario = scenarios().get(name)
    if scenario is None:
        raise KeyError(name)
    policy = load_policy(_file(root, "policies/oxygen_advisory.bpr"))
    decision = evaluate(policy, scenario.observations, scenario.evaluated_at).to_dict()
    return {"mode": "live_synthetic_policy_evaluation", "fresh_execution": True, "gemma_inference": False,
            "equipment_actuation": False, "audit_file_written": False, "notice": NOTICE,
            "evaluation_clock": "Fixed synthetic scenario timestamp; evaluated by the policy engine on this request.",
            "expected_status": scenario.expected_status, "matches_expected": decision["status"] == scenario.expected_status, "decision": decision}


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, root: Path, port: int = 8765):
        self.root = root.resolve()
        if not self.root.is_dir():
            raise ValueError("Repository root must be an existing directory")
        super().__init__(("127.0.0.1", port), DemoHandler)


class DemoHandler(BaseHTTPRequestHandler):
    server: DemoServer
    server_version = "BioprocessDemo"
    sys_version = ""

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, format, *args):
        pass

    def respond(self, status: int, body: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'")
        self.end_headers()
        if self.command != "HEAD":
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_GET(self):
        port = self.server.server_port
        if self.headers.get("Host") not in {f"127.0.0.1:{port}", f"localhost:{port}"}:
            self.respond(403, b"Local host required", "text/plain; charset=utf-8")
            return
        origin = self.headers.get("Origin")
        if origin is not None and origin not in {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}:
            self.respond(403, b"Same-origin request required", "text/plain; charset=utf-8")
            return
        if len(self.path) > 1024:
            self.respond(414, b"Request path too long", "text/plain; charset=utf-8")
            return
        if not self.path.startswith("/") or self.path.startswith("//"):
            self.respond(404, b"Not found", "text/plain; charset=utf-8")
            return
        path = urlsplit(self.path)
        if path.query or path.fragment:
            self.respond(404, b"Not found", "text/plain; charset=utf-8")
            return
        static = {"/": (HTML, "text/html"), "/app.css": (CSS, "text/css"), "/app.js": (JAVASCRIPT, "application/javascript")}
        if path.path in static:
            text, mime = static[path.path]
            self.respond(200, text.encode("utf-8"), mime + "; charset=utf-8")
            return
        try:
            if path.path == "/api/catalog":
                result = catalog(self.server.root)
            elif path.path.startswith("/api/evidence/"):
                result = evidence_detail(self.server.root, path.path.removeprefix("/api/evidence/"))
            elif path.path == "/api/scenarios":
                result = scenario_catalog()
            elif path.path.startswith("/api/scenarios/"):
                result = run_scenario(self.server.root, path.path.removeprefix("/api/scenarios/"))
            elif path.path == "/api/status":
                result = verification_status(self.server.root)
            else:
                raise KeyError(path.path)
            self.respond(200, canonical_json(result).encode("utf-8"), "application/json; charset=utf-8")
        except KeyError:
            self.respond(404, b"Not found", "text/plain; charset=utf-8")
        except (OSError, ValueError, TypeError):
            self.respond(503, b"Demonstration data unavailable", "text/plain; charset=utf-8")

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        self.respond(405, b"Read-only demonstration server", "text/plain; charset=utf-8")

    do_PUT = do_POST
    do_DELETE = do_POST
    do_PATCH = do_POST


def main() -> None:
    parser = argparse.ArgumentParser(description="Local-only evidence viewer and synthetic advisory demonstration; no model inference or equipment actuation.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("Port must be between 0 and 65535")
    with DemoServer(args.root, args.port) as server:
        print(f"Demonstration UI: http://127.0.0.1:{server.server_port}", flush=True)
        print(NOTICE, flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
