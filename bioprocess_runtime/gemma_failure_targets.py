from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from decimal import Decimal
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digit_records(generation: dict[str, Any]) -> dict[int, dict[str, Any]]:
    ids = generation["generated_token_ids"]
    if not isinstance(ids, list) or any(type(value) is not int or value < 0 for value in ids):
        raise ValueError("Expected nonnegative integer output token IDs")
    records = {}
    for record in generation["digit_token_top5"]:
        step = record["step"]
        if type(step) is not int or not 0 <= step < len(ids) or step in records:
            raise ValueError("Invalid or duplicate digit-token step")
        if record["token_id"] != ids[step] or not isinstance(record["token"], str):
            raise ValueError("Digit-token record disagrees with generated token IDs")
        top = record["top5"]
        if len(top) != 5 or len({item["token"] for item in top}) != 5:
            raise ValueError("Expected five distinct recorded candidate strings")
        if any(type(item["logit"]) not in (int, float) or not math.isfinite(item["logit"]) for item in top):
            raise ValueError("Expected finite recorded candidate scores")
        selected = [item for item in top if item["token"] == record["token"]]
        if len(selected) != 1 or selected[0]["logit"] != max(item["logit"] for item in top):
            raise ValueError("Selected digit is not a maximum in the saved greedy scores")
        records[step] = record
    return records


def extract_targets(probe: dict[str, Any]) -> list[dict[str, Any]]:
    digit_ids = {}
    ledgers = []
    for case in probe["cases"]:
        ledger = _digit_records(case["generation"])
        ledgers.append(ledger)
        for record in ledger.values():
            digit = record["token"]
            if re.fullmatch(r"[0-9]", digit):
                if digit in digit_ids and digit_ids[digit] != record["token_id"]:
                    raise ValueError("Inconsistent saved single-digit token mapping")
                digit_ids[digit] = record["token_id"]
    if len(set(digit_ids.values())) != len(digit_ids):
        raise ValueError("Different digits share a saved token ID")

    targets = []
    for index, (case, ledger) in enumerate(zip(probe["cases"], ledgers)):
        if case["score_class"] != "misattribution":
            continue
        generation = case["generation"]
        emitted = generation["generated_text"]
        asked = Decimal(str(case["asked_value"]))
        if not asked.is_finite() or not Decimal(0) <= asked < Decimal(1):
            raise ValueError("Target requires a finite recorded value in [0, 1)")
        expected = format(asked, ".3f")
        if Decimal(expected) != asked or not re.fullmatch(r"0\.[0-9]{3}", expected):
            raise ValueError("Target requires a recorded value exact at three decimals")
        if not isinstance(emitted, str) or not re.fullmatch(r"0\.[0-9]{1,3}", emitted):
            raise ValueError("Target requires one bare decimal; general text alignment is unsupported")
        if Decimal(emitted) == asked:
            raise ValueError("Recorded misattribution contains the asked value")
        numeric_steps = {step: char for step, char in enumerate(emitted) if char.isdigit()}
        if set(ledger) != set(numeric_steps) or any(ledger[step]["token"] != digit for step, digit in numeric_steps.items()):
            raise ValueError("Numeric text does not align with the saved single-digit steps")
        ids = generation["generated_token_ids"]
        step = next((position for position, pair in enumerate(zip(emitted, expected)) if pair[0] != pair[1]), len(emitted))
        shortfall = step == len(emitted)
        if not expected[step].isdigit() or expected[step] not in digit_ids:
            raise ValueError("Expected digit has no recorded token-ID mapping")
        record = ledger.get(step)
        correct = next((item for item in record["top5"] if item["token"] == expected[step]), None) if record else None
        selected = next(item for item in record["top5"] if item["token"] == record["token"]) if record else None
        targets.append({
            "source_case_index": index,
            "set": case["set"],
            "level": case["level"],
            "target": case["target"],
            "recorded_score_class": case["score_class"],
            "inspection_kind": "numeric_prefix_shortfall" if shortfall else "digit_substitution",
            "expected_decimal": expected,
            "recorded_output": emitted,
            "prompt": case["prompt"],
            "prompt_utf8_sha256": _sha(case["prompt"].encode("utf-8")),
            "generated_step_zero_based": step,
            "recorded_prefix_text": emitted[:step],
            "generated_prefix_token_ids": ids[:step],
            "observed_token_id": ids[step] if step < len(ids) else None,
            "observed_digit": record["token"] if record else None,
            "expected_digit": expected[step],
            "expected_token_id_from_saved_digit_records": digit_ids[expected[step]],
            "recorded_top5": record["top5"] if record else None,
            "expected_digit_in_recorded_top5": correct is not None if record else None,
            "selected_minus_expected_recorded_score": float(Decimal(str(selected["logit"])) - Decimal(str(correct["logit"]))) if selected and correct else None,
            "prompt_token_count": None,
            "total_context_token_count_at_decision": None,
            "independent_engine_coverage_established": False,
            "mechanism_established": False,
        })
    return targets


def build_report(probe_path: Path, protocol_path: Path, heldout_path: Path, baseline_path: Path) -> dict[str, Any]:
    paths = {"probe": probe_path, "protocol": protocol_path, "heldout": heldout_path, "baseline": baseline_path}
    raw = {name: path.read_bytes() for name, path in paths.items()}
    probe = json.loads(raw["probe"])
    for name in ("protocol", "heldout"):
        if probe[name + "_sha256"] != _sha(raw[name]):
            raise ValueError(f"Saved {name} hash does not match the supplied source")
    sealed = json.loads(raw["heldout"])
    observed = [case for case in probe["cases"] if case["set"] == "heldout"]
    if len(sealed) != len(observed) or any(any(case[key] != entry[key] for key in ("set", "level", "target", "prompt")) for entry, case in zip(sealed, observed)):
        raise ValueError("Saved held-out prompts/order differ from the sealed inputs")
    baseline = json.loads(raw["baseline"])
    if baseline["input"]["token_count"] != 30:
        raise ValueError("This inspector describes only the recorded 30-token baseline")
    if probe["environment"]["decode"] != "greedy":
        raise ValueError("This inspector requires the recorded greedy decoding experiment")
    targets = extract_targets(probe)
    body = {
        "schema_version": 1,
        "kind": "offline_saved_failure_target_inspection",
        "scope": "Exploratory inspection of already observed outputs, not fresh holdout validation, numerical replay, causal explanation, or a proof certificate.",
        "sources": {name: {"filename": path.name, "sha256": _sha(raw[name])} for name, path in paths.items()},
        "inspector_sha256": _sha(Path(__file__).read_bytes()),
        "original_labels_preserved": True,
        "original_protocol_and_heldout_hashes_match": True,
        "heldout_prompt_records_match": True,
        "model_inference_performed": False,
        "recorded_environment": probe["environment"],
        "recorded_baseline": baseline["input"],
        "target_count": len(targets),
        "targets": targets,
        "limitations": [
            "Source hashes bind file bytes; they do not authenticate the original acquisition or prove preregistration.",
            "Expected decimals use saved asked_value fields; the external oncology ground truth is not independently revalidated.",
            "Alignment supports only bare decimal outputs with checked single-character digit records; punctuation and trailing token IDs are not independently decoded.",
            "Saved top-five values are rounded generation scores, not complete raw logits; missing candidates have unknown scores, not zero scores.",
            "A numeric prefix shortfall can match another recorded value without establishing retrieval of that value.",
            "Prompt token IDs/counts and explicit cache settings were not saved; retokenization requires hash-matched tokenizer/template assets.",
            "The historical independent engine covers a fixed 30-position full-context path, not arbitrary sequence lengths or cached decoding.",
            "These cases have already been inspected and must not be relabeled as fresh confirmatory holdouts.",
        ],
        "next_acquisition_requirements": [
            "Bind the original checkpoint, tokenizer, chat template, generation configuration and runtime before replay.",
            "Recover exact prompt IDs and effective cache settings; reproduce the saved output and decision scores before interpreting interventions.",
            "For a no-cache diagnostic, compare against the acquisition decoding path explicitly; do not assume equivalence.",
            "Capture the full vocabulary scores and declared residual/attention/MLP boundaries immediately before each target token.",
            "Use controlled interventions and newly declared confirmation cases before claiming a causal mechanism.",
            "Keep new sequence-length/backend evidence separate from the frozen 30-token numerical certificates.",
        ],
    }
    body["report_sha256"] = _sha(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, default=ROOT / "results/gemma3_270m_copy_drift_v1.json")
    parser.add_argument("--protocol", type=Path, default=ROOT / "results/gemma3_270m_copy_drift_protocol_v1.md")
    parser.add_argument("--heldout", type=Path, default=ROOT / "results/gemma3_270m_copy_drift_heldout_v1.json")
    parser.add_argument("--baseline", type=Path, default=ROOT / "results/gemma3_270m_operational_semantics_summary.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = build_report(args.probe, args.protocol, args.heldout, args.baseline)
        text = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(text)
        else:
            print(text, end="")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        parser.exit(1, f"Failure-target inspection rejected: {error}\n")


if __name__ == "__main__":
    main()
