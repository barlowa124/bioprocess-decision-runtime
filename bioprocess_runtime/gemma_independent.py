from __future__ import annotations

import copy
import hashlib
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from . import gemma_first_layer as first
from . import gemma_gemv as gemv
from .gemma_checkpoint import CheckpointStore, code_sha256 as checkpoint_code_sha256
from .gemma_rotary_slice import _sha, _seal, _check_hash
from .operational_semantics import append_chain_record, verify_trace_chain
from .serialization import canonical_json

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
GLOBAL_FREQUENCY = "model.rotary_emb.inv_freq"
LINEAR_ROLES = {"self_attn.q_proj": "serial", "self_attn.k_proj": "key_value", "self_attn.v_proj": "key_value", "self_attn.o_proj": "output", "mlp.gate_proj": "serial", "mlp.up_proj": "serial", "mlp.down_proj": "down"}
FALSE_FLAGS = ("qualified", "native_comparison_performed", "full_model_qualified", "hardware_semantics_established", "global_exactness_activation_allowed", "unrestricted_input_qualified", "candidate_refitting_allowed", "framework_arithmetic_fallback_used")
SUPPORTED = {"ARANGE", "EMBEDDING", "SCALE", "ROTARY_TABLE", "CAUSAL_MASK", "RMS_NORM", "LINEAR", "RESHAPE_TRANSPOSE_HEADS", "ROTARY_APPLY_PAIR", "REPEAT_KV", "MATMUL_QK", "SOFTMAX", "MATMUL_AV", "TRANSPOSE_RESHAPE_HEADS", "GELU_TANH", "ADD", "MUL", "SLICE_LAST_TOKEN", "ARGMAX"}


class UnsupportedArithmetic(ValueError):
    pass


def code_sha256() -> str:
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Independent interpreter source changed after import")
    return _sha({"interpreter": SOURCE_SHA256, "frozen_arithmetic": first._code_sha(), "checkpoint_storage": checkpoint_code_sha256(), "vocabulary": vocabulary_evidence()})


def vocabulary_evidence():
    return {"arithmetic_code_sha256": gemv.code_sha256(),
            "plan_sha256": "339cf67f63b776890975d52e997381b77cabc2b18eb17cd92a52d4736d706715",
            "report_sha256": "8c093ca3179fe68889a00fe08dc0c454f8553034cccd446475498032e08a13cb",
            "replay_sha256": "0d66c91284f26ae8d4c9ca51fda2c98755aa4c9dbb557e4fc96a369882f23970"}


def linear_evidence(node, profiles, vocabulary_candidate):
    mode = linear_mode(node)
    if mode == "vocabulary_candidate":
        if vocabulary_candidate is not True:
            raise UnsupportedArithmetic("Vocabulary projection requires explicit bounded GEMV candidate opt-in")
        return {"mode": "gemv", "profile": gemv.PROFILE, "gemv_evidence": vocabulary_evidence(),
                "vocabulary_shape_status": "bounded_component_evidence_not_full_model_qualification"}
    return {"mode": mode, "profile": profiles[mode]}


def check_linear_evidence(program, node, provider, profiles, vocabulary_candidate):
    expected = {"opcode": "LINEAR", "transfer_prequalified": False, **linear_evidence(node, profiles, vocabulary_candidate)}
    if canonical_json(provider) != canonical_json(expected):
        raise ValueError("Recorded LINEAR profile/evidence mismatch")
    if expected["mode"] == "gemv" and (first._shape(program, node["inputs"][0]) != [1, 1, 640] or program["parameter_commitments"][node["parameter_refs"][0]]["shape"] != [262144, 640]):
        raise ValueError("Recorded vocabulary GEMV shape differs from validated geometry")


def dependency_cone(program: dict[str, Any], target: str) -> list[dict[str, Any]]:
    if program.get("program_sha256") != first.PROGRAM_SHA256 or not first.verify_gemma_ir(program)["valid"]:
        raise ValueError("Independent execution requires the fixed checkpoint's intact typed IR")
    if target not in program["tensors"] or target == "input_ids":
        raise ValueError("Target must be a produced IR state")
    needed, nodes = {target}, []
    for node in reversed(program["instructions"]):
        if needed.intersection(node["outputs"]):
            nodes.append(node)
            needed.difference_update(node["outputs"])
            needed.update(node["inputs"])
    nodes.reverse()
    if needed != {"input_ids"}:
        raise ValueError("Target is disconnected from token inputs")
    available = {"input_ids"}
    for node in nodes:
        if any(name not in available for name in node["inputs"]) or available.intersection(node["outputs"]):
            raise ValueError("Missing producer or duplicate write in target cone")
        available.update(node["outputs"])
    return nodes


def linear_mode(node: dict[str, Any]) -> str:
    if node["opcode"] != "LINEAR" or len(node["parameter_refs"]) != 1:
        raise ValueError("Expected one-weight LINEAR instruction")
    name = node["parameter_refs"][0]
    if name == "lm_head.weight" and node["layer"] is None:
        return "vocabulary_candidate"
    matched = re.fullmatch(r"model\.layers\.(\d+)\.(self_attn\.[qkvo]_proj|mlp\.(?:gate|up|down)_proj)\.weight", name)
    if not matched or int(matched[1]) != node["layer"]:
        raise ValueError("LINEAR weight does not belong to its actual IR layer")
    return LINEAR_ROLES[matched[2]]


def causal_mask(sequence: int, window: int | None) -> np.ndarray:
    if type(sequence) is not int or sequence != 30 or (window is not None and (type(window) is not int or window != 512)):
        raise UnsupportedArithmetic("Only S=30 and full/sliding512 masks are declared")
    row = np.arange(sequence)[:, None]
    column = np.arange(sequence)[None, :]
    allowed = column <= row
    if window is not None:
        allowed &= column > row - window
    return np.where(allowed, 0, 0xFF7F).astype(np.uint16)[None, None, ...]


def argmax_bfloat16(values: np.ndarray) -> np.ndarray:
    if values.dtype != np.uint16 or values.ndim != 3 or values.shape[:2] != (1, 1) or values.shape[-1] == 0:
        raise ValueError("Expected one nonempty vocabulary row of BF16 bits")
    bits = values[0, 0].astype(np.uint32)
    if np.any((bits & 0x7F80) == 0x7F80):
        raise UnsupportedArithmetic("Nonfinite vocabulary logits are outside declared token-selection support")
    bits = np.where((bits & 0x7FFF) == 0, 0, bits)
    ordered = np.where(bits & 0x8000, (~bits) & 0xFFFF, bits | 0x8000)
    return np.asarray([int(np.argmax(ordered))], dtype=np.int64)


def validate_parameters(program, ids, snapshots, nodes):
    required = {name for node in nodes for name in node["parameter_refs"]}
    if set(snapshots) != required:
        raise ValueError("Snapshot bank must exactly cover reached parameter references")
    result, metadata = {}, {}
    for name, snapshot in snapshots.items():
        expected = program["parameter_commitments"][name]
        if name == first.EMBEDDING and snapshot.get("format") == "selected_token_rows_v1":
            if set(snapshot) != {"format", "token_indices", "bits", "descriptor", "full_table_commitment", "membership_attestation"} or snapshot["token_indices"] != ids[0].tolist() or snapshot["full_table_commitment"] != expected or snapshot["membership_attestation"] != "selected_by_index_during_fresh_hash_verified_model_binding":
                raise ValueError("Selected embedding-row binding mismatch")
            value = first._array(snapshot["bits"], [30, 640], "torch.bfloat16")
            for index in range(30):
                for previous in range(index):
                    if ids[0, index] == ids[0, previous] and not np.array_equal(value[index], value[previous]):
                        raise ValueError("Repeated token has inconsistent selected rows")
        else:
            if set(snapshot) != {"format", "bits", "descriptor", "commitment"} or snapshot["format"] != "full_parameter_bits_v1" or snapshot["commitment"] != expected:
                raise ValueError("Full parameter snapshot format or commitment mismatch")
            value = first._array(snapshot["bits"], expected["shape"], expected["dtype"])
            if first._descriptor(value)["sha256"] != expected["sha256"]:
                raise ValueError("Snapshot does not match the frozen checkpoint")
        if first._descriptor(value) != snapshot["descriptor"]:
            raise ValueError("Snapshot descriptor mismatch")
        result[name] = value
        metadata[name] = {key: item for key, item in snapshot.items() if key != "bits"}
    return result, metadata


def check_global_rotary(packet, program, frequency, runtime):
    _check_hash(packet, "provider_sha256")
    if packet.get("kind") != "fixed_position_empirical_global_rotary_v1" or packet.get("program_sha256") != program["program_sha256"] or packet.get("positions") != list(range(30)) or canonical_json(packet.get("runtime")) != canonical_json(runtime):
        raise ValueError("Global rotary provider scope/runtime mismatch")
    if packet.get("frequency") != first._descriptor(first._array(frequency, [128], "torch.float32")) or packet["frequency"]["sha256"] != program["parameter_commitments"][GLOBAL_FREQUENCY]["sha256"]:
        raise ValueError("Global rotary provider frequency mismatch")
    if packet.get("empirical_primitive_data") is not True or packet.get("hidden_states_used") is not False or packet.get("native_repeat_count") != 3 or packet.get("native_repeat_exact") is not True:
        raise ValueError("Global rotary provider lacks declared primitive calibration")
    arrays = {name: first._array(packet[name + "_bits"], [1, 30, 256], "torch.bfloat16") for name in ("cosine", "sine")}
    if any(first._descriptor(value) != packet["descriptors"][name] for name, value in arrays.items()):
        raise ValueError("Global rotary provider table changed")
    return arrays["cosine"], arrays["sine"]


def _worker_init(expected_code):
    if code_sha256() != expected_code:
        raise ValueError("Persistent numerical worker source differs from parent")


def _column_block(task):
    offset, left, weights, mode, profile = task
    rows = np.asarray([[first._dot(row.tolist(), column.tolist(), mode, profile) for column in weights] for row in left], dtype=np.uint16)
    return offset, rows


class ColumnProjector:
    def __init__(self, workers):
        if type(workers) is not int or not 1 <= workers <= 4:
            raise ValueError("Expected one to four workers")
        self.workers, self.pool = workers, None

    def __enter__(self):
        if self.workers > 1:
            self.pool = ProcessPoolExecutor(max_workers=self.workers, initializer=_worker_init, initargs=(code_sha256(),))
        return self

    def __exit__(self, *args):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)

    def project(self, left, weights, mode, profile):
        if left.dtype != np.uint16 or weights.dtype != np.uint16 or left.ndim != 2 or weights.ndim != 2 or left.shape[1] != weights.shape[1] or not left.shape[0] or not weights.shape[0]:
            raise ValueError("Invalid projection operands")
        tasks = ((offset, left, weights[offset:offset + 256], mode, profile) for offset in range(0, weights.shape[0], 256))
        results = map(_column_block, tasks) if self.pool is None else self.pool.map(_column_block, tasks)
        output = np.empty((left.shape[0], weights.shape[0]), dtype=np.uint16)
        next_offset = 0
        for offset, values in results:
            width = min(256, weights.shape[0] - offset)
            if offset != next_offset or values.dtype != np.uint16 or values.shape != (left.shape[0], width):
                raise ValueError("Projection block missing, malformed or reordered")
            output[:, offset:offset + width] = values
            next_offset += width
        if next_offset != weights.shape[0]:
            raise ValueError("Projection output coverage incomplete")
        return output


def _operation(node, inputs, parameters, snapshots, providers, profiles, runtime, global_rotary, program, projector, vocabulary_candidate):
    opcode, attrs = node["opcode"], node["attributes"]
    params = [parameters[name] for name in node["parameter_refs"]]
    auxiliary, evidence = {}, {"opcode": opcode, "transfer_prequalified": False}
    if opcode not in SUPPORTED:
        raise UnsupportedArithmetic("Unregistered independent opcode: " + opcode)
    if opcode == "ARANGE":
        values = [np.arange(30, dtype=np.int64)[None, :]]
    elif opcode == "EMBEDDING":
        value = params[0]
        selected = snapshots[node["parameter_refs"][0]]["format"] == "selected_token_rows_v1"
        values = [value[None, ...].copy() if selected else value[inputs[0]].copy()]
    elif opcode == "ROTARY_TABLE":
        if inputs[1].tolist() != [list(range(30))] or attrs["attention_scaling"] != 1.0:
            raise UnsupportedArithmetic("Rotary position/scaling domain unsupported")
        if node["parameter_refs"] == [first.FREQUENCY]:
            if first._descriptor(params[0])["sha256"] != providers.rotary_frequency_sha256:
                raise ValueError("Local rotary frequency does not match registered provider")
            values = [providers.rotary_cosine.copy(), providers.rotary_sine.copy()]
            evidence["provider"] = providers.rotary_evidence
        elif node["parameter_refs"] == [GLOBAL_FREQUENCY]:
            if global_rotary is None:
                raise UnsupportedArithmetic("Frozen global rotary primitive provider is required")
            values = [value.copy() for value in check_global_rotary(global_rotary, program, params[0], runtime)]
            evidence["provider_sha256"] = global_rotary["provider_sha256"]
        else:
            raise UnsupportedArithmetic("Unregistered rotary frequency")
    elif opcode == "CAUSAL_MASK":
        values = [causal_mask(inputs[0].shape[1], attrs["sliding_window"])]
    elif opcode == "RMS_NORM":
        value = inputs[0]
        rows = [first._rms_lookup_row(row, params[0].tolist(), attrs["epsilon"], providers.rsqrt, runtime) for row in value.reshape(-1, value.shape[-1]).tolist()]
        values = [np.asarray([row["output_bits"] for row in rows], dtype=np.uint16).reshape(value.shape)]
        auxiliary["rms"] = {key: [row[key] for row in rows] for key in first.STAGES}
        evidence["provider"] = providers.rsqrt.evidence
    elif opcode == "LINEAR":
        evidence.update(linear_evidence(node, profiles, vocabulary_candidate))
        left, mode = inputs[0], evidence["mode"]
        if mode == "gemv":
            if left.shape != (1, 1, 640) or params[0].shape != (262144, 640):
                raise UnsupportedArithmetic("Vocabulary GEMV requires M1/N262144/K640")
            try:
                projected = gemv.project_bits(left.reshape(1, 640), params[0])
            except gemv.UnsupportedGemvArithmetic as error:
                raise UnsupportedArithmetic(str(error)) from error
        else:
            projected = projector.project(left.reshape(-1, left.shape[-1]), params[0], mode, profiles[mode])
        values = [projected.reshape(*left.shape[:-1], params[0].shape[0])]
    elif opcode == "RESHAPE_TRANSPOSE_HEADS":
        values = [inputs[0].reshape(1, 30, attrs["heads"], attrs["head_dimension"]).transpose(0, 2, 1, 3).copy()]
    elif opcode == "ROTARY_APPLY_PAIR":
        values = [first.rotate_bfloat16_bits(value, inputs[2], inputs[3]) for value in inputs[:2]]
    elif opcode == "REPEAT_KV":
        values = [np.repeat(inputs[0], attrs["repetitions"], axis=1)]
    elif opcode in ("MATMUL_QK", "MATMUL_AV"):
        results = []
        for head in range(inputs[0].shape[1]):
            if opcode == "MATMUL_QK":
                left, right = inputs[0][0, head], inputs[1][0, head]
            else:
                left = np.pad(inputs[0][0, head], ((0, 0), (0, 2)))
                right = np.pad(inputs[1][0, head].T, ((0, 0), (0, 2)))
            results.append(projector.project(left, right, "serial", profiles["serial"]))
        values = [np.stack(results)[None, ...]]
        evidence.update(mode="operand_alignment_v1", reduction_width=256 if opcode == "MATMUL_QK" else 32, zero_padding=0 if opcode == "MATMUL_QK" else 2)
    elif opcode == "SOFTMAX":
        if attrs["axis"] != -1 or inputs[0].shape[-1] != 30:
            raise UnsupportedArithmetic("Softmax supports only the fixed thirty-position rows")
        rows = [first.lookup_softmax_row([int(value) << 16 for value in row], providers.exp, runtime) for row in inputs[0].reshape(-1, 30)]
        values = [np.asarray([row["output_bf16_bits"] for row in rows], dtype=np.uint16).reshape(inputs[0].shape)]
        auxiliary["softmax_f32"] = np.asarray([row["output_f32_bits"] for row in rows], dtype=np.uint32).reshape(inputs[0].shape).tolist()
        evidence["provider"] = providers.exp.evidence
    elif opcode == "TRANSPOSE_RESHAPE_HEADS":
        values = [first.concatenate_heads(inputs[0]).copy()]
    elif opcode == "GELU_TANH":
        values = [np.asarray([providers.gelu.predict_bits(int(value), runtime) for value in inputs[0].reshape(-1)], dtype=np.uint16).reshape(inputs[0].shape)]
        evidence["provider"] = providers.gelu.evidence
    elif opcode == "SCALE":
        if params:
            scale = int(params[0].item())
        elif attrs["scalar"] == 0.0625:
            scale = 0x3D80
        else:
            raise UnsupportedArithmetic("Unregistered scalar constant")
        values = [np.asarray([first.bfloat16_multiply_bits(int(value), scale) for value in inputs[0].reshape(-1)], dtype=np.uint16).reshape(inputs[0].shape)]
    elif opcode == "ADD" and node["inputs"][1] in ("mask.sliding", "mask.full"):
        values = [first.mask_score_bits(*inputs)]
    elif opcode in ("ADD", "MUL"):
        if inputs[0].shape != inputs[1].shape:
            raise ValueError("Elementwise operands must have matching declared shapes")
        operation = first.bfloat16_add_bits if opcode == "ADD" else first.bfloat16_multiply_bits
        values = [np.asarray([operation(int(left), int(right)) for left, right in zip(inputs[0].flat, inputs[1].flat)], dtype=np.uint16).reshape(inputs[0].shape)]
    elif opcode == "SLICE_LAST_TOKEN":
        values = [inputs[0][:, -1:, :].copy()]
    elif opcode == "ARGMAX":
        values = [argmax_bfloat16(inputs[0])]
        evidence["tie_rule"] = "first vocabulary index; signed zeros compare equal; nonfinite logits abstain"
    return values, auxiliary, evidence


def execute(program, token_ids, snapshots, providers, profiles, runtime, *, target="selected_token_id", global_rotary=None, workers=4, vocabulary_candidate=False, retain_states=False, progress=None, checkpoint_dir=None, resume=False, checkpoint_every=8, checkpoint_context=None):
    from contextlib import ExitStack
    code, program_guard = code_sha256(), _sha(program)
    nodes = dependency_cone(program, target)
    ids = first._tokens(program, token_ids).copy()
    if type(providers) is not first.Providers or type(vocabulary_candidate) is not bool or type(retain_states) is not bool or type(resume) is not bool or type(checkpoint_every) is not int or not 1 <= checkpoint_every <= 533 or (resume and checkpoint_dir is None):
        raise ValueError("Invalid independent execution options/provider/checkpoint configuration")
    first._profiles(profiles)
    parameters, parameter_metadata = validate_parameters(program, ids, snapshots, nodes)
    providers.validate(runtime, parameters.get(first.FREQUENCY))
    provider_guard = _sha(providers.commitments())
    global_guard, profile_guard = _sha(global_rotary), _sha(profiles)
    parameter_guard, context_guard, runtime_guard = _sha(parameter_metadata), _sha(checkpoint_context), _sha(runtime)
    binding = {"engine_code_sha256": code, "program_payload_sha256": program_guard, "input_ids": first._descriptor(ids), "target": target,
               "parameter_metadata_sha256": parameter_guard, "provider_sha256": provider_guard, "global_packet_sha256": global_guard,
               "profiles_sha256": profile_guard, "runtime": copy.deepcopy(runtime), "retain_states": retain_states,
               "vocabulary_candidate": vocabulary_candidate, "caller_context_sha256": context_guard}
    states, descriptors = {"input_ids": ids}, {"input_ids": first._descriptor(ids)}
    hashes = {"input_ids": "ROOT:" + descriptors["input_ids"]["sha256"]}
    uses = Counter(name for node in nodes for name in node["inputs"])
    retained, records, scalar_stages, softmax_f32 = {}, [], {}, {}
    failure, resumed = None, None

    def guard():
        _, metadata = validate_parameters(program, ids, snapshots, nodes)
        providers.validate(runtime, parameters.get(first.FREQUENCY))
        if code != code_sha256() or program_guard != _sha(program) or provider_guard != _sha(providers.commitments()) or global_guard != _sha(global_rotary) or profile_guard != _sha(profiles) or parameter_guard != _sha(metadata) or context_guard != _sha(checkpoint_context) or runtime_guard != _sha(runtime):
            raise ValueError("Frozen program/arithmetic/provider/parameter context changed during execution")

    def result(status):
        complete = status == "complete_uncompared"
        return {"schema_version": 3, "status": status, "target": target,
                "input_token_ids": ids.tolist(), "program_sha256": program["program_sha256"], "code_sha256": code,
                "runtime": copy.deepcopy(runtime), "profiles": copy.deepcopy(profiles), "providers": providers.commitments(), "global_rotary_sha256": global_rotary.get("provider_sha256") if global_rotary else None,
                "required_instruction_ids": [node["id"] for node in nodes], "completed_instruction_count": len(records), "state_retention": "all" if retain_states else "boundaries",
                "state_bits": {name: value.tolist() for name, value in retained.items()}, "state_descriptors": copy.deepcopy(descriptors),
                "scalar_stages": copy.deepcopy(scalar_stages), "softmax_f32_bits": copy.deepcopy(softmax_f32), "records": copy.deepcopy(records), "trace_root": records[-1]["record_hash"] if records else None,
                "abstention": copy.deepcopy(failure), "selected_token_id": int(retained["selected_token_id"][0]) if complete and "selected_token_id" in retained else None,
                "vocabulary_candidate_enabled": vocabulary_candidate, "empirical_primitive_specifications_reused": True,
                "checkpoint_resume": copy.deepcopy(resumed), "stored_boundary_predictions_used": resumed is not None,
                "snapshot_membership_scope": "Fresh parameter binding required; resumed prefixes restore source-bound state and are not numerically rerun in this invocation.",
                **dict.fromkeys(FALSE_FLAGS, False)}

    def save(store, status):
        guard()
        value = result(status)
        check_execution(program, value, allow_checkpoint=True)
        live = {name: array.tolist() for name, array in states.items() if uses[name] > 0 and name not in retained}
        committed = store.save(value, live)
        if progress is not None:
            progress({"event": "checkpoint_committed", "index": len(records), "total": len(nodes), "checkpoint_sha256": committed["checkpoint_sha256"]})

    with ExitStack() as stack:
        store = stack.enter_context(CheckpointStore(Path(checkpoint_dir), binding, resume=resume)) if checkpoint_dir is not None else None
        start = 0
        if resume:
            saved = store.latest
            prefix = saved["execution"]
            check_execution(program, prefix, allow_checkpoint=True)
            if prefix["status"] not in ("checkpoint", "complete_uncompared") or prefix["target"] != target or prefix["input_token_ids"] != ids.tolist() or prefix["completed_instruction_count"] != saved["completed_instruction_count"] or prefix["state_retention"] != ("all" if retain_states else "boundaries") or prefix["vocabulary_candidate_enabled"] is not vocabulary_candidate:
                raise ValueError("Checkpoint execution frontier differs from its binding")
            start = len(prefix["records"])
            records, descriptors = copy.deepcopy(prefix["records"]), copy.deepcopy(prefix["state_descriptors"])
            scalar_stages, softmax_f32 = copy.deepcopy(prefix["scalar_stages"]), copy.deepcopy(prefix["softmax_f32_bits"])
            retained = {name: first._array(bits, first._shape(program, name), program["tensors"][name]["dtype"]).copy() for name, bits in prefix["state_bits"].items()}
            hashes = {"input_ids": "ROOT:" + descriptors["input_ids"]["sha256"]}
            for record in records:
                hashes.update({name: record["record_hash"] for name in record["payload"]["outputs"]})
            uses = Counter(name for node in nodes[start:] for name in node["inputs"])
            live_names = {name for name in descriptors if uses[name] > 0}
            if set(saved["live_states"]) != live_names - set(retained):
                raise ValueError("Checkpoint live-state coverage differs from remaining dependencies")
            states = {}
            for name in live_names:
                array = retained[name] if name in retained else first._array(saved["live_states"][name], first._shape(program, name), program["tensors"][name]["dtype"])
                if first._descriptor(array) != descriptors[name]:
                    raise ValueError("Checkpoint live state differs from its verified ledger")
                states[name] = array.copy()
            if _sha(prefix["providers"]) != provider_guard or _sha(prefix["profiles"]) != profile_guard or prefix["runtime"] != runtime or prefix["global_rotary_sha256"] != (global_rotary.get("provider_sha256") if global_rotary else None):
                raise ValueError("Checkpoint execution provider/runtime mismatch")
            for record in records:
                if any(value != parameter_metadata[name] for name, value in record["payload"]["parameters"].items()):
                    raise ValueError("Checkpoint prefix parameters differ from fresh snapshots")
            resumed = {"checkpoint_sha256": saved["checkpoint_sha256"], "completed_instruction_count": start}
            if progress is not None:
                progress({"event": "checkpoint_restored", "index": start, "total": len(nodes), **resumed})
        elif store is not None:
            save(store, "checkpoint")
        projector = stack.enter_context(ColumnProjector(workers))
        for index in range(start, len(nodes)):
            node = nodes[index]
            try:
                values, auxiliary, provider = _operation(node, [states[name] for name in node["inputs"]], parameters, snapshots, providers, profiles, runtime, global_rotary, program, projector, vocabulary_candidate)
            except (UnsupportedArithmetic, ArithmeticError) as error:
                failure = {"instruction_id": node["id"], "opcode": node["opcode"], "layer": node["layer"], "type": type(error).__name__, "message": str(error)}
                break
            except ValueError as error:
                from .gemma_first_layer_holdout import DOMAIN_ERRORS
                if str(error) not in DOMAIN_ERRORS:
                    raise
                failure = {"instruction_id": node["id"], "opcode": node["opcode"], "layer": node["layer"], "type": type(error).__name__, "message": str(error)}
                break
            if len(values) != len(node["outputs"]):
                raise ValueError("Opcode output arity mismatch")
            for name, value in zip(node["outputs"], values):
                if name in descriptors:
                    raise ValueError("Duplicate independent state write")
                value = first._array(value, first._shape(program, name), program["tensors"][name]["dtype"]).copy()
                states[name], descriptors[name] = value, first._descriptor(value)
                if retain_states or name.startswith("hidden.") and "." not in name[7:] or name in (target, "logits.last", "selected_token_id"):
                    retained[name] = value
            output_name = node["outputs"][0]
            if "rms" in auxiliary:
                scalar_stages[output_name] = auxiliary["rms"]
            if "softmax_f32" in auxiliary:
                softmax_f32[output_name] = auxiliary["softmax_f32"]
            payload = {"instruction_id": node["id"], "instruction_sha256": node["instruction_sha256"], "layer": node["layer"], "opcode": node["opcode"],
                       "inputs": {name: {"descriptor": descriptors[name], "producer_record_hash": hashes[name]} for name in node["inputs"]},
                       "parameters": {name: parameter_metadata[name] for name in node["parameter_refs"]}, "provider": provider,
                       "outputs": {name: descriptors[name] for name in node["outputs"]}, "auxiliary_sha256": _sha(auxiliary)}
            record = append_chain_record(records, payload)
            hashes.update({name: record["record_hash"] for name in node["outputs"]})
            for name in node["inputs"]:
                uses[name] -= 1
                if uses[name] == 0:
                    states.pop(name, None)
            if store is not None and index + 1 < len(nodes) and ((index + 1) % checkpoint_every == 0 or node["opcode"] == "LINEAR" or any(name.startswith("hidden.") for name in node["outputs"])):
                save(store, "checkpoint")
            if progress is not None:
                progress({"event": "instruction_complete", "index": index + 1, "total": len(nodes), "instruction_id": node["id"], "layer": node["layer"], "opcode": node["opcode"], "record_hash": record["record_hash"]})
        guard()
        complete = failure is None and len(records) == len(nodes) and target in retained
        output = result("complete_uncompared" if complete else "abstained")
        check_execution(program, output)
        if store is not None and complete and store.last_index < len(records):
            save(store, "complete_uncompared")
        return output


def check_execution(program, result, *, allow_checkpoint=False):
    if type(result.get("schema_version")) is not int or result["schema_version"] != 3 or type(result.get("completed_instruction_count")) is not int:
        raise ValueError("Unsupported independent execution schema/count")
    resumed = result.get("checkpoint_resume")
    if result.get("stored_boundary_predictions_used") is not (resumed is not None):
        raise ValueError("Checkpoint continuation disclosure mismatch")
    if resumed is not None and (not isinstance(resumed, dict) or set(resumed) != {"checkpoint_sha256", "completed_instruction_count"} or not isinstance(resumed["checkpoint_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", resumed["checkpoint_sha256"]) or type(resumed["completed_instruction_count"]) is not int or not 0 <= resumed["completed_instruction_count"] <= result["completed_instruction_count"]):
        raise ValueError("Invalid checkpoint continuation provenance")
    nodes = dependency_cone(program, result["target"])
    ids = first._tokens(program, result["input_token_ids"])
    if result["program_sha256"] != program["program_sha256"] or result["code_sha256"] != code_sha256() or any(result.get(flag) is not False for flag in FALSE_FLAGS):
        raise ValueError("Independent result program/code/qualification mismatch")
    first._profiles(result["profiles"])
    records = result["records"]
    if result["required_instruction_ids"] != [node["id"] for node in nodes] or result["completed_instruction_count"] != len(records) or len(records) > len(nodes) or not verify_trace_chain(records)["valid"]:
        raise ValueError("Independent trace coverage mismatch")
    complete = result["status"] == "complete_uncompared"
    if complete:
        if len(records) != len(nodes) or result["abstention"] is not None:
            raise ValueError("Complete result contains incomplete execution")
    elif result["status"] == "checkpoint" and allow_checkpoint:
        if result["abstention"] is not None or len(records) == len(nodes):
            raise ValueError("Invalid intermediate checkpoint frontier")
    elif result["status"] != "abstained" or len(records) == len(nodes) or not isinstance(result["abstention"], dict) or result["abstention"].get("instruction_id") != nodes[len(records)]["id"]:
        raise ValueError("Invalid abstention frontier")
    descriptors = {"input_ids": first._descriptor(ids)}
    hashes = {"input_ids": "ROOT:" + descriptors["input_ids"]["sha256"]}
    rms, softmax = set(), set()
    for node, record in zip(nodes, records):
        payload = record["payload"]
        if payload["instruction_id"] != node["id"] or payload["instruction_sha256"] != node["instruction_sha256"] or payload["layer"] != node["layer"] or payload["opcode"] != node["opcode"] or set(payload["parameters"]) != set(node["parameter_refs"]) or set(payload["outputs"]) != set(node["outputs"]):
            raise ValueError("Recorded operation differs from actual IR")
        if payload["inputs"] != {name: {"descriptor": descriptors[name], "producer_record_hash": hashes[name]} for name in node["inputs"]}:
            raise ValueError("Recorded input producer linkage mismatch")
        for name, metadata in payload["parameters"].items():
            commitment = program["parameter_commitments"][name]
            if name == first.EMBEDDING and metadata.get("format") == "selected_token_rows_v1":
                if metadata.get("token_indices") != ids[0].tolist() or metadata.get("full_table_commitment") != commitment or metadata.get("membership_attestation") != "selected_by_index_during_fresh_hash_verified_model_binding":
                    raise ValueError("Recorded embedding membership binding mismatch")
            elif metadata.get("commitment") != commitment or metadata.get("format") != "full_parameter_bits_v1" or metadata["descriptor"]["sha256"] != commitment["sha256"]:
                raise ValueError("Recorded parameter commitment mismatch")
        provider = payload["provider"]
        if provider.get("opcode") != node["opcode"] or provider.get("transfer_prequalified") is not False:
            raise ValueError("Recorded arithmetic qualification mismatch")
        if node["opcode"] == "LINEAR":
            check_linear_evidence(program, node, provider, result["profiles"], result["vocabulary_candidate_enabled"])
        auxiliary = {}
        name = node["outputs"][0]
        shape = first._shape(program, name)
        if node["opcode"] == "RMS_NORM":
            rms.add(name)
            stages = result["scalar_stages"][name]
            if set(stages) != set(first.STAGES):
                raise ValueError("RMS auxiliary coverage mismatch")
            for bits in stages.values():
                first._array(bits, [int(np.prod(shape[:-1]))], "torch.float32")
            auxiliary["rms"] = stages
        if node["opcode"] == "SOFTMAX":
            softmax.add(name)
            bits = result["softmax_f32_bits"][name]
            first._array(bits, shape, "torch.float32")
            auxiliary["softmax_f32"] = bits
        if payload["auxiliary_sha256"] != _sha(auxiliary):
            raise ValueError("Auxiliary values differ from ledger")
        for name, descriptor in payload["outputs"].items():
            declaration = program["tensors"][name]
            if descriptor["shape"] != first._shape(program, name) or descriptor["dtype"] != declaration["dtype"] or not re.fullmatch(r"[0-9a-f]{64}", descriptor["sha256"]):
                raise ValueError("Output descriptor differs from typed IR")
            descriptors[name], hashes[name] = descriptor, record["record_hash"]
    if result["state_descriptors"] != descriptors or set(result["scalar_stages"]) != rms or set(result["softmax_f32_bits"]) != softmax:
        raise ValueError("Descriptor/auxiliary state coverage mismatch")
    expected_retention = set(descriptors) - {"input_ids"}
    if result["state_retention"] == "boundaries":
        expected_retention = {name for name in expected_retention if (name.startswith("hidden.") and "." not in name[7:]) or name in (result["target"], "logits.last", "selected_token_id")}
    elif result["state_retention"] != "all":
        raise ValueError("Unknown state-retention mode")
    if set(result["state_bits"]) != expected_retention:
        raise ValueError("Retained-state coverage mismatch")
    for name, bits in result["state_bits"].items():
        if first._descriptor(first._array(bits, first._shape(program, name), program["tensors"][name]["dtype"])) != descriptors[name]:
            raise ValueError("Retained state differs from execution ledger")
    if result["trace_root"] != (records[-1]["record_hash"] if records else None):
        raise ValueError("Final trace root mismatch")
    expected_token = None
    if complete and result["target"] == "selected_token_id":
        logits = first._array(result["state_bits"]["logits.last"], first._shape(program, "logits.last"), "torch.bfloat16")
        expected_token = int(argmax_bfloat16(logits)[0])
        if result["state_bits"]["selected_token_id"] != [expected_token]:
            raise ValueError("Selected-token state does not match full vocabulary logits")
    if result["selected_token_id"] != expected_token:
        raise ValueError("Published token differs from completed execution")
