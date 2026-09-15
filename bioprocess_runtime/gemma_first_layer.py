from __future__ import annotations

import copy
import hashlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import gemma_post_feedforward as ff
from .gemma_attention_entry import _bits
from .gemma_attention_output import concatenate_heads
from .gemma_attention_scores import causal_mask_bits, mask_score_bits
from .gemma_exp_lookup import CheckedExpLookup
from .gemma_float_semantics import bfloat16_add_bits, bfloat16_multiply_bits
from .gemma_gelu_lookup import CheckedGeluLookup
from .gemma_ir import verify_gemma_ir
from .gemma_ir_interpreter import bind_model_tensors
from .gemma_mlp_down import down_dot, SUPPORTED_CANDIDATE_JSON
from .gemma_output_survivor import survivor_dot, GRID
from .gemma_rotary_slice import _sha, _seal, _check_hash, check_rotary_table, rotate_bfloat16_bits
from .gemma_rsqrt_lookup import CheckedRsqrtLookup, _rms_lookup_row, _runtime
from .gemma_softmax_lookup import lookup_softmax_row
from .gemma_wmma_candidate import operand_aligned_product_bits, split_k_candidate_bits, DENSE_SPLIT_PROFILE
from .operational_semantics import append_chain_record, verify_trace_chain
from .serialization import canonical_json

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_DEPENDENCY_SHA256 = {name: hashlib.sha256(Path(__file__).with_name(name + ".py").read_bytes()).hexdigest() for name in (
    "gemma_float_semantics", "gemma_reduction_semantics", "gemma_wmma_candidate", "gemma_softmax_lookup", "gemma_softmax_slice", "gemma_exp_lookup",
    "gemma_gelu_lookup", "gemma_rsqrt_lookup", "gemma_rotary_slice", "gemma_attention_entry", "gemma_attention_scores", "gemma_attention_output",
    "gemma_output_survivor", "gemma_mlp_down", "gemma_ir", "gemma_ir_interpreter", "operational_semantics", "reference_gemma", "serialization")}
PROGRAM_SHA256 = "a32274520915ed7dcbf48634499c9e434270b59116ea6c19fdee8ca8637e4712"
POST_FEEDFORWARD_PLAN_SHA256 = "b18692848f4e85c962ca2f688f8e580ff569846024cc150688d3fc2cb53517ac"
EMBEDDING = "model.embed_tokens.weight"
FREQUENCY = "model.rotary_emb_local.inv_freq"
STAGES = ("mean_bits", "denominator_bits", "rsqrt_bits")
EXCLUDED = [{"id": "i0003", "outputs": ["rotary.global.cosine", "rotary.global.sine"], "reason": "Global RoPE is not in the hidden.1 local-attention dependency cone."},
            {"id": "i0005", "outputs": ["mask.full"], "reason": "Full-attention mask is not in the hidden.1 sliding-attention dependency cone."}]
SCOPE = "Fixed Gemma3 270m checkpoint and previously observed 30-token source case, independently connected input_ids-to-hidden.1 execution of 34 dependency-cone IR nodes and 36 output states. Registered empirical rsqrt/exp/GELU and checked fixed-position local rotary constants are reused primitive specifications, not cached activations or reconstructed native internals. Only six RMS scalar stages (810 positions) and softmax FP32 outputs (3600 positions) are compared as FP32 auxiliaries. Not all 36 prefix instructions: global RoPE/full mask may execute natively but are outside the validated cone. No fresh-prompt holdout, later layer, final model norm, logits, full hardware or unrestricted qualification."
FALSE_FLAGS = ("qualified", "completeFirstLayerQualified", "full_first_layer_qualified", "hardware_semantics_established", "global_exactness_activation_allowed", "qualification_promotion_allowed", "candidate_refitting_allowed", "native_arithmetic_reconstructed", "all_fp32_vector_internals_observed", "native_fused_registers_observed", "fresh_prompt_holdout", "later_layers_executed", "final_model_norm_executed", "logits_executed", "prefix_boundary_reused", "stored_intermediate_predictions_used")
_PROVIDER_METHODS = {cls: cls.predict_bits for cls in (CheckedRsqrtLookup, CheckedExpLookup, CheckedGeluLookup)}
_OUTPUT_PROFILE = next(copy.deepcopy(item) for item in GRID if item["id"] == "k128:bfloat16_rne:sequential_float32_rne")
_WORKER: tuple[Any, ...] = ()


def _code_sha() -> str:
    from . import gemma_first_layer_capture as capture
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("First-layer source changed after import")
    if any(hashlib.sha256(Path(__file__).with_name(name + ".py").read_bytes()).hexdigest() != digest for name, digest in _DEPENDENCY_SHA256.items()):
        raise ValueError("First-layer arithmetic dependency source changed after import")
    return _sha({"module": SOURCE_SHA256, "dependencies": _DEPENDENCY_SHA256, "previous_arithmetic": ff._code_sha(), "capture": capture._code_sha()})


def first_layer_instructions(program: dict[str, Any]) -> list[dict[str, Any]]:
    if not verify_gemma_ir(program)["valid"] or program.get("program_sha256") != PROGRAM_SHA256:
        raise ValueError("First-layer execution requires the exact fixed-checkpoint typed IR")
    needed, selected = {"hidden.1"}, []
    writers: dict[str, str] = {}
    for node in program["instructions"]:
        _check_hash(node, "instruction_sha256")
        for output in node["outputs"]:
            if output in writers or output == "input_ids":
                raise ValueError("Duplicate IR state write")
            writers[output] = node["id"]
    for node in reversed(program["instructions"]):
        if needed.intersection(node["outputs"]):
            selected.append(node)
            needed.difference_update(node["outputs"])
            needed.update(node["inputs"])
    selected.reverse()
    if needed != {"input_ids"} or program["external_inputs"] != ["input_ids"] or [node["id"] for node in selected] != [f"i{index:04d}" for index in range(36) if index not in (3, 5)]:
        raise ValueError("First-layer exact dependency closure/order/root mismatch")
    available = {"input_ids"}
    for node in selected:
        if any(name not in available for name in node["inputs"]) or any(name in available for name in node["outputs"]):
            raise ValueError("Missing, reordered or duplicate dependency")
        available.update(node["outputs"])
        for name in node["outputs"]:
            declaration = program["tensors"][name]
            if declaration["producer"] != node["id"] or declaration["dtype"] != ("torch.int64" if name == "position_ids" else "torch.bfloat16"):
                raise ValueError("First-layer tensor declaration mismatch")
    if len(available) != 37:
        raise ValueError("Expected all 36 freshly written states")
    return selected


def _shape(program: dict[str, Any], name: str) -> list[int]:
    return [{"B": 1, "S": 30}.get(dim, dim) for dim in program["tensors"][name]["shape"]]


def _array(value: Any, shape: list[int], dtype: str) -> np.ndarray:
    types = {"torch.bfloat16": (np.uint16, 0, 0xFFFF), "torch.float32": (np.uint32, 0, 0xFFFFFFFF), "torch.int64": (np.int64, 0, 0x7FFFFFFFFFFFFFFF)}
    target, lower, upper = types[dtype]
    if isinstance(value, np.ndarray):
        if value.dtype != target or list(value.shape) != shape:
            raise ValueError("Wrong array shape or bit dtype")
        return value
    raw = np.asarray(value, dtype=object)
    if list(raw.shape) != shape or any(type(item) is not int or not lower <= item <= upper for item in raw.reshape(-1)):
        raise ValueError("Malformed integer-bit tensor")
    return np.asarray(value, dtype=target)


def _descriptor(value: np.ndarray, dtype: str | None = None) -> dict[str, Any]:
    expected = {np.dtype(np.uint16): "torch.bfloat16", np.dtype(np.uint32): "torch.float32", np.dtype(np.int64): "torch.int64"}
    actual = expected.get(value.dtype)
    if actual is None or (dtype is not None and actual != dtype):
        raise ValueError("Unsupported descriptor dtype")
    metadata = f"{tuple(value.shape)}|{actual}|torch.strided".encode("utf-8")
    digest = hashlib.sha256(metadata + np.ascontiguousarray(value).tobytes()).hexdigest()
    return {"kind": "tensor", "shape": list(value.shape), "dtype": actual, "device": "cpu", "layout": "torch.strided", "numel": int(value.size), "sha256": digest}


def _tokens(program: dict[str, Any], token_ids: Any) -> np.ndarray:
    ids = _array(token_ids, [1, 30], "torch.int64")
    if np.any(ids < 0) or np.any(ids >= program["configuration"]["vocabulary_size"]):
        raise ValueError("First-layer root requires thirty valid token IDs")
    return ids.copy()


def _profiles(profiles: dict[str, Any]) -> None:
    expected = {"serial": "operand_alignment_v1", "key_value": list(DENSE_SPLIT_PROFILE), "output": _OUTPUT_PROFILE}
    if set(profiles) != {*expected, "down"} or any(canonical_json(profiles[name]) != canonical_json(value) for name, value in expected.items()) or canonical_json(profiles["down"]) != SUPPORTED_CANDIDATE_JSON:
        raise ValueError("Unsupported first-layer frozen arithmetic profiles")


@dataclass(frozen=True)
class Providers:
    rsqrt: CheckedRsqrtLookup
    exp: CheckedExpLookup
    gelu: CheckedGeluLookup
    rotary_cosine: np.ndarray
    rotary_sine: np.ndarray
    rotary_frequency_sha256: str
    rotary_runtime: dict[str, Any]
    rotary_evidence: dict[str, Any]

    def validate(self, runtime: dict[str, Any], frequency: np.ndarray | None = None) -> None:
        for provider, cls in ((self.rsqrt, CheckedRsqrtLookup), (self.exp, CheckedExpLookup), (self.gelu, CheckedGeluLookup)):
            if type(provider) is not cls or getattr(provider.predict_bits, "__func__", None) is not _PROVIDER_METHODS[cls] or cls.predict_bits is not _PROVIDER_METHODS[cls]:
                raise ValueError("First-layer requires registered checked provider objects, not substitutes")
            if canonical_json(provider._runtime) != canonical_json(runtime) or not provider.evidence:
                raise ValueError("Primitive provider runtime/evidence mismatch")
            table = provider._table
            dtype = np.uint16 if cls is CheckedGeluLookup else np.uint32
            if not isinstance(table, np.ndarray) or table.dtype != dtype or not table.flags.c_contiguous or hashlib.sha256(memoryview(table)).hexdigest() != provider.evidence["table_sha256"]:
                raise ValueError("Checked primitive provider table changed after registration")
        for name, value in (("cosine", self.rotary_cosine), ("sine", self.rotary_sine)):
            _array(value, [1, 30, 256], "torch.bfloat16")
            if _descriptor(value) != self.rotary_evidence["table_descriptors"][name]:
                raise ValueError("Rotary constant provider changed")
        if canonical_json(runtime) != canonical_json(self.rotary_runtime) or self.rotary_evidence["positions"] != list(range(30)):
            raise ValueError("Rotary provider runtime/position mismatch")
        if frequency is not None and (_array(frequency, [128], "torch.float32") is not frequency or _descriptor(frequency)["sha256"] != self.rotary_frequency_sha256):
            raise ValueError("Rotary provider is not bound to fresh inverse-frequency bits")

    def commitments(self) -> dict[str, Any]:
        return {"rsqrt": self.rsqrt.evidence, "exp": self.exp.evidence, "gelu": self.gelu.evidence,
                "rotary": copy.deepcopy(self.rotary_evidence), "rotary_frequency_sha256": self.rotary_frequency_sha256,
                "empirical_primitive_data_reused": True, "native_transcendental_internals_reconstructed": False}


def _source_tree_commitments(root: Any) -> Any:
    memo: dict[int, Any] = {}

    def visit(value: Any) -> Any:
        if id(value) in memo:
            return memo[id(value)]
        if type(value) in _PROVIDER_METHODS:
            result = {"checked_provider": type(value).__name__, "evidence": value.evidence}
        elif is_dataclass(value):
            result = {field.name: visit(getattr(value, field.name)) for field in fields(value)}
        elif isinstance(value, dict):
            result = {"payload_sha256": _sha(value)}
        elif isinstance(value, (tuple, list)):
            result = [visit(item) for item in value]
        elif value is None or type(value) in (str, int, bool, float):
            result = value
        else:
            raise ValueError("Unknown source lineage container")
        memo[id(value)] = result
        return result

    return visit(root)


@dataclass(frozen=True)
class FirstLayerSources:
    post_feedforward: ff.PostFeedforwardSources
    post_feedforward_plan: dict[str, Any]
    post_feedforward_bundle: dict[str, Any]
    post_feedforward_report: dict[str, Any]

    @property
    def program(self) -> dict[str, Any]:
        return self.post_feedforward.program

    @property
    def runtime(self) -> dict[str, Any]:
        return self.post_feedforward.runtime

    def commitments(self) -> dict[str, Any]:
        return {"post_feedforward_sources": self.post_feedforward.commitments(), "full_source_payloads": _source_tree_commitments(self.post_feedforward),
                "post_feedforward_plan_sha256": _sha(self.post_feedforward_plan),
                "post_feedforward_bundle_sha256": _sha(self.post_feedforward_bundle),
                "post_feedforward_report_sha256": _sha(self.post_feedforward_report)}

    def validate(self) -> tuple[list[list[int]], Providers, dict[str, Any]]:
        first_layer_instructions(self.program)
        if self.post_feedforward_plan.get("plan_sha256") != POST_FEEDFORWARD_PLAN_SHA256:
            raise ValueError("First-layer requires the declared latest post-feedforward source plan")
        checked = ff.verify_post(self.post_feedforward, self.post_feedforward_plan, self.post_feedforward_bundle, self.post_feedforward_report)
        if checked.get("valid") is not True or checked.get("post_feedforward_matches") is not True:
            raise ValueError("First-layer requires passing post-feedforward full previous lineage")
        product = self.post_feedforward.dense_sources.down.product
        post = product.entry.post
        output = post.survivor_context[0]
        scores = output.softmax.scores
        entry = scores.rotary_bundle["entry_plan"]
        ids = _tokens(self.program, entry["input_token_ids"])
        if _descriptor(ids) != entry["input_ids"]:
            raise ValueError("Source token ID descriptor mismatch")
        check_rotary_table(self.program, scores.table_plan, scores.table_manifest, scores.table_bundle)
        if any(canonical_json(value) != canonical_json(self.runtime) for value in (entry["runtime"], scores.table_plan["runtime"], scores.table_manifest["runtime"], self.post_feedforward_plan["runtime"])):
            raise ValueError("Source primitive runtimes disagree")
        if any(canonical_json(value) != canonical_json(entry["model_binding"]) for value in (scores.table_plan["model_binding"], self.post_feedforward_plan["model_binding"])):
            raise ValueError("Source model bindings disagree")
        evidence = {"plan_sha256": scores.table_plan["plan_sha256"], "manifest_sha256": scores.table_manifest["manifest_sha256"],
                    "table_descriptors": scores.table_manifest["table_descriptors"], "positions": list(range(30))}
        constants = scores.table_bundle["repetitions"][0]
        providers = Providers(post.lookup, output.lookup, product.gelu,
                              _array(constants["cosine"], [1, 30, 256], "torch.bfloat16").copy(),
                              _array(constants["sine"], [1, 30, 256], "torch.bfloat16").copy(),
                              scores.table_plan["frequency"]["sha256"], copy.deepcopy(self.runtime), copy.deepcopy(evidence))
        profiles = {"serial": "operand_alignment_v1", "key_value": list(DENSE_SPLIT_PROFILE),
                    "output": copy.deepcopy(post.survivor_plan["candidate"]), "down": copy.deepcopy(self.post_feedforward.dense_sources.model_plan["candidate"])}
        providers.validate(self.runtime)
        _profiles(profiles)
        return ids.tolist(), providers, profiles


def snapshot_parameters(program: dict[str, Any], token_ids: list[list[int]], parameters: dict[str, Any]) -> dict[str, Any]:
    import torch
    ids = _tokens(program, token_ids)
    reached = sorted({name for node in first_layer_instructions(program) for name in node["parameter_refs"]})
    snapshots = {}
    for name in reached:
        tensor = parameters[name]
        commitment = program["parameter_commitments"][name]
        if list(tensor.shape) != commitment["shape"] or str(tensor.dtype) != commitment["dtype"]:
            raise ValueError("Fresh parameter shape/dtype differs from the declared checkpoint")
        if name == EMBEDDING:
            values = np.stack([_bits(tensor[int(index)]) for index in ids[0]])
            snapshots[name] = {"format": "selected_token_rows_v1", "token_indices": ids[0].tolist(), "bits": values.tolist(),
                               "descriptor": _descriptor(values), "full_table_commitment": copy.deepcopy(commitment),
                               "membership_attestation": "selected_by_index_during_fresh_hash_verified_model_binding"}
        else:
            if name == FREQUENCY:
                if tensor.dtype != torch.float32:
                    raise ValueError("Local inverse frequency must retain FP32 bits")
                values = tensor.detach().contiguous().cpu().view(torch.int32).numpy().astype(np.uint32)
            else:
                values = _bits(tensor)
            snapshots[name] = {"format": "full_parameter_bits_v1", "bits": values.tolist(), "descriptor": _descriptor(values), "commitment": copy.deepcopy(commitment)}
    check_parameter_snapshots(program, token_ids, snapshots)
    return snapshots


def check_parameter_snapshots(program: dict[str, Any], token_ids: list[list[int]], snapshots: dict[str, Any]) -> dict[str, np.ndarray]:
    ids = _tokens(program, token_ids)
    reached = {name for node in first_layer_instructions(program) for name in node["parameter_refs"]}
    if set(snapshots) != reached:
        raise ValueError("Parameter snapshot coverage mismatch")
    result = {}
    for name, snapshot in snapshots.items():
        commitment = program["parameter_commitments"][name]
        if name == EMBEDDING:
            if set(snapshot) != {"format", "token_indices", "bits", "descriptor", "full_table_commitment", "membership_attestation"} or snapshot["format"] != "selected_token_rows_v1" or canonical_json(snapshot["token_indices"]) != canonical_json(ids[0].tolist()) or snapshot["full_table_commitment"] != commitment or snapshot["membership_attestation"] != "selected_by_index_during_fresh_hash_verified_model_binding":
                raise ValueError("Embedding selected-row token/commitment/format mismatch")
            value = _array(snapshot["bits"], [30, 640], "torch.bfloat16")
            for index in range(30):
                for prior in range(index):
                    if ids[0, index] == ids[0, prior] and not np.array_equal(value[index], value[prior]):
                        raise ValueError("Repeated token has inconsistent selected embedding rows")
        else:
            if set(snapshot) != {"format", "bits", "descriptor", "commitment"} or snapshot["format"] != "full_parameter_bits_v1" or snapshot["commitment"] != commitment:
                raise ValueError("Full parameter snapshot format/commitment mismatch")
            value = _array(snapshot["bits"], commitment["shape"], commitment["dtype"])
            if _descriptor(value)["sha256"] != commitment["sha256"]:
                raise ValueError("Parameter snapshot does not match checkpoint commitment")
        if _descriptor(value) != snapshot["descriptor"]:
            raise ValueError("Parameter snapshot descriptor/hash mismatch")
        result[name] = value.copy()
    return result


def _snapshot_model(sources: FirstLayerSources, model: Any, token_ids: list[list[int]]) -> dict[str, Any]:
    ff._model_context(sources.post_feedforward, model)
    parameters = bind_model_tensors(sources.program, model, verify_hashes=True)
    snapshots = snapshot_parameters(sources.program, token_ids, parameters)
    ff._model_context(sources.post_feedforward, model)
    bind_model_tensors(sources.program, model, verify_hashes=True)
    return snapshots


def _dot(left: list[int], right: list[int], mode: str, profile: Any) -> int:
    if mode == "serial":
        return operand_aligned_product_bits(left, right)
    if mode == "key_value":
        return split_k_candidate_bits(left, right, *profile)
    if mode == "output":
        return survivor_dot(left, right, profile)
    if mode == "down":
        return down_dot(left, right, profile)
    raise ValueError("Unknown row-worker arithmetic")


def _worker_init(weights: list[list[int]], mode: str, profile: Any, expected_code: str, weight_hash: str) -> None:
    global _WORKER
    if _code_sha() != expected_code or _sha(weights) != weight_hash:
        raise ValueError("First-layer worker source/weight commitment mismatch")
    expected = {"serial": canonical_json("operand_alignment_v1"), "key_value": canonical_json(list(DENSE_SPLIT_PROFILE)), "output": canonical_json(_OUTPUT_PROFILE), "down": SUPPORTED_CANDIDATE_JSON}
    if mode not in expected or canonical_json(profile) != expected[mode]:
        raise ValueError("First-layer worker profile commitment mismatch")
    _WORKER = (weights, mode, profile)


def _worker_row(task: tuple[int, list[int]]) -> tuple[int, list[int]]:
    weights, mode, profile = _WORKER
    index, row = task
    return index, [_dot(row, weight, mode, profile) for weight in weights]


def _project_rows(left: np.ndarray, weights: np.ndarray, mode: str, profile: Any, workers: int) -> np.ndarray:
    if left.dtype != np.uint16 or weights.dtype != np.uint16 or left.ndim != 2 or weights.ndim != 2 or left.shape[1] != weights.shape[1] or type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Invalid row-worker geometry/count")
    rows, tasks = weights.tolist(), list(enumerate(left.tolist()))
    if workers == 1:
        results = [(index, [_dot(row, weight, mode, profile) for weight in rows]) for index, row in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(rows, mode, profile, _code_sha(), _sha(rows))) as pool:
            results = list(pool.map(_worker_row, tasks))
    if [index for index, _ in results] != list(range(left.shape[0])):
        raise ValueError("First-layer worker reordered or omitted rows")
    return np.asarray([row for _, row in results], dtype=np.uint16).reshape(left.shape[0], weights.shape[0])


def _linear_mode(node: dict[str, Any]) -> str:
    name = node["outputs"][0]
    if name in ("layer.0.key.flat", "layer.0.value.flat"):
        return "key_value"
    if name == "layer.0.attention.projected":
        return "output"
    if name == "layer.0.mlp.down":
        return "down"
    return "serial"


def _provider_record(node: dict[str, Any], providers: Providers, profiles: dict[str, Any]) -> dict[str, Any]:
    opcode = node["opcode"]
    if opcode == "LINEAR":
        mode = _linear_mode(node)
        return {"name": mode, "profile": profiles[mode]}
    if opcode in ("MATMUL_QK", "MATMUL_AV"):
        return {"name": "operand_alignment_v1", "k": 256 if opcode == "MATMUL_QK" else 32, "zero_padding": 0 if opcode == "MATMUL_QK" else 2}
    if opcode in ("RMS_NORM", "SOFTMAX", "GELU_TANH"):
        key = {"RMS_NORM": "rsqrt", "SOFTMAX": "exp", "GELU_TANH": "gelu"}[opcode]
        return {"name": opcode, "checked_empirical_provider": key, "evidence": getattr(providers, key).evidence}
    if opcode == "ROTARY_TABLE":
        return {"name": "checked_empirical_local_rotary_constants", "evidence": providers.rotary_evidence}
    return {"name": {"ARANGE": "integer_positions", "EMBEDDING": "fresh_selected_coordinate_load", "SCALE": "bf16_multiply_rne", "ADD": "bf16_add_rne", "MUL": "bf16_multiply_rne", "CAUSAL_MASK": "integer_causal_coordinates", "ROTARY_APPLY_PAIR": "bf16_mul_mul_add_rne", "REPEAT_KV": "repeat_coordinates", "RESHAPE_TRANSPOSE_HEADS": "head_coordinates", "TRANSPOSE_RESHAPE_HEADS": "concatenate_coordinates"}[opcode]}


def _payload(node: dict[str, Any], states: dict[str, np.ndarray], state_hashes: dict[str, str], snapshots: dict[str, Any], providers: Providers, profiles: dict[str, Any], auxiliary: dict[str, Any]) -> dict[str, Any]:
    return {"instruction_id": node["id"], "instruction_sha256": node["instruction_sha256"], "opcode": node["opcode"],
            "inputs": {name: {"descriptor": _descriptor(states[name]), "producer_record_hash": state_hashes[name]} for name in node["inputs"]},
            "parameters": {name: {key: value for key, value in snapshots[name].items() if key != "bits"} for name in node["parameter_refs"]},
            "provider": _provider_record(node, providers, profiles), "outputs": {name: _descriptor(states[name]) for name in node["outputs"]},
            "auxiliary_stage_hashes": {name: _sha(value) for name, value in auxiliary.items()}}


def execute_first_layer(program: dict[str, Any], token_ids: list[list[int]], parameter_snapshots: dict[str, Any], providers: Providers, profiles: dict[str, Any], runtime: dict[str, Any], workers: int = 4) -> dict[str, Any]:
    code = _code_sha()
    nodes = first_layer_instructions(program)
    if type(workers) is not int or not 1 <= workers <= 4 or type(providers) is not Providers:
        raise ValueError("Invalid first-layer workers/provider container")
    _profiles(profiles)
    parameters = check_parameter_snapshots(program, token_ids, parameter_snapshots)
    providers.validate(runtime, parameters[FREQUENCY])
    states = {"input_ids": _tokens(program, token_ids)}
    state_hashes = {"input_ids": "ROOT:" + _descriptor(states["input_ids"])["sha256"]}
    records, scalar_stages = [], {}
    softmax_f32 = None
    for node in nodes:
        inputs = [states[name] for name in node["inputs"]]
        params = [parameters[name] for name in node["parameter_refs"]]
        opcode, attributes, output_name = node["opcode"], node["attributes"], node["outputs"][0]
        auxiliary = {}
        if opcode == "ARANGE":
            values = [np.arange(inputs[0].shape[1], dtype=np.int64)[None, :]]
        elif opcode == "EMBEDDING":
            if inputs[0].tolist()[0] != parameter_snapshots[EMBEDDING]["token_indices"]:
                raise ValueError("Embedding root tokens differ from selected fresh rows")
            values = [params[0][None, ...].copy()]
        elif opcode == "SCALE":
            scale = int(params[0].item()) if params else 0x3D80
            values = [np.asarray([bfloat16_multiply_bits(int(value), scale) for value in inputs[0].reshape(-1)], dtype=np.uint16).reshape(inputs[0].shape)]
        elif opcode == "RMS_NORM":
            rows = [_rms_lookup_row(row, params[0].tolist(), attributes["epsilon"], providers.rsqrt, runtime) for row in inputs[0].reshape(-1, inputs[0].shape[-1]).tolist()]
            values = [np.asarray([row["output_bits"] for row in rows], dtype=np.uint16).reshape(inputs[0].shape)]
            scalar_stages[output_name] = {key: [row[key] for row in rows] for key in STAGES}
            auxiliary[output_name] = scalar_stages[output_name]
        elif opcode == "LINEAR":
            mode = _linear_mode(node)
            values = [_project_rows(inputs[0][0], params[0], mode, profiles[mode], workers)[None, ...]]
        elif opcode == "RESHAPE_TRANSPOSE_HEADS":
            values = [inputs[0].reshape(1, 30, attributes["heads"], 256).transpose(0, 2, 1, 3).copy()]
        elif opcode == "ROTARY_TABLE":
            if inputs[1].tolist() != [list(range(30))]:
                raise ValueError("Rotary table input positions differ from checked domain")
            providers.validate(runtime, params[0])
            values = [providers.rotary_cosine.copy(), providers.rotary_sine.copy()]
        elif opcode == "CAUSAL_MASK":
            values = [causal_mask_bits()]
        elif opcode == "ROTARY_APPLY_PAIR":
            values = [rotate_bfloat16_bits(value, inputs[2], inputs[3]) for value in inputs[:2]]
        elif opcode == "REPEAT_KV":
            values = [np.repeat(inputs[0], attributes["repetitions"], axis=1)]
        elif opcode == "MATMUL_QK":
            values = [np.stack([_project_rows(inputs[0][0, head], inputs[1][0, head], "serial", profiles["serial"], workers) for head in range(4)])[None, ...]]
        elif opcode == "MATMUL_AV":
            values = [np.stack([_project_rows(np.pad(inputs[0][0, head], ((0, 0), (0, 2))), np.pad(inputs[1][0, head].T, ((0, 0), (0, 2))), "serial", profiles["serial"], workers) for head in range(4)])[None, ...]]
        elif opcode == "TRANSPOSE_RESHAPE_HEADS":
            values = [concatenate_heads(inputs[0]).copy()]
        elif opcode == "ADD" and node["inputs"][1] == "mask.sliding":
            values = [mask_score_bits(*inputs)]
        elif opcode in ("ADD", "MUL"):
            primitive = bfloat16_add_bits if opcode == "ADD" else bfloat16_multiply_bits
            if inputs[0].shape != inputs[1].shape:
                raise ValueError("Elementwise state shape mismatch")
            values = [np.asarray([primitive(int(left), int(right)) for left, right in zip(inputs[0].reshape(-1), inputs[1].reshape(-1))], dtype=np.uint16).reshape(inputs[0].shape)]
        elif opcode == "SOFTMAX":
            rows = [lookup_softmax_row([int(value) << 16 for value in row], providers.exp, runtime) for row in inputs[0].reshape(-1, 30)]
            values = [np.asarray([row["output_bf16_bits"] for row in rows], dtype=np.uint16).reshape(1, 4, 30, 30)]
            softmax_f32 = np.asarray([row["output_f32_bits"] for row in rows], dtype=np.uint32).reshape(1, 4, 30, 30).tolist()
            auxiliary["softmax_f32_bits"] = softmax_f32
        elif opcode == "GELU_TANH":
            values = [np.asarray([providers.gelu.predict_bits(int(value), runtime) for value in inputs[0].reshape(-1)], dtype=np.uint16).reshape(inputs[0].shape)]
        else:
            raise ValueError("Unregistered first-layer opcode")
        if len(values) != len(node["outputs"]):
            raise ValueError("Missing multi-output state")
        for name, value in zip(node["outputs"], values):
            if name in states:
                raise ValueError("State overwritten during connected execution")
            states[name] = _array(value, _shape(program, name), program["tensors"][name]["dtype"]).copy()
        record = append_chain_record(records, _payload(node, states, state_hashes, parameter_snapshots, providers, profiles, auxiliary))
        state_hashes.update({name: record["record_hash"] for name in node["outputs"]})
    result = {"state_bits": {name: value.tolist() for name, value in states.items() if name != "input_ids"}, "scalar_stages": scalar_stages, "softmax_f32_bits": softmax_f32, "records": records}
    check_execution(program, token_ids, parameter_snapshots, providers, profiles, runtime, result)
    if code != _code_sha():
        raise ValueError("First-layer arithmetic/helper source changed during execution")
    return result


def check_execution(program: dict[str, Any], token_ids: list[list[int]], snapshots: dict[str, Any], providers: Providers, profiles: dict[str, Any], runtime: dict[str, Any], execution: dict[str, Any]) -> None:
    nodes = first_layer_instructions(program)
    parameters = check_parameter_snapshots(program, token_ids, snapshots)
    _profiles(profiles)
    providers.validate(runtime, parameters[FREQUENCY])
    if set(execution) != {"state_bits", "scalar_stages", "softmax_f32_bits", "records"} or set(execution["state_bits"]) != {name for node in nodes for name in node["outputs"]}:
        raise ValueError("Connected output state coverage mismatch")
    rms_nodes = [node for node in nodes if node["opcode"] == "RMS_NORM"]
    if set(execution["scalar_stages"]) != {node["outputs"][0] for node in rms_nodes}:
        raise ValueError("Missing RMS auxiliary stages")
    for node in rms_nodes:
        name = node["outputs"][0]
        count = int(np.prod(_shape(program, name)[:-1]))
        if set(execution["scalar_stages"][name]) != set(STAGES):
            raise ValueError("Unexpected RMS auxiliary keys")
        for stage in STAGES:
            _array(execution["scalar_stages"][name][stage], [count], "torch.float32")
    _array(execution["softmax_f32_bits"], [1, 4, 30, 30], "torch.float32")
    records = execution["records"]
    if len(records) != 34 or not verify_trace_chain(records)["valid"]:
        raise ValueError("Missing or invalid connected trace chain")
    states = {"input_ids": _tokens(program, token_ids)}
    hashes = {"input_ids": "ROOT:" + _descriptor(states["input_ids"])["sha256"]}
    expected_records = []
    for node, record in zip(nodes, records):
        auxiliary = {}
        for name in node["outputs"]:
            if name in states:
                raise ValueError("Duplicate ledger output")
            states[name] = _array(execution["state_bits"][name], _shape(program, name), program["tensors"][name]["dtype"])
        if node["opcode"] == "RMS_NORM":
            auxiliary[node["outputs"][0]] = execution["scalar_stages"][node["outputs"][0]]
        if node["opcode"] == "SOFTMAX":
            auxiliary["softmax_f32_bits"] = execution["softmax_f32_bits"]
        expected = append_chain_record(expected_records, _payload(node, states, hashes, snapshots, providers, profiles, auxiliary))
        if canonical_json(expected) != canonical_json(record):
            raise ValueError("Connected ledger input/output/parameter/provider/auxiliary mismatch")
        hashes.update({name: record["record_hash"] for name in node["outputs"]})


def _coverage(program: dict[str, Any]) -> dict[str, Any]:
    nodes = first_layer_instructions(program)
    return {"target": "hidden.1", "root_inputs": ["input_ids"], "instruction_ids": [node["id"] for node in nodes], "instruction_count": 34,
            "states": {name: {"shape": _shape(program, name), "dtype": program["tensors"][name]["dtype"]} for node in nodes for name in node["outputs"]},
            "state_count": 36, "excluded_prefix_nodes": copy.deepcopy(EXCLUDED), "all_36_prefix_instructions_claimed": False,
            "rms_scalar_positions": 810, "softmax_fp32_positions": 3600, "hidden_1_value_count": 19200}


def _native_kernel_roles(program: dict[str, Any]) -> list[str]:
    return sorted("softmax" if node["opcode"] == "SOFTMAX" else node["outputs"][0] for node in first_layer_instructions(program)
                  if node["opcode"] in ("RMS_NORM", "LINEAR", "GELU_TANH", "MATMUL_QK", "MATMUL_AV", "ADD", "MUL", "SOFTMAX") or node["outputs"] == ["layer.0.attention.scaled_scores"])


def _source_kernel_expectations(sources: FirstLayerSources) -> dict[str, list[str]]:
    product = sources.post_feedforward.dense_sources.down.product
    entry, post = product.entry, product.entry.post
    output = post.survivor_context[0]
    groups = {
        "layer.0.mlp.post_normalized": [pair["traced"]["cuda_events"] for pair in sources.post_feedforward_report["observations"]],
        "hidden.1": [pair["traced"]["add"]["kernel_names"] for pair in sources.post_feedforward_report["observations"]],
        "layer.0.attention.post_normalized": [pair["traced"]["cuda_events"] for pair in entry.post_report["observations"]],
        "layer.0.mlp.normalized": [pair["traced"]["norm_cuda_events"] for pair in product.entry_report["observations"]],
        "layer.0.mlp.gate": [pair["traced"]["gate_cuda_events"] for pair in product.entry_report["observations"]],
        "layer.0.mlp.up": [pair["traced"]["up_cuda_events"] for pair in product.entry_report["observations"]],
        "layer.0.mlp.down": [sources.post_feedforward.dense_sources.model_plan["expected_kernel_names"]] * 3,
        "layer.0.attention.head_output": [pair["traced"]["pv_cuda_events"] for pair in post.survivor_report["native_report"]["observations"]],
        "layer.0.attention.projected": [pair["traced"]["projection_cuda_events"] for pair in post.survivor_report["native_report"]["observations"]],
        "softmax": [pair["traced"]["cuda_events"] for pair in output.lookup_report["native_report"]["observations"]],
    }
    expected = {}
    for role, repetitions in groups.items():
        normalized = [ff.down._kernel_symbols(names) for names in repetitions]
        if len(normalized) != 3 or any(names != normalized[0] for names in normalized):
            raise ValueError("Prior role kernel symbols are not repeat stable: " + role)
        expected[role] = normalized[0]
    return expected


def _plan_body(sources: FirstLayerSources, token_ids: list[list[int]], providers: Providers, profiles: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": 1, "scope": SCOPE, "program_sha256": sources.program["program_sha256"], "coverage": _coverage(sources.program),
            "sources": sources.commitments(), "code_sha256": _code_sha(), "runtime": copy.deepcopy(sources.runtime),
            "native_kernel_roles": _native_kernel_roles(sources.program), "prior_role_kernel_symbols": _source_kernel_expectations(sources),
            "model_binding": copy.deepcopy(sources.post_feedforward_plan["model_binding"]), "input_token_ids": token_ids,
            "input_ids": _descriptor(_tokens(sources.program, token_ids)), "providers": providers.commitments(), "profiles": profiles,
            "parameter_snapshots": {name: {key: value for key, value in snapshot.items() if key != "bits"} for name, snapshot in bundle["parameter_snapshots"].items()},
            "state_descriptors": {name: _descriptor(_array(value, _shape(sources.program, name), sources.program["tensors"][name]["dtype"])) for name, value in bundle["execution"]["state_bits"].items()},
            "trace_root": bundle["execution"]["records"][-1]["record_hash"], "bundle_sha256": _sha(bundle),
            "repetitions": 3, "original_forward_count": 6, "epoch_scope": "fixed_previously_observed_source_case_only",
            "parameter_snapshot_membership": "embedding_selected_rows_attested_during_fresh_model_binding; revalidation_requires_model",
            "empirical_primitive_data_reused": True, "connected_first_layer_independently_recomputed": False,
            **{key: False for key in FALSE_FLAGS}}


def build_first_layer_plan(sources: FirstLayerSources, model: Any, workers: int = 4, input_token_ids: list[list[int]] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    code, guard = _code_sha(), _sha(sources.commitments())
    ids, providers, profiles = sources.validate()
    if input_token_ids is not None and canonical_json(input_token_ids) != canonical_json(ids):
        raise ValueError("Public first-layer plan is restricted to the declared source token case")
    snapshots = _snapshot_model(sources, model, ids)
    execution = execute_first_layer(sources.program, ids, snapshots, providers, profiles, sources.runtime, workers)
    if canonical_json(_snapshot_model(sources, model, ids)) != canonical_json(snapshots):
        raise ValueError("Live checkpoint parameter snapshots changed during execution")
    bundle = {"parameter_snapshots": snapshots, "execution": execution}
    plan = _seal(_plan_body(sources, ids, providers, profiles, bundle), "plan_sha256")
    if code != _code_sha() or guard != _sha(sources.commitments()):
        raise ValueError("Source lineage changed during connected execution")
    return plan, bundle


def _check_plan(sources: FirstLayerSources, plan: dict[str, Any], bundle: dict[str, Any], context: tuple[Any, ...]) -> None:
    _check_hash(plan, "plan_sha256")
    if set(bundle) != {"parameter_snapshots", "execution"}:
        raise ValueError("Unknown first-layer prediction bundle material")
    ids, providers, profiles = context
    check_execution(sources.program, ids, bundle["parameter_snapshots"], providers, profiles, sources.runtime, bundle["execution"])
    expected = _seal(_plan_body(sources, ids, providers, profiles, bundle), "plan_sha256")
    if canonical_json(plan) != canonical_json(expected):
        raise ValueError("First-layer frozen plan/bundle binding mismatch; no refit")


def check_first_layer_plan(sources: FirstLayerSources, plan: dict[str, Any], bundle: dict[str, Any]) -> None:
    code, guard = _code_sha(), _sha(sources.commitments())
    _check_plan(sources, plan, bundle, sources.validate())
    if code != _code_sha() or guard != _sha(sources.commitments()):
        raise ValueError("Source changed during first-layer integrity check")


def _comparison(expected: Any, actual: Any, shape: list[int], dtype: str) -> dict[str, Any]:
    left, right = _array(expected, shape, dtype), _array(actual, shape, dtype)
    different = np.argwhere(left != right)
    first = different[0].tolist() if len(different) else None
    return {"value_count": int(left.size), "mismatch_count": len(different), "first_divergence": None if first is None else {"coordinate": first, "expected_bits": int(left[tuple(first)]), "actual_bits": int(right[tuple(first)])}}


def _geometry(geometry: Any, shape: list[int], dtype: str) -> bool:
    if not isinstance(geometry, dict):
        return False
    strides = geometry.get("strides")
    return (geometry.get("shape") == shape and geometry.get("dtype") == dtype and isinstance(strides, list) and len(strides) == len(shape)
            and all(type(stride) is int and stride >= 0 for stride in strides) and str(geometry.get("device", "")).startswith("cuda")
            and type(geometry.get("alignment_mod16")) is int and geometry["alignment_mod16"] == 0)


def first_layer_report(plan: dict[str, Any], bundle: dict[str, Any], observations: list[dict[str, Any]]) -> dict[str, Any]:
    from . import gemma_first_layer_capture as capture
    _check_hash(plan, "plan_sha256")
    if plan["scope"] != SCOPE or any(plan.get(key) is not False for key in FALSE_FLAGS) or _sha(bundle) != plan["bundle_sha256"]:
        raise ValueError("Native comparison requires the frozen bounded plan/bundle")
    if not isinstance(observations, list) or len(observations) != 3:
        raise ValueError("Expected three original first-layer traced/plain pairs")
    execution, coverage = bundle["execution"], plan["coverage"]
    if len(execution["records"]) != 34 or not verify_trace_chain(execution["records"], plan["trace_root"])["valid"]:
        raise ValueError("Native comparison requires the complete connected prediction ledger")
    comparisons, controls, kernel_sets = [], [], []
    checks = {"capture_checks": True, "geometry": True, "runtime": True, "capture_code": True, "nonempty_kernel_sets": True, "prior_role_kernels_match": True, "same_cuda_device": True}
    for pair in observations:
        if set(pair) != {"traced", "untraced"}:
            raise ValueError("Original observation pair fields mismatch")
        traced, plain = pair["traced"], pair["untraced"]
        if set(traced["state_bits"]) != set(coverage["states"]) or set(traced["geometry"]) != set(coverage["states"]) or set(plain["state_bits"]) != {"hidden.1"} or set(plain["geometry"]) != {"hidden.1"}:
            raise ValueError("Native state/geometry coverage mismatch")
        if any(plain.get(key) for key in ("scalar_stages", "softmax_f32_bits", "kernels")):
            raise ValueError("Plain control must use only the minimal hidden.1 hook")
        result = {}
        for name, declaration in coverage["states"].items():
            result[name] = _comparison(execution["state_bits"][name], traced["state_bits"][name], declaration["shape"], declaration["dtype"])
            checks["geometry"] &= _geometry(traced["geometry"][name], declaration["shape"], declaration["dtype"])
        if set(traced["scalar_stages"]) != set(execution["scalar_stages"]):
            raise ValueError("Native RMS scalar coverage mismatch")
        for name, stages in execution["scalar_stages"].items():
            native = traced["scalar_stages"][name]
            if set(native) != {*STAGES, "mean_input_metadata"}:
                raise ValueError("Native RMS auxiliary stage fields mismatch")
            for key in STAGES:
                result[name + ":" + key] = _comparison(stages[key], native[key], [len(stages[key])], "torch.float32")
            metadata = native["mean_input_metadata"]
            shape = coverage["states"][name]["shape"]
            strides = metadata.get("input_strides")
            checks["geometry"] &= metadata.get("input_shape") == shape and metadata.get("input_dtype") == "torch.float32" and metadata.get("axes") == [-1] and metadata.get("keepdim") is True and metadata.get("alignment_mod16") == 0 and isinstance(strides, list) and len(strides) == len(shape) and strides[-1] == 1 and all(type(stride) is int and stride > 0 and stride % 4 == 0 for stride in strides[:-1])
        result["softmax_f32_bits"] = _comparison(execution["softmax_f32_bits"], traced["softmax_f32_bits"], [1, 4, 30, 30], "torch.float32")
        controls.append(_comparison(execution["state_bits"]["hidden.1"], plain["state_bits"]["hidden.1"], [1, 30, 640], "torch.bfloat16"))
        checks["geometry"] &= _geometry(plain["geometry"]["hidden.1"], [1, 30, 640], "torch.bfloat16")
        checks["same_cuda_device"] &= len({value["device"] for record in (traced, plain) for value in record["geometry"].values()}) == 1
        for record in (traced, plain):
            required = {"token_commitment_unchanged", "stopped_at_decoder_tuple", "decoder_tuple_length_one", "no_later_layer", "no_final_norm", "no_lm_head", "all_needed_states_present", "code_unchanged", "runtime_unchanged"}
            required.update({"original_operations_once", "all_operand_links_unchanged", "rms_scalar_stages_complete", "softmax_f32_complete"} if record is traced else {"plain_minimal_control"})
            checks["capture_checks"] &= isinstance(record.get("checks"), dict) and required.issubset(record["checks"]) and all(value is True for value in record["checks"].values())
            checks["runtime"] &= canonical_json(record["runtime_before"]) == canonical_json(plan["runtime"]) == canonical_json(record["runtime_after"])
            checks["capture_code"] &= record["code_before"] == capture._code_sha() == record["code_after"]
        kernels = traced["kernels"]
        checks["nonempty_kernel_sets"] &= isinstance(kernels, dict) and set(kernels) == set(plan["native_kernel_roles"]) and all(isinstance(role, str) and role and isinstance(names, list) and names and all(isinstance(name, str) and name.strip() for name in names) and names == sorted(set(names)) for role, names in kernels.items())
        checks["prior_role_kernels_match"] &= all(kernels.get(role) == names for role, names in plan["prior_role_kernel_symbols"].items())
        kernel_sets.append(kernels)
        comparisons.append(result)
    checks["repeat_stable_kernel_sets"] = all(canonical_json(item) == canonical_json(kernel_sets[0]) for item in kernel_sets)
    counts = {name: sum(item[name]["mismatch_count"] for item in comparisons) for name in comparisons[0]}
    counts["plain_hidden.1"] = sum(item["mismatch_count"] for item in controls)
    first = next(({"repetition": index, "state": name, **result["first_divergence"]} for index, comparison in enumerate(comparisons) for name, result in comparison.items() if result["first_divergence"] is not None), None)
    if first is None:
        first = next(({"repetition": index, "state": "plain_hidden.1", **item["first_divergence"]} for index, item in enumerate(controls) if item["first_divergence"] is not None), None)
    if first is None:
        first = next(({"check": name} for name, passed in checks.items() if not passed), None)
    matches = all(checks.values()) and not any(counts.values())
    return _seal({"schema_version": 1, "scope": SCOPE, "plan_sha256": plan["plan_sha256"], "bundle_sha256": _sha(bundle), "coverage": coverage,
                  "observations": observations, "comparisons": comparisons, "untraced_controls": controls, "checks": checks,
                  "mismatch_counts": counts, "first_divergence": first, "first_layer_matches": matches,
                  "connected_first_layer_independently_recomputed": matches, "empirical_primitive_data_reused": True,
                  "kernel_provenance": "distinct-symbol-set-only; no launch-order or native-internal claim", **{key: False for key in FALSE_FLAGS}}, "report_sha256")


def acquire_first_layer(sources: FirstLayerSources, model: Any, plan: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    from .gemma_first_layer_capture import capture_first_layer
    code, guard = _code_sha(), _sha(sources.commitments())
    context = sources.validate()
    _check_plan(sources, plan, bundle, context)
    if canonical_json(_snapshot_model(sources, model, context[0])) != canonical_json(bundle["parameter_snapshots"]):
        raise ValueError("Fresh acquisition checkpoint/embedding membership differs from plan")
    observations = [{"traced": capture_first_layer(model, context[0], True), "untraced": capture_first_layer(model, context[0], False)} for _ in range(3)]
    if canonical_json(_snapshot_model(sources, model, context[0])) != canonical_json(bundle["parameter_snapshots"]) or code != _code_sha() or guard != _sha(sources.commitments()):
        raise ValueError("Source/runtime/model changed across native acquisition")
    return first_layer_report(plan, bundle, observations)


def verify_first_layer(sources: FirstLayerSources, plan: dict[str, Any], bundle: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    try:
        code, guard = _code_sha(), _sha(sources.commitments())
        _check_plan(sources, plan, bundle, sources.validate())
        _check_hash(report, "report_sha256")
        expected = first_layer_report(plan, bundle, report["observations"])
        valid = canonical_json(report) == canonical_json(expected) and code == _code_sha() and guard == _sha(sources.commitments())
        return {"valid": valid, "mode": "integrity_hashchain_snapshot_coverage_and_report_recount_only",
                "connected_numerical_recomputation_performed": False, "embedding_membership_revalidated_with_model": False,
                "previous_lineage_validation": "full_source_integrity_and_short_RMS_recomputation_not_all_previous_matmuls",
                "first_layer_matches": expected["first_layer_matches"], "connected_first_layer_independently_recomputed": valid and expected["connected_first_layer_independently_recomputed"],
                "mismatch_counts": expected["mismatch_counts"], "first_divergence": expected["first_divergence"], **{key: False for key in FALSE_FLAGS}}
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError, IndexError, StopIteration) as error:
        return {"valid": False, "reason": str(error), "connected_numerical_recomputation_performed": False, "embedding_membership_revalidated_with_model": False, "connected_first_layer_independently_recomputed": False, **{key: False for key in FALSE_FLAGS}}


def reexecute_first_layer(sources: FirstLayerSources, model: Any, plan: dict[str, Any], bundle: dict[str, Any], report: dict[str, Any], workers: int = 4) -> dict[str, Any]:
    regenerated_plan, regenerated_bundle = build_first_layer_plan(sources, model, workers)
    same = canonical_json(regenerated_plan) == canonical_json(plan) and canonical_json(regenerated_bundle) == canonical_json(bundle)
    replay = acquire_first_layer(sources, model, regenerated_plan, regenerated_bundle) if same else None
    exact = same and canonical_json(replay) == canonical_json(report)
    return {"valid": exact, "mode": "all_34_nodes_fresh_parameter_regeneration_before_six_native_forward_replay",
            "predictions_recomputed_exact": same, "reexecution_exact": exact, "connected_numerical_recomputation_performed": True,
            "embedding_membership_revalidated_with_model": True, "connected_first_layer_independently_recomputed": exact and replay["first_layer_matches"], **{key: False for key in FALSE_FLAGS}}


def first_layer_summary(plan: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in report.items() if key not in ("observations", "comparisons", "untraced_controls", "report_sha256")}
    body.update({"source_report_sha256": report["report_sha256"], "sources": plan["sources"], "code_sha256": plan["code_sha256"], "trace_root": plan["trace_root"], "providers": plan["providers"]})
    return _seal(body, "summary_sha256")
