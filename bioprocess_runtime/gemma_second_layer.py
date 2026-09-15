from __future__ import annotations

import copy
import hashlib
import inspect
import marshal
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import gemma_first_layer as first
from . import gemma_first_layer_holdout as holdout
from .gemma_rotary_slice import _sha, _seal, _check_hash
from .operational_semantics import append_chain_record, verify_trace_chain
from .serialization import canonical_json

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
HOLDOUT_PROTOCOL_SHA256 = "37af351612e293750f60678176980cb4b735467756b20641a909cf71a4c31905"
HOLDOUT_PLAN_SHA256 = "669affa2474328bf4af99b4d6571124f7d1bc9eb5707a1150854ee538e35f16b"
HOLDOUT_REPORT_SHA256 = "ec6e51abcabb97996ac05a5b68db0e7e09af8bae0370c4675de1e72aa442b3f4"
ROOTS = ("hidden.1", "rotary.local.cosine", "rotary.local.sine", "mask.sliding")
ROOT_SHAPES = ([1, 30, 640], [1, 30, 256], [1, 30, 256], [1, 1, 30, 30])
SCOPE = "Bounded actual Gemma3 270m decoder index 1, sliding attention with local RoPE and window 512: 29 independently computed instructions i0036..i0064, 30 new states hidden.1-to-hidden.2. Reuses the verified layer-zero hidden.1 boundary and checked local rotary/mask roots from the previously observed baseline token case. Fresh actual layer-one weights; unchanged empirical arithmetic profiles are a transfer hypothesis, not prequalified. First-layer holdouts are prerequisite evidence, not new layer-one cases. Six RMS scalar stages (810 positions) and softmax FP32 (3600 positions) only. Not fresh connected two-layer prediction, fresh-prompt holdout, full-layer/model, hardware or unrestricted qualification. No layer index 2, final norm or logits."
FALSE_FLAGS = ("qualified", "completeFirstLayerQualified", "full_first_layer_qualified", "full_model_qualified", "hardware_semantics_established", "global_exactness_activation_allowed", "qualification_promotion_allowed", "candidate_refitting_allowed", "native_arithmetic_reconstructed", "all_fp32_vector_internals_observed", "native_fused_registers_observed", "fresh_prompt_holdout", "fresh_layer_one_holdout", "later_layers_executed", "final_model_norm_executed", "logits_executed", "connected_two_layers_independently_recomputed", "connected_first_layer_independently_recomputed", "internal_layer_one_intermediates_reused", "shape_transfer_prequalified")
LINEAR_MODES = {"layer.1.query.flat": "serial", "layer.1.key.flat": "key_value", "layer.1.value.flat": "key_value", "layer.1.attention.projected": "output", "layer.1.mlp.gate": "serial", "layer.1.mlp.up": "serial", "layer.1.mlp.down": "down"}


def _flags() -> dict[str, bool]:
    return {**dict.fromkeys(FALSE_FLAGS, False), "prefix_boundary_reused": True, "stored_boundary_predictions_used": True,
            "stored_intermediate_predictions_used": True, "empirical_primitive_data_reused": True, "empirical_providers_reused": True}


def _implementation_commitment() -> dict[str, Any]:
    import torch
    from transformers.models.gemma3 import modeling_gemma3 as gemma
    from transformers.activations import PytorchGELUTanh
    functions = [cls.forward for cls in (gemma.Gemma3ForCausalLM, gemma.Gemma3TextModel, gemma.Gemma3DecoderLayer, gemma.Gemma3Attention, gemma.Gemma3MLP, gemma.Gemma3RMSNorm, torch.nn.Linear, PytorchGELUTanh)]
    functions.extend((gemma.apply_rotary_pos_emb, gemma.repeat_kv, gemma.eager_attention_forward))
    result = {}
    for function in functions:
        name, versions = function.__module__ + "." + function.__qualname__, []
        seen = set()
        while function is not None:
            if id(function) in seen:
                raise ValueError("Cyclic native method wrapper")
            seen.add(id(function))
            versions.append({"source_sha256": hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest(), "bytecode_sha256": hashlib.sha256(marshal.dumps(function.__code__)).hexdigest()})
            function = getattr(function, "__wrapped__", None)
        result[name] = versions
    return result


def _code_sha() -> str:
    from . import gemma_second_layer_capture as capture
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Second-layer source changed after import")
    return _sha({"module": SOURCE_SHA256, "arithmetic": first._code_sha(), "holdout": holdout._code_sha(), "capture": capture._code_sha(), "native_model_implementation": _implementation_commitment()})


def second_layer_instructions(program: dict[str, Any]) -> list[dict[str, Any]]:
    if program.get("program_sha256") != first.PROGRAM_SHA256 or not first.verify_gemma_ir(program)["valid"]:
        raise ValueError("Second-layer requires the exact fixed-checkpoint typed IR")
    config = program["configuration"]
    if config["layer_types"][:6] != ["sliding_attention"] * 5 + ["full_attention"] or any(config[name] != value for name, value in {"hidden_size": 640, "intermediate_size": 2048, "attention_heads": 4, "key_value_heads": 1, "head_dimension": 256, "attention_scale": 0.0625, "sliding_window": 512}.items()):
        raise ValueError("Unsupported layer-one geometry or sliding regime")
    selected = [node for node in program["instructions"] if node["layer"] == 1]
    if [node["id"] for node in selected] != [f"i{index:04d}" for index in range(36, 65)]:
        raise ValueError("Second-layer exact instruction order mismatch")
    available, needed = set(ROOTS), set()
    for node in selected:
        _check_hash(node, "instruction_sha256")
        if type(node["layer"]) is not int or any(name not in available for name in node["inputs"]) or available.intersection(node["outputs"]):
            raise ValueError("Second-layer dependency/producer mismatch")
        needed.update(set(node["inputs"]) & set(ROOTS))
        available.update(node["outputs"])
        for name in node["outputs"]:
            declaration = program["tensors"][name]
            if declaration["producer"] != node["id"] or declaration["dtype"] != "torch.bfloat16":
                raise ValueError("Second-layer tensor declaration mismatch")
        for name in node["parameter_refs"]:
            if not name.startswith("model.layers.1.") or program["parameter_commitments"][name]["dtype"] != "torch.bfloat16":
                raise ValueError("Second-layer must use actual layer-one BF16 parameters")
    if needed != set(ROOTS) or len(available) != 34 or selected[-1]["outputs"] != ["hidden.2"]:
        raise ValueError("Second-layer root/output closure mismatch")
    for name, shape in zip(ROOTS, ROOT_SHAPES):
        if first._shape(program, name) != shape or program["tensors"][name]["dtype"] != "torch.bfloat16":
            raise ValueError("Second-layer root declaration mismatch")
    return selected


def _roots(program: dict[str, Any], root_bits: dict[str, Any], providers: first.Providers) -> dict[str, np.ndarray]:
    if set(root_bits) != set(ROOTS):
        raise ValueError("Exactly four stored boundary roots are required")
    values = {name: first._array(root_bits[name], shape, "torch.bfloat16").copy() for name, shape in zip(ROOTS, ROOT_SHAPES)}
    for name, expected in ((ROOTS[1], providers.rotary_cosine), (ROOTS[2], providers.rotary_sine), (ROOTS[3], first.causal_mask_bits())):
        if not np.array_equal(values[name], expected):
            raise ValueError("Stored local rotary/mask root differs from checked provider")
    return values


def check_parameter_snapshots(program: dict[str, Any], snapshots: dict[str, Any]) -> dict[str, np.ndarray]:
    reached = {name for node in second_layer_instructions(program) for name in node["parameter_refs"]}
    if len(reached) != 13 or set(snapshots) != reached:
        raise ValueError("Exactly thirteen actual layer-one parameter snapshots are required")
    values = {}
    for name, snapshot in snapshots.items():
        commitment = program["parameter_commitments"][name]
        if set(snapshot) != {"format", "bits", "descriptor", "commitment"} or snapshot["format"] != "full_parameter_bits_v1" or snapshot["commitment"] != commitment or commitment["dtype"] != "torch.bfloat16":
            raise ValueError("Layer-one full parameter snapshot format/dtype/commitment mismatch")
        value = first._array(snapshot["bits"], commitment["shape"], "torch.bfloat16")
        if first._descriptor(value) != snapshot["descriptor"] or first._descriptor(value)["sha256"] != commitment["sha256"]:
            raise ValueError("Layer-one parameter snapshot checkpoint hash mismatch")
        values[name] = value.copy()
    return values


def snapshot_parameters(program: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    snapshots = {}
    for name in sorted({name for node in second_layer_instructions(program) for name in node["parameter_refs"]}):
        tensor, commitment = parameters[name], program["parameter_commitments"][name]
        if list(tensor.shape) != commitment["shape"] or str(tensor.dtype) != "torch.bfloat16":
            raise ValueError("Fresh actual layer-one tensor geometry/dtype mismatch")
        value = first._bits(tensor)
        snapshots[name] = {"format": "full_parameter_bits_v1", "bits": value.tolist(), "descriptor": first._descriptor(value), "commitment": copy.deepcopy(commitment)}
    check_parameter_snapshots(program, snapshots)
    return snapshots


def _original(module: Any, cls: Any) -> None:
    method = getattr(module.forward, "__func__", None)
    if type(module) is not cls or method is not cls.forward or method.__module__ != cls.__module__ or method.__qualname__ != cls.__qualname__ + ".forward" or "forward" in module.__dict__:
        raise ValueError("Expected original unmodified layer-one class and forward method")
    if module.training or module._forward_hooks or module._forward_pre_hooks:
        raise ValueError("Layer-one must be evaluation mode without preinstalled hooks")


def _model_context(sources: SecondLayerSources, model: Any) -> dict[str, Any]:
    import torch
    from transformers.models.gemma3 import modeling_gemma3 as gemma
    from transformers.activations import PytorchGELUTanh
    first.ff._model_context(sources.baseline.first_sources.post_feedforward, model)
    parameters = first.bind_model_tensors(sources.program, model, verify_hashes=True)
    layer = model.model.layers[1]
    attention, mlp = layer.self_attn, layer.mlp
    for module, cls in ((layer, gemma.Gemma3DecoderLayer), (attention, gemma.Gemma3Attention), (mlp, gemma.Gemma3MLP), (mlp.act_fn, PytorchGELUTanh)):
        _original(module, cls)
    if layer.layer_idx != 1 or layer.attention_type != "sliding_attention" or layer.hidden_size != 640 or attention.layer_idx != 1 or attention.is_sliding is not True or attention.sliding_window != 512 or attention.attn_logit_softcapping is not None or attention.config is not model.config or model.config._attn_implementation != "eager" or model.config.layer_types[1] != "sliding_attention" or model.config.sliding_window != 512 or attention.head_dim != 256 or attention.num_key_value_groups != 4 or attention.scaling != 0.0625 or attention.attention_dropout != 0.0:
        raise ValueError("Actual checkpoint layer one must use local RoPE and sliding attention")
    for field, expected in (("hidden_size", 640), ("intermediate_size", 2048), ("num_attention_heads", 4), ("num_key_value_heads", 1), ("head_dim", 256), ("rms_norm_eps", 1e-6)):
        if getattr(model.config, field) != expected:
            raise ValueError("Actual layer-one configuration geometry differs from IR")
    for node in second_layer_instructions(sources.program):
        if not node["parameter_refs"]:
            continue
        name = node["parameter_refs"][0]
        module = model
        for component in name.split(".")[:-1]:
            module = module[int(component)] if component.isdigit() else getattr(module, component)
        _original(module, torch.nn.Linear if node["opcode"] == "LINEAR" else gemma.Gemma3RMSNorm)
        weight = parameters[name]
        if module.weight is not weight or weight.dtype != torch.bfloat16 or not weight.is_contiguous() or weight.device != model.model.embed_tokens.weight.device:
            raise ValueError("Layer-one weight identity/geometry mismatch")
        if node["opcode"] == "LINEAR":
            if module.bias is not None or [module.out_features, module.in_features] != list(weight.shape):
                raise ValueError("Layer-one LINEAR bias/dimensions mismatch")
        elif module.eps != 1e-6 or list(weight.shape) != [first._shape(sources.program, node["inputs"][0])[-1]]:
            raise ValueError("Layer-one RMS epsilon/dimensions mismatch")
    if canonical_json(first._runtime()) != canonical_json(sources.runtime):
        raise ValueError("Second-layer runtime differs from frozen source")
    return parameters


def _snapshot_model(sources: SecondLayerSources, model: Any) -> dict[str, Any]:
    snapshots = snapshot_parameters(sources.program, _model_context(sources, model))
    _model_context(sources, model)
    return snapshots


def _linear_mode(node: dict[str, Any]) -> str:
    if node["outputs"][0] not in LINEAR_MODES:
        raise ValueError("Unregistered layer-one LINEAR profile")
    return LINEAR_MODES[node["outputs"][0]]


def _provider_record(node: dict[str, Any], providers: first.Providers, profiles: dict[str, Any]) -> dict[str, Any]:
    if node["opcode"] == "LINEAR":
        mode = _linear_mode(node)
        return {"name": mode, "profile": profiles[mode], "shape_transfer_prequalified": False}
    return first._provider_record(node, providers, profiles)


def _payload(node: dict[str, Any], states: dict[str, np.ndarray], hashes: dict[str, str], snapshots: dict[str, Any], providers: first.Providers, profiles: dict[str, Any], auxiliary: dict[str, Any]) -> dict[str, Any]:
    return {"instruction_id": node["id"], "instruction_sha256": node["instruction_sha256"], "opcode": node["opcode"],
            "inputs": {name: {"descriptor": first._descriptor(states[name]), "producer_record_hash": hashes[name]} for name in node["inputs"]},
            "parameters": {name: {key: value for key, value in snapshots[name].items() if key != "bits"} for name in node["parameter_refs"]},
            "provider": _provider_record(node, providers, profiles), "outputs": {name: first._descriptor(states[name]) for name in node["outputs"]},
            "auxiliary_stage_hashes": {name: _sha(value) for name, value in auxiliary.items()}}


def execute_second_layer(program: dict[str, Any], root_bits: dict[str, Any], parameter_snapshots: dict[str, Any], providers: first.Providers, profiles: dict[str, Any], runtime: dict[str, Any], workers: int = 4) -> dict[str, Any]:
    code = _code_sha()
    nodes = second_layer_instructions(program)
    if type(workers) is not int or not 1 <= workers <= 4 or type(providers) is not first.Providers:
        raise ValueError("Invalid second-layer workers/provider container")
    first._profiles(profiles)
    providers.validate(runtime)
    parameters = check_parameter_snapshots(program, parameter_snapshots)
    states = _roots(program, root_bits, providers)
    hashes = {name: "ROOT:" + first._descriptor(value)["sha256"] for name, value in states.items()}
    records, scalar_stages, softmax_f32 = [], {}, None
    for node in nodes:
        inputs = [states[name] for name in node["inputs"]]
        params = [parameters[name] for name in node["parameter_refs"]]
        opcode, attributes, name = node["opcode"], node["attributes"], node["outputs"][0]
        auxiliary = {}
        if opcode == "RMS_NORM":
            rows = [first._rms_lookup_row(row, params[0].tolist(), attributes["epsilon"], providers.rsqrt, runtime) for row in inputs[0].reshape(-1, inputs[0].shape[-1]).tolist()]
            values = [np.asarray([row["output_bits"] for row in rows], dtype=np.uint16).reshape(inputs[0].shape)]
            scalar_stages[name] = {stage: [row[stage] for row in rows] for stage in first.STAGES}
            auxiliary[name] = scalar_stages[name]
        elif opcode == "LINEAR":
            mode = _linear_mode(node)
            values = [first._project_rows(inputs[0][0], params[0], mode, profiles[mode], workers)[None, ...]]
        elif opcode == "RESHAPE_TRANSPOSE_HEADS":
            values = [inputs[0].reshape(1, 30, attributes["heads"], 256).transpose(0, 2, 1, 3).copy()]
        elif opcode == "ROTARY_APPLY_PAIR":
            values = [first.rotate_bfloat16_bits(value, inputs[2], inputs[3]) for value in inputs[:2]]
        elif opcode == "REPEAT_KV":
            values = [np.repeat(inputs[0], attributes["repetitions"], axis=1)]
        elif opcode == "MATMUL_QK":
            values = [np.stack([first._project_rows(inputs[0][0, head], inputs[1][0, head], "serial", profiles["serial"], workers) for head in range(4)])[None, ...]]
        elif opcode == "MATMUL_AV":
            values = [np.stack([first._project_rows(np.pad(inputs[0][0, head], ((0, 0), (0, 2))), np.pad(inputs[1][0, head].T, ((0, 0), (0, 2))), "serial", profiles["serial"], workers) for head in range(4)])[None, ...]]
        elif opcode == "TRANSPOSE_RESHAPE_HEADS":
            values = [first.concatenate_heads(inputs[0]).copy()]
        elif opcode == "SCALE":
            values = [np.asarray([first.bfloat16_multiply_bits(int(value), 0x3D80) for value in inputs[0].reshape(-1)], dtype=np.uint16).reshape(inputs[0].shape)]
        elif opcode == "ADD" and node["inputs"][1] == "mask.sliding":
            values = [first.mask_score_bits(*inputs)]
        elif opcode in ("ADD", "MUL"):
            primitive = first.bfloat16_add_bits if opcode == "ADD" else first.bfloat16_multiply_bits
            if inputs[0].shape != inputs[1].shape:
                raise ValueError("Layer-one elementwise shape mismatch")
            values = [np.asarray([primitive(int(left), int(right)) for left, right in zip(inputs[0].reshape(-1), inputs[1].reshape(-1))], dtype=np.uint16).reshape(inputs[0].shape)]
        elif opcode == "SOFTMAX":
            rows = [first.lookup_softmax_row([int(value) << 16 for value in row], providers.exp, runtime) for row in inputs[0].reshape(-1, 30)]
            values = [np.asarray([row["output_bf16_bits"] for row in rows], dtype=np.uint16).reshape(1, 4, 30, 30)]
            softmax_f32 = np.asarray([row["output_f32_bits"] for row in rows], dtype=np.uint32).reshape(1, 4, 30, 30).tolist()
            auxiliary["softmax_f32_bits"] = softmax_f32
        elif opcode == "GELU_TANH":
            values = [np.asarray([providers.gelu.predict_bits(int(value), runtime) for value in inputs[0].reshape(-1)], dtype=np.uint16).reshape(inputs[0].shape)]
        else:
            raise ValueError("Unregistered second-layer instruction")
        if len(values) != len(node["outputs"]):
            raise ValueError("Missing layer-one multi-output state")
        for output, value in zip(node["outputs"], values):
            if output in states:
                raise ValueError("Layer-one state overwrite")
            states[output] = first._array(value, first._shape(program, output), "torch.bfloat16").copy()
        record = append_chain_record(records, _payload(node, states, hashes, parameter_snapshots, providers, profiles, auxiliary))
        hashes.update({output: record["record_hash"] for output in node["outputs"]})
    result = {"state_bits": {name: value.tolist() for name, value in states.items()}, "scalar_stages": scalar_stages, "softmax_f32_bits": softmax_f32, "records": records}
    check_execution(program, root_bits, parameter_snapshots, providers, profiles, runtime, result)
    if code != _code_sha():
        raise ValueError("Second-layer arithmetic source changed during execution")
    return result


def check_execution(program: dict[str, Any], root_bits: dict[str, Any], snapshots: dict[str, Any], providers: first.Providers, profiles: dict[str, Any], runtime: dict[str, Any], execution: dict[str, Any]) -> None:
    nodes = second_layer_instructions(program)
    check_parameter_snapshots(program, snapshots)
    first._profiles(profiles)
    providers.validate(runtime)
    states = _roots(program, root_bits, providers)
    if set(execution) != {"state_bits", "scalar_stages", "softmax_f32_bits", "records"} or set(execution["state_bits"]) != set(ROOTS) | {name for node in nodes for name in node["outputs"]}:
        raise ValueError("Second-layer 30 new states plus four roots coverage mismatch")
    for name in ROOTS:
        if not np.array_equal(states[name], first._array(execution["state_bits"][name], first._shape(program, name), "torch.bfloat16")):
            raise ValueError("Stored execution root changed")
    rms = [node for node in nodes if node["opcode"] == "RMS_NORM"]
    if set(execution["scalar_stages"]) != {node["outputs"][0] for node in rms}:
        raise ValueError("Layer-one RMS auxiliary coverage mismatch")
    for node in rms:
        name = node["outputs"][0]
        stages = execution["scalar_stages"][name]
        if set(stages) != set(first.STAGES):
            raise ValueError("Layer-one RMS auxiliary keys mismatch")
        for stage in first.STAGES:
            first._array(stages[stage], [int(np.prod(first._shape(program, name)[:-1]))], "torch.float32")
    first._array(execution["softmax_f32_bits"], [1, 4, 30, 30], "torch.float32")
    records = execution["records"]
    if len(records) != 29 or not verify_trace_chain(records)["valid"]:
        raise ValueError("Missing or invalid layer-one trace chain")
    hashes = {name: "ROOT:" + first._descriptor(value)["sha256"] for name, value in states.items()}
    expected_records = []
    for node, record in zip(nodes, records):
        auxiliary = {}
        for name in node["outputs"]:
            states[name] = first._array(execution["state_bits"][name], first._shape(program, name), "torch.bfloat16")
        if node["opcode"] == "RMS_NORM":
            auxiliary[node["outputs"][0]] = execution["scalar_stages"][node["outputs"][0]]
        if node["opcode"] == "SOFTMAX":
            auxiliary["softmax_f32_bits"] = execution["softmax_f32_bits"]
        expected = append_chain_record(expected_records, _payload(node, states, hashes, snapshots, providers, profiles, auxiliary))
        if canonical_json(expected) != canonical_json(record):
            raise ValueError("Layer-one ledger producer/input/weight/provider/auxiliary/output mismatch")
        hashes.update({name: record["record_hash"] for name in node["outputs"]})


def _coverage(program: dict[str, Any]) -> dict[str, Any]:
    nodes = second_layer_instructions(program)
    names = [*ROOTS, *(name for node in nodes for name in node["outputs"])]
    return {"target": "hidden.2", "root_inputs": list(ROOTS), "instruction_ids": [node["id"] for node in nodes], "instruction_count": 29,
            "states": {name: {"shape": first._shape(program, name), "dtype": "torch.bfloat16"} for name in names},
            "state_order": names, "new_state_count": 30, "root_state_count": 4, "state_count": 34, "parameter_count": 13,
            "rms_scalar_positions": 810, "softmax_fp32_positions": 3600, "hidden_2_value_count": 19200,
            "layer_index": 1, "attention_type": "sliding_attention", "rotary_profile": "local", "sliding_window": 512, "first_full_attention_index": 5}


def _native_kernel_roles(program: dict[str, Any]) -> list[str]:
    return sorted("softmax" if node["opcode"] == "SOFTMAX" else node["outputs"][0] for node in second_layer_instructions(program)
                  if node["opcode"] in ("RMS_NORM", "LINEAR", "GELU_TANH", "MATMUL_QK", "MATMUL_AV", "ADD", "MUL", "SOFTMAX", "SCALE"))


def _kernel_transfer(program: dict[str, Any], baseline: holdout.BaselineSources) -> dict[str, Any]:
    mapping = {name: "hidden.1" if name == "hidden.2" else "softmax" if name == "softmax" else "layer.0." + name[len("layer.1."):] for name in _native_kernel_roles(program)}
    prior = baseline.baseline_report["observations"][0]["traced"]["kernels"]
    if len(mapping) != 22 or set(mapping.values()) != set(first._native_kernel_roles(program)) or set(prior) != set(mapping.values()):
        raise ValueError("All twenty-two baseline symbol roles are required")
    return {"role_mapping_for_symbol_sets_only": mapping, "expected_symbols": {name: copy.deepcopy(prior[old]) for name, old in mapping.items()},
            "shape_transfer_prequalified": False, "meaning": "Unchanged arithmetic transfer hypothesis; mapping applies only to expected distinct native symbol sets, never IR, weights or states."}


@dataclass(frozen=True)
class SecondLayerSources:
    baseline: holdout.BaselineSources
    protocol: dict[str, Any]
    holdout_plan: dict[str, Any]
    holdout_bundle: dict[str, Any]
    holdout_report: dict[str, Any]

    @property
    def program(self) -> dict[str, Any]:
        return self.baseline.first_sources.program

    @property
    def runtime(self) -> dict[str, Any]:
        return self.baseline.first_sources.runtime

    def commitments(self) -> dict[str, Any]:
        return {"baseline": self.baseline.commitments(), "holdout_protocol_sha256": _sha(self.protocol), "holdout_plan_sha256": _sha(self.holdout_plan),
                "holdout_bundle_sha256": _sha(self.holdout_bundle), "holdout_report_sha256": _sha(self.holdout_report), "program_payload_sha256": _sha(self.program)}

    def validate(self, model_path: Path) -> tuple[Any, ...]:
        code, guard = _code_sha(), _sha(self.commitments())
        second_layer_instructions(self.program)
        for payload, field, digest in ((self.protocol, "protocol_sha256", HOLDOUT_PROTOCOL_SHA256), (self.holdout_plan, "plan_sha256", HOLDOUT_PLAN_SHA256), (self.holdout_report, "report_sha256", HOLDOUT_REPORT_SHA256)):
            _check_hash(payload, field)
            if payload[field] != digest:
                raise ValueError("Second-layer requires the declared passing first-layer holdout evidence")
        cached = holdout._CachedFirstLayerSources(self.baseline)
        checked = holdout.verify_holdout(cached, self.protocol, self.holdout_plan, self.holdout_bundle, self.holdout_report, model_path)
        if any(checked.get(name) is not True for name in ("valid", "holdout_matches", "prediction_complete", "connected_first_layer_independently_recomputed")):
            raise ValueError("Second-layer requires passing full both-case holdout source validation")
        report = self.holdout_report
        if report.get("native_coverage_complete") is not True or report.get("original_forward_count") != 12 or report.get("required_case_ids") != list(holdout.CASE_IDS) or [case["case_id"] for case in report["cases"]] != list(holdout.CASE_IDS) or any(case.get("case_matches") is not True or case.get("native_coverage_complete") is not True for case in report["cases"]):
            raise ValueError("Omitted or unmatched prerequisite holdout case")
        ids, providers, profiles = cached.validate()
        frequency = self.baseline.baseline_bundle["parameter_snapshots"][first.FREQUENCY]
        providers.validate(self.runtime, first._array(frequency["bits"], [128], "torch.float32"))
        first._profiles(profiles)
        baseline_execution = self.baseline.baseline_bundle["execution"]
        positions = first._array(baseline_execution["state_bits"]["position_ids"], [1, 30], "torch.int64")
        if positions.tolist() != [list(range(30))] or ids != self.baseline.baseline_plan["input_token_ids"] or first._descriptor(first._tokens(self.program, ids)) != self.baseline.baseline_plan["input_ids"]:
            raise ValueError("Stored roots require the exact baseline token IDs and positions 0..29")
        roots = _roots(self.program, {name: baseline_execution["state_bits"][name] for name in ROOTS}, providers)
        bindings = {}
        for name in ROOTS:
            producers = [record for record in baseline_execution["records"] if name in record["payload"]["outputs"]]
            if len(producers) != 1:
                raise ValueError("Missing unique verified baseline root producer")
            producer, descriptor = producers[0], first._descriptor(roots[name])
            if producer["payload"]["outputs"][name] != descriptor or self.baseline.baseline_plan["state_descriptors"][name] != descriptor or producer["payload"]["instruction_id"] != self.program["tensors"][name]["producer"]:
                raise ValueError("Stored root descriptor/source producer mismatch")
            bindings[name] = {"descriptor": descriptor, "source_producer_record_hash": producer["record_hash"], "source_instruction_id": producer["payload"]["instruction_id"], "execution_root_producer": "ROOT:" + descriptor["sha256"]}
        _kernel_transfer(self.program, self.baseline)
        if code != _code_sha() or guard != _sha(self.commitments()):
            raise ValueError("Second-layer source changed during validation")
        return ids, providers, profiles, {name: value.tolist() for name, value in roots.items()}, bindings


def _plan_body(sources: SecondLayerSources, context: tuple[Any, ...], bundle: dict[str, Any]) -> dict[str, Any]:
    from . import gemma_second_layer_capture as capture
    ids, providers, profiles, roots, bindings = context
    execution = bundle["execution"]
    return {"schema_version": 1, "artifact_kind": "gemma_second_layer_plan", "scope": SCOPE, "program_sha256": sources.program["program_sha256"],
            "coverage": _coverage(sources.program), "sources": sources.commitments(), "code_sha256": _code_sha(), "capture_code_sha256": capture._code_sha(), "native_model_implementation": _implementation_commitment(),
            "runtime": copy.deepcopy(sources.runtime), "model_binding": copy.deepcopy(sources.baseline.baseline_plan["model_binding"]),
            "input_token_ids": copy.deepcopy(ids), "input_ids": first._descriptor(first._tokens(sources.program, ids)),
            "providers": providers.commitments(), "profiles": copy.deepcopy(profiles), "source_root_bindings": bindings, "root_bits_sha256": _sha(roots),
            "kernel_transfer": _kernel_transfer(sources.program, sources.baseline), "native_kernel_roles": _native_kernel_roles(sources.program),
            "parameter_snapshots": {name: {key: value for key, value in snapshot.items() if key != "bits"} for name, snapshot in bundle["parameter_snapshots"].items()},
            "state_descriptors": {name: first._descriptor(first._array(value, first._shape(sources.program, name), "torch.bfloat16")) for name, value in execution["state_bits"].items()},
            "bundle_sha256": _sha(bundle), "trace_root": execution["records"][-1]["record_hash"], "repetitions": 3, "original_forward_count": 6,
            "epoch_scope": "previously_observed_baseline_case_with_prerequisite_first_layer_holdouts_not_new_layer_one_cases",
            "parameter_snapshot_membership": "all_thirteen_full_actual_layer_one_tensors_checkpoint_hash_checked_even_without_model",
            "layer_one_independently_recomputed": False, **_flags()}


def _predict(sources: SecondLayerSources, model: Any, context: tuple[Any, ...], workers: int) -> tuple[dict[str, Any], dict[str, Any]]:
    code, guard = _code_sha(), _sha(sources.commitments())
    _, providers, profiles, roots, _ = context
    snapshots = _snapshot_model(sources, model)
    execution = execute_second_layer(sources.program, roots, snapshots, providers, profiles, sources.runtime, workers)
    if canonical_json(_snapshot_model(sources, model)) != canonical_json(snapshots):
        raise ValueError("Actual layer-one checkpoint changed across forecast")
    if code != _code_sha() or guard != _sha(sources.commitments()):
        raise ValueError("Second-layer sources changed across forecast")
    bundle = {"parameter_snapshots": snapshots, "execution": execution}
    return _seal(_plan_body(sources, context, bundle), "plan_sha256"), bundle


def build_second_layer_plan(sources: SecondLayerSources, model: Any, model_path: Path, workers: int = 4) -> tuple[dict[str, Any], dict[str, Any]]:
    return _predict(sources, model, sources.validate(model_path), workers)


def _check_plan(sources: SecondLayerSources, plan: dict[str, Any], bundle: dict[str, Any], context: tuple[Any, ...]) -> None:
    _check_hash(plan, "plan_sha256")
    if set(bundle) != {"parameter_snapshots", "execution"}:
        raise ValueError("Unexpected second-layer prediction material")
    _, providers, profiles, roots, _ = context
    check_execution(sources.program, roots, bundle["parameter_snapshots"], providers, profiles, sources.runtime, bundle["execution"])
    if canonical_json(plan) != canonical_json(_seal(_plan_body(sources, context, bundle), "plan_sha256")):
        raise ValueError("Second-layer plan/source/root/bundle binding mismatch")


def check_second_layer_plan(sources: SecondLayerSources, plan: dict[str, Any], bundle: dict[str, Any], model_path: Path) -> None:
    _check_plan(sources, plan, bundle, sources.validate(model_path))


def _native_comparison(plan: dict[str, Any], bundle: dict[str, Any], observations: list[dict[str, Any]]) -> dict[str, Any]:
    from . import gemma_second_layer_capture as capture
    if not isinstance(observations, list) or len(observations) != 3:
        raise ValueError("Exactly three traced/plain pairs are required")
    execution, coverage = bundle["execution"], plan["coverage"]
    comparisons, controls, kernel_sets = [], [], []
    checks = dict.fromkeys(("capture_checks", "geometry", "runtime", "capture_code", "baseline_kernel_symbols", "same_cuda_device"), True)
    for pair in observations:
        if set(pair) != {"traced", "untraced"}:
            raise ValueError("Native pair fields mismatch")
        traced, plain = pair["traced"], pair["untraced"]
        if any("capture_failure" in record for record in (traced, plain)):
            raise ValueError("Native capture abstained; incomplete evidence")
        if set(traced["state_bits"]) != set(coverage["states"]) or set(traced["geometry"]) != set(coverage["states"]) or set(plain["state_bits"]) != {"hidden.1", "hidden.2"} or set(plain["geometry"]) != {"hidden.1", "hidden.2"}:
            raise ValueError("Native root/output state or geometry coverage mismatch")
        if any(plain.get(key) for key in ("scalar_stages", "softmax_f32_bits", "kernels")):
            raise ValueError("Plain control requires only minimal hidden.1/hidden.2 hooks")
        if set(traced["scalar_stages"]) != set(execution["scalar_stages"]):
            raise ValueError("Native six RMS scalar coverage mismatch")
        result = {}
        for name in coverage["state_order"]:
            declaration = coverage["states"][name]
            shape, dtype = declaration["shape"], declaration["dtype"]
            result[name] = first._comparison(execution["state_bits"][name], traced["state_bits"][name], shape, dtype)
            checks["geometry"] &= first._geometry(traced["geometry"][name], shape, dtype)
            if name in execution["scalar_stages"]:
                stages, native = execution["scalar_stages"][name], traced["scalar_stages"][name]
                if set(native) != {*first.STAGES, "mean_input_metadata"}:
                    raise ValueError("Native RMS auxiliary stage fields mismatch")
                for stage in first.STAGES:
                    result[name + ":" + stage] = first._comparison(stages[stage], native[stage], [len(stages[stage])], "torch.float32")
                metadata = native["mean_input_metadata"]
                strides = metadata.get("input_strides")
                checks["geometry"] &= metadata.get("input_shape") == shape and metadata.get("input_dtype") == "torch.float32" and metadata.get("axes") == [-1] and metadata.get("keepdim") is True and metadata.get("alignment_mod16") == 0 and isinstance(strides, list) and len(strides) == len(shape) and strides[-1] == 1 and all(type(stride) is int and stride > 0 and stride % 4 == 0 for stride in strides[:-1])
            if name == "layer.1.attention.probability":
                result["softmax_f32_bits"] = first._comparison(execution["softmax_f32_bits"], traced["softmax_f32_bits"], [1, 4, 30, 30], "torch.float32")
        control = {}
        for name in ("hidden.1", "hidden.2"):
            control[name] = first._comparison(execution["state_bits"][name], plain["state_bits"][name], [1, 30, 640], "torch.bfloat16")
            checks["geometry"] &= first._geometry(plain["geometry"][name], [1, 30, 640], "torch.bfloat16")
        controls.append(control)
        checks["same_cuda_device"] &= len({value["device"] for record in (traced, plain) for value in record["geometry"].values()}) == 1
        for record in (traced, plain):
            required = {"token_commitment_unchanged", "stopped_at_decoder_tuple", "decoder_tuple_length_one", "no_later_layer", "no_final_norm", "no_lm_head", "all_needed_states_present", "code_unchanged", "runtime_unchanged", "target_layer_one", "local_rotary_and_sliding_mask"}
            required.update({"original_operations_once", "all_operand_links_unchanged", "rms_scalar_stages_complete", "softmax_f32_complete"} if record is traced else {"plain_minimal_control"})
            checks["capture_checks"] &= isinstance(record.get("checks"), dict) and required.issubset(record["checks"]) and all(value is True for value in record["checks"].values())
            checks["runtime"] &= canonical_json(record["runtime_before"]) == canonical_json(plan["runtime"]) == canonical_json(record["runtime_after"])
            checks["capture_code"] &= record["code_before"] == plan["capture_code_sha256"] == capture._code_sha() == record["code_after"]
        kernels = traced["kernels"]
        checks["baseline_kernel_symbols"] &= kernels == plan["kernel_transfer"]["expected_symbols"] and set(kernels) == set(plan["native_kernel_roles"]) and len(kernels) == 22 and all(isinstance(names, list) and names and names == sorted(set(names)) and all(isinstance(name, str) and name.strip() for name in names) for names in kernels.values())
        kernel_sets.append(kernels)
        comparisons.append(result)
    checks["repeat_stable_kernel_sets"] = all(value == kernel_sets[0] for value in kernel_sets)
    counts = {name: sum(item[name]["mismatch_count"] for item in comparisons) for name in comparisons[0]}
    counts.update({"plain_" + name: sum(item[name]["mismatch_count"] for item in controls) for name in ("hidden.1", "hidden.2")})
    divergence = next(({"repetition": index, "state": name, **item["first_divergence"]} for name in comparisons[0] for index, result in enumerate(comparisons) if (item := result[name])["first_divergence"] is not None), None)
    if divergence is None:
        divergence = next(({"repetition": index, "state": "plain_" + name, **item["first_divergence"]} for index, result in enumerate(controls) for name, item in result.items() if item["first_divergence"] is not None), None)
    if divergence is None:
        divergence = next(({"check": name} for name, passed in checks.items() if not passed), None)
    return {"comparisons": comparisons, "untraced_controls": controls, "checks": checks, "mismatch_counts": counts,
            "aggregate_mismatch_count": sum(counts.values()), "first_divergence": divergence,
            "source_root_comparison": [{name: comparison[name] for name in ROOTS} for comparison in comparisons],
            "kernel_trace": kernel_sets, "native_coverage_complete": True, "second_layer_matches": all(checks.values()) and not any(counts.values())}


def second_layer_report(plan: dict[str, Any], bundle: dict[str, Any], observations: list[dict[str, Any]], acquisition_guards: dict[str, bool] | None = None) -> dict[str, Any]:
    _check_hash(plan, "plan_sha256")
    if plan["scope"] != SCOPE or _sha(bundle) != plan["bundle_sha256"] or any(plan.get(key) is not value for key, value in _flags().items()):
        raise ValueError("Second-layer comparison requires frozen bounded predictions")
    if len(bundle["execution"]["records"]) != 29 or not verify_trace_chain(bundle["execution"]["records"], plan["trace_root"])["valid"]:
        raise ValueError("Second-layer report requires complete prediction ledger")
    guards = dict.fromkeys(("checkpoint_unchanged", "source_unchanged", "frozen_files_unchanged"), True) if acquisition_guards is None else acquisition_guards
    try:
        result = _native_comparison(plan, bundle, observations)
    except (KeyError, TypeError, ValueError, IndexError) as error:
        result = {"second_layer_matches": False, "native_coverage_complete": False, "mismatch_counts": {}, "aggregate_mismatch_count": 0,
                  "first_divergence": {"check": "native_coverage"}, "abstention": {"type": type(error).__name__, "message": str(error)}}
    guarded = set(guards) == {"checkpoint_unchanged", "source_unchanged", "frozen_files_unchanged"} and all(value is True for value in guards.values())
    matches = result["second_layer_matches"] and guarded
    if not guarded and result["first_divergence"] is None:
        result["first_divergence"] = {"check": "acquisition_guards"}
    return _seal({"schema_version": 1, "artifact_kind": "gemma_second_layer_report", "scope": SCOPE, "plan_sha256": plan["plan_sha256"], "bundle_sha256": _sha(bundle),
                  "coverage": plan["coverage"], "observations": observations, **result, "acquisition_guards": guards, "second_layer_matches": matches,
                  "layer_one_independently_recomputed": matches, "required_forward_count": 6,
                  "original_forward_count": sum(isinstance(record, dict) and "capture_failure" not in record for pair in observations for record in pair.values()),
                  "kernel_provenance": "22 distinct-symbol-set transfer tests, not launch order, instruction-level semantics or prequalified shape transfer", **_flags()}, "report_sha256")


def acquire_second_layer(sources: SecondLayerSources, model: Any, plan: dict[str, Any], bundle: dict[str, Any], model_path: Path, plan_path: Path, bundle_path: Path) -> dict[str, Any]:
    from . import gemma_second_layer_capture as capture
    for path, payload in ((plan_path, plan), (bundle_path, bundle)):
        holdout.require_frozen(path, payload)
    context = sources.validate(model_path)
    _check_plan(sources, plan, bundle, context)
    code, guard = _code_sha(), _sha(sources.commitments())
    if canonical_json(_snapshot_model(sources, model)) != canonical_json(bundle["parameter_snapshots"]):
        raise ValueError("Fresh actual layer-one acquisition weights differ from frozen predictions")
    observations = []
    for _ in range(3):
        pair = {}
        for name, traced in (("traced", True), ("untraced", False)):
            try:
                pair[name] = capture.capture_second_layer(model, context[0], traced)
            except ValueError as error:
                pair[name] = {"capture_failure": {"type": type(error).__name__, "message": str(error), "kind": "capture_abstention_not_numerical_evidence"}}
        observations.append(pair)
    guards = {}
    try:
        guards["checkpoint_unchanged"] = canonical_json(_snapshot_model(sources, model)) == canonical_json(bundle["parameter_snapshots"])
    except (ValueError, RuntimeError):
        guards["checkpoint_unchanged"] = False
    try:
        guards["source_unchanged"] = code == _code_sha() and guard == _sha(sources.commitments())
    except ValueError:
        guards["source_unchanged"] = False
    try:
        for path, payload in ((plan_path, plan), (bundle_path, bundle)):
            holdout.require_frozen(path, payload)
        guards["frozen_files_unchanged"] = True
    except ValueError:
        guards["frozen_files_unchanged"] = False
    return second_layer_report(plan, bundle, observations, guards)


def verify_second_layer(sources: SecondLayerSources, plan: dict[str, Any], bundle: dict[str, Any], report: dict[str, Any], model_path: Path) -> dict[str, Any]:
    result = {"mode": "integrity_full_source_bindings_snapshot_hashchain_and_native_recount_only", "actual_layer1_numerical_recompute": False,
              "layer_one_numerical_recomputation_performed": False, "previous_lineage_validation": "full_source_integrity_and_short_RMS_recomputation_not_previous_matmuls", **_flags()}
    try:
        code, guard = _code_sha(), _sha(sources.commitments())
        check_second_layer_plan(sources, plan, bundle, model_path)
        _check_hash(report, "report_sha256")
        expected = second_layer_report(plan, bundle, report["observations"], report["acquisition_guards"])
        valid = canonical_json(report) == canonical_json(expected) and code == _code_sha() and guard == _sha(sources.commitments())
        return {**result, "valid": valid, "second_layer_matches": valid and expected["second_layer_matches"], "layer_one_independently_recomputed": valid and expected["second_layer_matches"], "mismatch_counts": expected["mismatch_counts"], "first_divergence": expected["first_divergence"]}
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError, IndexError) as error:
        return {**result, "valid": False, "second_layer_matches": False, "layer_one_independently_recomputed": False, "reason": str(error)}


def reexecute_second_layer(sources: SecondLayerSources, model: Any, plan: dict[str, Any], bundle: dict[str, Any], report: dict[str, Any], model_path: Path, plan_path: Path, bundle_path: Path, workers: int = 4) -> dict[str, Any]:
    checked = verify_second_layer(sources, plan, bundle, report, model_path)
    if not checked["valid"]:
        return checked
    regenerated_plan, regenerated_bundle = build_second_layer_plan(sources, model, model_path, workers)
    same = canonical_json(regenerated_plan) == canonical_json(plan) and canonical_json(regenerated_bundle) == canonical_json(bundle)
    replay = acquire_second_layer(sources, model, plan, bundle, model_path, plan_path, bundle_path) if same else None
    exact = same and canonical_json(replay) == canonical_json(report)
    return {"valid": exact, "mode": "fresh_actual_thirteen_layer_one_weights_and_all_29_nodes_before_six_native_forwards",
            "predictions_recomputed_exact": same, "reexecution_exact": exact, "actual_layer1_numerical_recompute": True, "layer_one_numerical_recomputation_performed": True,
            "second_layer_matches": exact and replay["second_layer_matches"], "layer_one_independently_recomputed": exact and replay["second_layer_matches"], **_flags()}


def second_layer_summary(plan: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in report.items() if key not in ("observations", "comparisons", "untraced_controls", "report_sha256")}
    body.update(source_report_sha256=report["report_sha256"], sources=plan["sources"], source_root_bindings=plan["source_root_bindings"], code_sha256=plan["code_sha256"], trace_root=plan["trace_root"], providers=plan["providers"], profiles=plan["profiles"])
    return _seal(body, "summary_sha256")
