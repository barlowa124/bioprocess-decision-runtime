from __future__ import annotations

import copy
import hashlib
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from . import gemma_first_layer as first
from . import gemma_second_layer as second
from . import gemma_two_layers_holdout as holdout
from .gemma_rotary_slice import _sha, _seal, _check_hash
from .operational_semantics import append_chain_record, verify_trace_chain
from .gemma_mlp_down import _kernel_symbols
from .reference_gemma import _observe_rms_module
from .serialization import canonical_json

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
PROTOCOL_SHA256 = "91b708325ae4e7ebe08251c7a66283380f9e1dc22410c1fa6d5936c4823617b7"
PLAN_SHA256 = "23424f5e22df5ec12b3a680c25a589983301593309cbedda41ffb2724300c7aa"
REPORT_SHA256 = "e61317b19a462af21c0cb0bf471b883e97231be922811579b28720fa5d87303d"
ROOT = "hidden.2"
PREFIX = "layer.2."
V_HEADS = PREFIX + "value.heads"
SCOPE = "Fixed baseline case, actual decoder index 2 attention entry i0065..i0073 from a reused verified hidden.2 boundary. Fresh six layer-2 parameter tensors; unchanged independent RMS/rsqrt and Q/K/V reduction profiles tested as transfer hypotheses. Nine new states and 540 RMS scalar positions; V heads are a coordinate reinterpretation of native v_proj output, not a separately intercepted original head-view operation. Stop after k_norm before target-layer RoPE/scores/MLP. Not fresh connected three-layer execution, a layer-2 holdout, all FP32 internals, hardware or unrestricted qualification."
FALSE_FLAGS = ("qualified", "full_model_qualified", "hardware_semantics_established", "global_exactness_activation_allowed", "qualification_promotion_allowed", "shape_transfer_prequalified", "candidate_refitting_allowed", "connected_three_layers_independently_recomputed", "prefix_independently_recomputed", "internal_layer_two_intermediates_reused", "fresh_prompt_holdout", "fresh_layer_two_holdout", "target_rotary_executed", "target_attention_scores_executed", "target_mlp_executed", "later_layers_executed", "final_model_norm_executed", "logits_executed", "all_fp32_vector_internals_observed", "native_arithmetic_reconstructed")
V_MAPPING = "coordinate reinterpretation of captured original v_proj output; reshape(1,30,1,256), transpose(0,2,1,3); no floating arithmetic"


def _flags() -> dict[str, bool]:
    return {**dict.fromkeys(FALSE_FLAGS, False), "prefix_boundary_reused": True, "stored_boundary_predictions_used": True, "empirical_primitive_data_reused": True}


def _code_sha() -> str:
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Layer-2 entry source changed after import")
    return _sha({"module": SOURCE_SHA256, "first": first._code_sha(), "second": second._code_sha(), "holdout": holdout._code_sha()})


def instructions(program: dict[str, Any]) -> list[dict[str, Any]]:
    if program.get("program_sha256") != first.PROGRAM_SHA256 or not first.verify_gemma_ir(program)["valid"]:
        raise ValueError("Layer-2 entry requires the exact checkpoint IR")
    if program["configuration"]["layer_types"][:6] != ["sliding_attention"] * 5 + ["full_attention"]:
        raise ValueError("Unexpected layer-2 rotary/mask regime")
    nodes = [node for node in program["instructions"] if node["id"] in {f"i{i:04d}" for i in range(65, 74)}]
    available = {ROOT}
    if [node["id"] for node in nodes] != [f"i{i:04d}" for i in range(65, 74)]:
        raise ValueError("Layer-2 entry order/coverage mismatch")
    for node in nodes:
        if type(node["layer"]) is not int or node["layer"] != 2 or len(node["outputs"]) != 1 or node["opcode"] not in ("RMS_NORM", "LINEAR", "RESHAPE_TRANSPOSE_HEADS") or any(name not in available for name in node["inputs"]) or available.intersection(node["outputs"]):
            raise ValueError("Invalid layer-2 entry dependency")
        available.update(node["outputs"])
        if any(not name.startswith("model.layers.2.") for name in node["parameter_refs"]):
            raise ValueError("Layer-2 entry must use actual layer-2 parameters")
    if len(available) != 10 or len({name for node in nodes for name in node["parameter_refs"]}) != 6:
        raise ValueError("Layer-2 entry state/parameter coverage mismatch")
    return nodes


def _mode(node: dict[str, Any]) -> str:
    return "serial" if node["outputs"] == [PREFIX + "query.flat"] else "key_value"


def _coverage(program: dict[str, Any]) -> dict[str, Any]:
    nodes = instructions(program)
    names = [ROOT, *(node["outputs"][0] for node in nodes)]
    return {"layer_index": 2, "instruction_ids": [node["id"] for node in nodes], "instruction_count": 9, "new_state_count": 9,
            "states": {name: {"shape": first._shape(program, name), "dtype": "torch.bfloat16"} for name in names},
            "state_order": names, "root_inputs": [ROOT], "parameter_count": 6, "rms_scalar_positions": 540,
            "target_names": [PREFIX + "query.normalized", PREFIX + "key.normalized", V_HEADS], "target_value_count": 46080,
            "attention_type": "sliding_attention", "rotary_profile": "local", "sliding_window": 512, "value_heads_mapping": V_MAPPING}


def check_snapshots(program: dict[str, Any], snapshots: dict[str, Any]) -> dict[str, np.ndarray]:
    names = {name for node in instructions(program) for name in node["parameter_refs"]}
    if set(snapshots) != names:
        raise ValueError("Exactly six layer-2 snapshots required")
    values = {}
    for name, snapshot in snapshots.items():
        commitment = program["parameter_commitments"][name]
        if set(snapshot) != {"bits", "descriptor", "commitment"} or snapshot["commitment"] != commitment or commitment["dtype"] != "torch.bfloat16":
            raise ValueError("Layer-2 snapshot commitment mismatch")
        value = first._array(snapshot["bits"], commitment["shape"], "torch.bfloat16")
        if first._descriptor(value) != snapshot["descriptor"] or snapshot["descriptor"]["sha256"] != commitment["sha256"]:
            raise ValueError("Layer-2 snapshot does not match checkpoint")
        values[name] = value
    return values


def _payload(node, states, hashes, snapshots, providers, profiles, auxiliary):
    provider = {"name": node["opcode"]}
    if node["opcode"] == "LINEAR":
        mode = _mode(node)
        provider.update(mode=mode, profile=profiles[mode])
    elif node["opcode"] == "RMS_NORM":
        provider["rsqrt_evidence"] = providers.rsqrt.evidence
    return {"instruction_id": node["id"], "instruction_sha256": node["instruction_sha256"], "opcode": node["opcode"],
            "inputs": {name: {"descriptor": first._descriptor(states[name]), "producer_record_hash": hashes[name]} for name in node["inputs"]},
            "parameters": {name: {key: value for key, value in snapshots[name].items() if key != "bits"} for name in node["parameter_refs"]},
            "provider": provider, "outputs": {name: first._descriptor(states[name]) for name in node["outputs"]}, "auxiliary_sha256": _sha(auxiliary)}


def execute_entry(program, hidden_bits, snapshots, providers, profiles, runtime, workers=4):
    code = _code_sha()
    if type(providers) is not first.Providers or type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Invalid layer-2 entry provider/workers")
    first._profiles(profiles)
    providers.validate(runtime)
    parameters = check_snapshots(program, snapshots)
    states = {ROOT: first._array(hidden_bits, [1, 30, 640], "torch.bfloat16").copy()}
    hashes, records, stages = {ROOT: "ROOT:" + first._descriptor(states[ROOT])["sha256"]}, [], {}
    for node in instructions(program):
        name, opcode = node["outputs"][0], node["opcode"]
        value, auxiliary = states[node["inputs"][0]], {}
        if opcode == "RMS_NORM":
            weight = parameters[node["parameter_refs"][0]].tolist()
            rows = [first._rms_lookup_row(row, weight, node["attributes"]["epsilon"], providers.rsqrt, runtime) for row in value.reshape(-1, value.shape[-1]).tolist()]
            output = np.asarray([row["output_bits"] for row in rows], dtype=np.uint16).reshape(value.shape)
            stages[name] = {key: [row[key] for row in rows] for key in first.STAGES}
            auxiliary = stages[name]
        elif opcode == "LINEAR":
            mode = _mode(node)
            output = first._project_rows(value[0], parameters[node["parameter_refs"][0]], mode, profiles[mode], workers)[None, ...]
        else:
            output = value.reshape(1, 30, node["attributes"]["heads"], 256).transpose(0, 2, 1, 3).copy()
        states[name] = first._array(output, first._shape(program, name), "torch.bfloat16")
        record = append_chain_record(records, _payload(node, states, hashes, snapshots, providers, profiles, auxiliary))
        hashes[name] = record["record_hash"]
    result = {"state_bits": {name: value.tolist() for name, value in states.items()}, "scalar_stages": stages, "records": records}
    check_execution(program, hidden_bits, snapshots, providers, profiles, runtime, result)
    if code != _code_sha():
        raise ValueError("Layer-2 entry source changed during prediction")
    return result


def check_execution(program, hidden_bits, snapshots, providers, profiles, runtime, execution):
    check_snapshots(program, snapshots)
    first._profiles(profiles)
    providers.validate(runtime)
    nodes = instructions(program)
    if set(execution) != {"state_bits", "scalar_stages", "records"} or set(execution["state_bits"]) != {ROOT, *(node["outputs"][0] for node in nodes)}:
        raise ValueError("Entry execution coverage mismatch")
    states = {ROOT: first._array(hidden_bits, [1, 30, 640], "torch.bfloat16")}
    if not np.array_equal(states[ROOT], first._array(execution["state_bits"][ROOT], [1, 30, 640], "torch.bfloat16")):
        raise ValueError("Stored entry root changed")
    rms = {node["outputs"][0] for node in nodes if node["opcode"] == "RMS_NORM"}
    if set(execution["scalar_stages"]) != rms or len(execution["records"]) != 9 or not verify_trace_chain(execution["records"])["valid"]:
        raise ValueError("Entry RMS/ledger coverage mismatch")
    hashes, expected = {ROOT: "ROOT:" + first._descriptor(states[ROOT])["sha256"]}, []
    for node, record in zip(nodes, execution["records"]):
        name = node["outputs"][0]
        states[name] = first._array(execution["state_bits"][name], first._shape(program, name), "torch.bfloat16")
        auxiliary = execution["scalar_stages"].get(name, {})
        if name in rms:
            if set(auxiliary) != set(first.STAGES):
                raise ValueError("Entry scalar-stage fields mismatch")
            count = int(np.prod(states[name].shape[:-1]))
            for bits in auxiliary.values():
                first._array(bits, [count], "torch.float32")
        expected_record = append_chain_record(expected, _payload(node, states, hashes, snapshots, providers, profiles, auxiliary))
        if canonical_json(record) != canonical_json(expected_record):
            raise ValueError("Entry producer/input/weight/provider/output ledger mismatch")
        hashes[name] = expected_record["record_hash"]


@dataclass(frozen=True)
class EntrySources:
    baseline: holdout.BaselineSources
    protocol: dict[str, Any]
    holdout_plan: dict[str, Any]
    holdout_bundle: dict[str, Any]
    holdout_report: dict[str, Any]

    @property
    def program(self):
        return self.baseline.program

    @property
    def runtime(self):
        return self.baseline.runtime

    def commitments(self):
        return {"baseline": self.baseline.commitments(), **{name + "_payload_sha256": _sha(getattr(self, name)) for name in ("protocol", "holdout_plan", "holdout_bundle", "holdout_report")}}

    def validate(self, model_path):
        code, guard = _code_sha(), _sha(self.commitments())
        for payload, field, digest in ((self.protocol, "protocol_sha256", PROTOCOL_SHA256), (self.holdout_plan, "plan_sha256", PLAN_SHA256), (self.holdout_report, "report_sha256", REPORT_SHA256)):
            _check_hash(payload, field)
            if payload[field] != digest:
                raise ValueError("Layer-2 entry requires declared two-layer holdout evidence")
        cached = holdout._CachedTwoSources(self.baseline)
        checked = holdout.verify_holdout(cached, self.protocol, self.holdout_plan, self.holdout_bundle, self.holdout_report, model_path)
        if not checked.get("valid") or not checked.get("two_layer_holdout_matches") or not checked.get("native_coverage_complete"):
            raise ValueError("Both connected two-layer holdouts must pass completely")
        ids, providers, profiles = cached.validate(model_path)
        baseline = self.baseline.two_layer_bundle["execution"]
        hidden = first._array(baseline["state_bits"][ROOT], [1, 30, 640], "torch.bfloat16")
        producers = [record for record in baseline["records"] if ROOT in record["payload"]["outputs"]]
        if len(producers) != 1 or producers[0]["payload"]["instruction_id"] != self.program["tensors"][ROOT]["producer"] or producers[0]["payload"]["outputs"][ROOT] != first._descriptor(hidden):
            raise ValueError("Entry boundary lacks its verified global producer")
        kernel_sets = self.baseline.two_sources.second_report["kernel_trace"]
        if len(kernel_sets) != 3 or any(value != kernel_sets[0] for value in kernel_sets):
            raise ValueError("Prior layer kernel roles are unstable")
        kernels = {node["outputs"][0]: kernel_sets[0][node["outputs"][0].replace(PREFIX, "layer.1.", 1)] for node in instructions(self.program) if node["opcode"] in ("RMS_NORM", "LINEAR")}
        if code != _code_sha() or guard != _sha(self.commitments()):
            raise ValueError("Entry sources changed during validation")
        return ids, providers, profiles, hidden.tolist(), {"descriptor": first._descriptor(hidden), "producer_record_hash": producers[0]["record_hash"], "instruction_id": producers[0]["payload"]["instruction_id"]}, kernels


def _modules(model):
    layer = model.model.layers[2]
    attention = layer.self_attn
    return {PREFIX + "attention.normalized": layer.input_layernorm, **{PREFIX + role + ".flat": getattr(attention, short + "_proj") for role, short in (("query", "q"), ("key", "k"), ("value", "v"))}, PREFIX + "query.normalized": attention.q_norm, PREFIX + "key.normalized": attention.k_norm}


def _snapshot_model(sources, model):
    import torch
    from transformers.models.gemma3 import modeling_gemma3 as gemma
    second._model_context(sources.baseline.two_sources.second_sources, model)
    parameters = first.bind_model_tensors(sources.program, model, verify_hashes=True)
    layer = model.model.layers[2]
    second._original(layer, gemma.Gemma3DecoderLayer)
    second._original(layer.self_attn, gemma.Gemma3Attention)
    if layer.layer_idx != 2 or layer.attention_type != "sliding_attention" or not layer.self_attn.is_sliding or layer.self_attn.layer_idx != 2 or layer.self_attn.sliding_window != 512 or model.config.layer_types[2] != "sliding_attention":
        raise ValueError("Wrong actual decoder-index-2 regime")
    snapshots = {}
    for node in instructions(sources.program):
        if not node["parameter_refs"]:
            continue
        module = _modules(model)[node["outputs"][0]]
        second._original(module, torch.nn.Linear if node["opcode"] == "LINEAR" else gemma.Gemma3RMSNorm)
        name = node["parameter_refs"][0]
        weight = parameters[name]
        if module.weight is not weight or weight.device != model.model.embed_tokens.weight.device or not weight.is_contiguous():
            raise ValueError("Layer-2 parameter identity/layout mismatch")
        if node["opcode"] == "LINEAR":
            if module.bias is not None or [module.out_features, module.in_features] != list(weight.shape):
                raise ValueError("Unexpected layer-2 projection dimensions/bias")
        elif module.eps != node["attributes"]["epsilon"]:
            raise ValueError("Layer-2 RMS epsilon differs from IR")
        bits = first._bits(weight)
        snapshots[name] = {"bits": bits.tolist(), "descriptor": first._descriptor(bits), "commitment": copy.deepcopy(sources.program["parameter_commitments"][name])}
    check_snapshots(sources.program, snapshots)
    return snapshots


def _plan_body(sources, context, bundle):
    ids, providers, profiles, hidden, root_binding, kernels = context
    execution = bundle["execution"]
    return {"schema_version": 1, "scope": SCOPE, "sources": sources.commitments(), "code_sha256": _code_sha(), "coverage": _coverage(sources.program),
            "input_token_ids": ids, "input_ids": first._descriptor(first._tokens(sources.program, ids)), "runtime": sources.runtime,
            "root_binding": root_binding, "root_bits_sha256": _sha(hidden), "expected_kernel_sets": kernels, "profiles": profiles, "providers": providers.commitments(),
            "model_binding": sources.baseline.two_layer_plan["model_binding"], "parameter_snapshots": {name: {key: value for key, value in item.items() if key != "bits"} for name, item in bundle["parameter_snapshots"].items()},
            "state_descriptors": {name: first._descriptor(first._array(bits, first._shape(sources.program, name), "torch.bfloat16")) for name, bits in execution["state_bits"].items()},
            "bundle_sha256": _sha(bundle), "trace_root": execution["records"][-1]["record_hash"], "repetitions": 3, "required_forward_count": 6, **_flags()}


def build_entry_plan(sources, model, model_path, workers=4):
    code, guard = _code_sha(), _sha(sources.commitments())
    context = sources.validate(model_path)
    ids, providers, profiles, hidden, _, _ = context
    snapshots = _snapshot_model(sources, model)
    execution = execute_entry(sources.program, hidden, snapshots, providers, profiles, sources.runtime, workers)
    if canonical_json(_snapshot_model(sources, model)) != canonical_json(snapshots) or code != _code_sha() or guard != _sha(sources.commitments()):
        raise ValueError("Entry checkpoint or source changed during prediction")
    bundle = {"parameter_snapshots": snapshots, "execution": execution}
    return _seal(_plan_body(sources, context, bundle), "plan_sha256"), bundle


def check_entry_plan(sources, plan, bundle, model_path):
    code, guard = _code_sha(), _sha(sources.commitments())
    _check_hash(plan, "plan_sha256")
    context = sources.validate(model_path)
    if set(bundle) != {"parameter_snapshots", "execution"}:
        raise ValueError("Unexpected entry prediction payload")
    check_execution(sources.program, context[3], bundle["parameter_snapshots"], context[1], context[2], sources.runtime, bundle["execution"])
    if canonical_json(plan) != canonical_json(_seal(_plan_body(sources, context, bundle), "plan_sha256")) or code != _code_sha() or guard != _sha(sources.commitments()):
        raise ValueError("Entry plan/source/profile binding mismatch")
    return context


def capture_entry(model, token_ids, traced):
    import torch
    from torch.profiler import profile, ProfilerActivity
    layer = model.model.layers[2]
    modules = _modules(model)
    record = {"state_bits": {}, "geometry": {}, "scalar_stages": {}, "kernels": {}, "value_heads_mapping": V_MAPPING}
    handles, inputs, prefix_outputs = [], {}, {}
    code, runtime, original_ids = _code_sha(), first._runtime(), copy.deepcopy(token_ids)
    ids = torch.tensor(token_ids, dtype=torch.int64, device=layer.input_layernorm.weight.device)

    class Complete(Exception):
        pass

    def store(name, value):
        if name in record["state_bits"] or value.dtype != torch.bfloat16:
            raise ValueError("Repeated or mistyped entry boundary")
        shape = [1, 30, 640] if name in (ROOT, PREFIX + "attention.normalized") else [1, 30, 1024 if "query" in name else 256] if name.endswith(".flat") else [1, 4 if "query" in name else 1, 30, 256]
        if list(value.shape) != shape:
            raise ValueError("Unexpected entry boundary shape")
        record["state_bits"][name] = first._bits(value).tolist()
        record["geometry"][name] = {"shape": list(value.shape), "dtype": str(value.dtype), "strides": list(value.stride()), "device": str(value.device), "alignment_mod16": value.data_ptr() % 16}

    def prefix_output(module, args, output):
        if type(output) is not tuple or len(output) != 1 or prefix_outputs:
            raise ValueError("Invalid native prefix tuple/occurrence")
        prefix_outputs[ROOT] = output[0]

    def target_input(module, args, kwargs):
        value = args[0] if args else kwargs["hidden_states"]
        if value is not prefix_outputs.get(ROOT):
            raise ValueError("Native layer-2 input lacks prefix output identity")
        positions = kwargs.get("position_ids")
        if positions is None or positions.dtype != torch.int64 or positions.tolist() != [list(range(30))]:
            raise ValueError("Unexpected entry positions")
        store(ROOT, value)

    def before(name):
        def hook(module, args):
            value = args[0]
            if name == PREFIX + "attention.normalized":
                source = ROOT
            elif name.endswith(".flat"):
                source = PREFIX + "attention.normalized"
            else:
                source = name.replace(".normalized", ".heads")
                store(source, value)
                flat = first._array(record["state_bits"][source.replace(".heads", ".flat")], [1, 30, 1024 if "query" in source else 256], "torch.bfloat16")
                heads = flat.reshape(1, 30, 4 if "query" in source else 1, 256).transpose(0, 2, 1, 3)
                if not np.array_equal(heads, first._bits(value)):
                    raise ValueError("Native head input differs from its projection coordinates")
            if record["state_bits"].get(source) != first._bits(value).tolist():
                raise ValueError("Native entry operand lineage mismatch")
            inputs[name] = value
        return hook

    def after(name):
        def hook(module, args, output):
            store(name, output)
            if name == PREFIX + "value.flat":
                store(V_HEADS, output.reshape(1, 30, 1, 256).transpose(1, 2))
            if name == PREFIX + "key.normalized":
                raise Complete()
        return hook

    def observed(name, original, rms):
        def forward(value):
            if value is not inputs.get(name) or name in record["kernels"]:
                raise ValueError("Repeated or unbound native entry call")
            activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if value.is_cuda else [])
            with profile(activities=activities) as profiled:
                if rms:
                    output, stages = _observe_rms_module(original, value)
                    record["scalar_stages"][name] = stages
                else:
                    output = original(value)
                if value.is_cuda:
                    torch.cuda.synchronize()
            record["kernels"][name] = _kernel_symbols(sorted({event.name for event in profiled.events() if event.device_type == torch.autograd.DeviceType.CUDA}))
            return output
        return forward

    def forbidden(module, args):
        raise ValueError("Execution passed the declared pre-RoPE entry boundary")

    stopped = False
    try:
        with ExitStack() as stack, torch.no_grad():
            handles.append(model.model.layers[1].register_forward_hook(prefix_output))
            handles.append(layer.register_forward_pre_hook(target_input, with_kwargs=True))
            for module in (layer.self_attn.o_proj, layer.mlp, model.model.layers[3], model.model.norm, model.lm_head):
                handles.append(module.register_forward_pre_hook(forbidden))
            for name, module in modules.items():
                handles.append(module.register_forward_pre_hook(before(name)))
                handles.append(module.register_forward_hook(after(name)))
                if traced:
                    stack.enter_context(patch.object(module, "forward", new=observed(name, module.forward, not name.endswith(".flat"))))
            model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, output_attentions=False, logits_to_keep=1)
    except Complete:
        stopped = True
    finally:
        for handle in handles:
            handle.remove()
    names = {ROOT, PREFIX + "attention.normalized", *(PREFIX + role + suffix for role in ("query", "key", "value") for suffix in (".flat", ".heads")), PREFIX + "query.normalized", PREFIX + "key.normalized"}
    if not stopped or set(record["state_bits"]) != names or (traced and set(record["kernels"]) != set(modules)):
        raise ValueError("Incomplete native entry capture")
    record.update(code_before=code, code_after=_code_sha(), runtime_before=runtime, runtime_after=first._runtime(), stopped_before_target_rotary=True,
                  token_commitment_unchanged=token_ids == original_ids and ids.cpu().tolist() == original_ids)
    return record


def entry_report(plan, bundle, observations, guards):
    _check_hash(plan, "plan_sha256")
    if plan["scope"] != SCOPE or _sha(bundle) != plan["bundle_sha256"] or any(plan.get(key) is not value for key, value in _flags().items()):
        raise ValueError("Entry report requires frozen bounded predictions")
    if not isinstance(observations, list) or len(observations) != 3:
        raise ValueError("Three traced/plain entry pairs required")
    comparisons, checks, sets = [], [], []
    expected = bundle["execution"]
    for repetition, pair in enumerate(observations):
        if set(pair) != {"traced", "untraced"}:
            raise ValueError("Incomplete entry observation pair")
        current, compared, devices = {}, {}, set()
        for mode, record in pair.items():
            if set(record["state_bits"]) != set(plan["coverage"]["states"]) or set(record["geometry"]) != set(record["state_bits"]):
                raise ValueError("Entry native state coverage mismatch")
            for name, declaration in plan["coverage"]["states"].items():
                compared[mode + ":" + name] = first._comparison(expected["state_bits"][name], record["state_bits"][name], declaration["shape"], declaration["dtype"])
            devices.update(value["device"] for value in record["geometry"].values())
            current[mode + "_geometry"] = all(first._geometry(record["geometry"][name], value["shape"], value["dtype"]) for name, value in plan["coverage"]["states"].items())
            current[mode + "_source"] = record["code_before"] == record["code_after"] == plan["code_sha256"]
            current[mode + "_runtime"] = canonical_json(record["runtime_before"]) == canonical_json(plan["runtime"]) == canonical_json(record["runtime_after"])
            current[mode + "_boundary"] = record["stopped_before_target_rotary"] is True and record["token_commitment_unchanged"] is True and record["value_heads_mapping"] == V_MAPPING
        traced, plain = pair["traced"], pair["untraced"]
        if plain["scalar_stages"] or plain["kernels"] or set(traced["scalar_stages"]) != set(expected["scalar_stages"]):
            raise ValueError("Incorrect traced/plain RMS stage coverage")
        for name, stages in expected["scalar_stages"].items():
            native = traced["scalar_stages"][name]
            if set(native) != {*first.STAGES, "mean_input_metadata"}:
                raise ValueError("Entry scalar fields mismatch")
            for key, bits in stages.items():
                compared[name + ":" + key] = first._comparison(bits, native[key], [len(bits)], "torch.float32")
            geometry = native["mean_input_metadata"]
            shape = plan["coverage"]["states"][name]["shape"]
            strides = geometry.get("input_strides")
            current[name + "_mean_geometry"] = geometry.get("input_shape") == shape and geometry.get("input_dtype") == "torch.float32" and geometry.get("axes") == [-1] and geometry.get("keepdim") is True and geometry.get("alignment_mod16") == 0 and isinstance(strides, list) and len(strides) == len(shape) and strides[-1] == 1 and all(type(value) is int and value > 0 and value % 4 == 0 for value in strides[:-1])
        current["same_cuda_device"] = len(devices) == 1
        current["kernel_sets"] = traced["kernels"] == plan["expected_kernel_sets"]
        current["plain_matches_traced"] = plain["state_bits"] == traced["state_bits"]
        sets.append(traced["kernels"])
        comparisons.append(compared)
        checks.append(current)
    stable = all(value == sets[0] for value in sets)
    counts = {name: sum(item[name]["mismatch_count"] for item in comparisons) for name in comparisons[0]}
    divergence = next(({"repetition": index, "state": name, **item["first_divergence"]} for index, comp in enumerate(comparisons) for name, item in comp.items() if item["first_divergence"] is not None), None)
    guards_ok = set(guards) == {"checkpoint_unchanged", "source_unchanged", "frozen_files_unchanged"} and all(value is True for value in guards.values())
    if divergence is None:
        divergence = next(({"check": name} for check in checks for name, value in check.items() if not value), None)
    if divergence is None and (not stable or not guards_ok):
        divergence = {"check": "kernel_stability_or_acquisition_guards"}
    passed = not any(counts.values()) and all(all(check.values()) for check in checks) and stable and guards_ok
    return _seal({"schema_version": 1, "scope": SCOPE, "plan_sha256": plan["plan_sha256"], "observations": observations, "comparisons": comparisons,
                  "checks": checks, "mismatch_counts": counts, "aggregate_mismatch_count": sum(counts.values()), "first_divergence": divergence,
                  "entry_matches": passed, "layer_two_entry_independently_recomputed": passed, "coverage": plan["coverage"], "original_forward_count": 6,
                  "kernel_sets_repeat_stable": stable, "acquisition_guards": guards, **_flags()}, "report_sha256")


def acquire_entry(sources, model, plan, bundle, model_path, plan_path, bundle_path):
    for path, value in ((plan_path, plan), (bundle_path, bundle)):
        holdout.require_frozen(path, value)
    context = check_entry_plan(sources, plan, bundle, model_path)
    code, guard = _code_sha(), _sha(sources.commitments())
    if canonical_json(_snapshot_model(sources, model)) != canonical_json(bundle["parameter_snapshots"]):
        raise ValueError("Acquisition weights differ from predictions")
    observations = [{"traced": capture_entry(model, context[0], True), "untraced": capture_entry(model, context[0], False)} for _ in range(3)]
    guards = {"checkpoint_unchanged": canonical_json(_snapshot_model(sources, model)) == canonical_json(bundle["parameter_snapshots"]),
              "source_unchanged": code == _code_sha() and guard == _sha(sources.commitments()), "frozen_files_unchanged": True}
    for path, value in ((plan_path, plan), (bundle_path, bundle)):
        holdout.require_frozen(path, value)
    return entry_report(plan, bundle, observations, guards)


def verify_entry(sources, plan, bundle, report, model_path):
    try:
        code, guard = _code_sha(), _sha(sources.commitments())
        check_entry_plan(sources, plan, bundle, model_path)
        _check_hash(report, "report_sha256")
        expected = entry_report(plan, bundle, report["observations"], report["acquisition_guards"])
        valid = canonical_json(expected) == canonical_json(report) and code == _code_sha() and guard == _sha(sources.commitments())
        return {"valid": valid, "mode": "integrity_source_snapshots_ledger_and_native_recount_only",
                "entry_matches": valid and expected["entry_matches"], "numerical_recomputation_performed": False, **_flags()}
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError, IndexError, OSError) as error:
        return {"valid": False, "entry_matches": False, "reason": str(error), **_flags()}


def replay_entry(sources, model, plan, bundle, report, model_path, plan_path, bundle_path, workers=4):
    checked = verify_entry(sources, plan, bundle, report, model_path)
    if not checked["valid"]:
        return checked
    new_plan, new_bundle = build_entry_plan(sources, model, model_path, workers)
    same = canonical_json(new_plan) == canonical_json(plan) and canonical_json(new_bundle) == canonical_json(bundle)
    actual = acquire_entry(sources, model, plan, bundle, model_path, plan_path, bundle_path) if same else None
    exact = same and canonical_json(actual) == canonical_json(report)
    return {"valid": exact, "entry_matches": exact and actual["entry_matches"], "predictions_recomputed_exact": same, "reexecution_exact": exact,
            "numerical_recomputation_performed": True, "mode": "fresh_six_weight_nine_node_prediction_then_native_replay", **_flags()}


def entry_summary(plan, report):
    return _seal({**{key: value for key, value in report.items() if key not in ("observations", "comparisons", "report_sha256")},
                  "source_report_sha256": report["report_sha256"], "sources": plan["sources"], "code_sha256": plan["code_sha256"], "root_binding": plan["root_binding"],
                  "trace_root": plan["trace_root"], "profiles": plan["profiles"], "kernels": [pair["traced"]["kernels"] for pair in report["observations"]]}, "summary_sha256")
