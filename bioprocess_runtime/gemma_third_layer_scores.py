from __future__ import annotations

import copy
import hashlib
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from . import gemma_third_layer_entry as entry
from . import gemma_first_layer as first
from .gemma_rotary_slice import _sha, _seal, _check_hash
from .gemma_mlp_down import _kernel_symbols
from .operational_semantics import append_chain_record, verify_trace_chain
from .serialization import canonical_json

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
ENTRY_PLAN = "b8b1d5bcfcdfce776f7dba5316abcee864ec9c68f023efd75b68e4dfd3e29a30"
ENTRY_REPORT = "d56dfe2d2609eccd714a452cf52c07cc6b3b954328c5a4b0dc7c84ab449b6b2a"
P = "layer.2."
Q, K, V = P + "query.normalized", P + "key.normalized", P + "value.heads"
COS, SIN, MASK = "rotary.local.cosine", "rotary.local.sine", "mask.sliding"
QR, KR, KP, VP = P + "query.rotary", P + "key.rotary", P + "key.repeated", P + "value.repeated"
RAW, SCALE, MASKED = (P + "attention." + name for name in ("unscaled_scores", "scaled_scores", "masked_scores"))
ROOTS = (Q, K, V, COS, SIN, MASK)
OUTPUTS = (QR, KR, KP, VP, RAW, SCALE, MASKED)
SHAPES = {Q: [1, 4, 30, 256], K: [1, 1, 30, 256], V: [1, 1, 30, 256], COS: [1, 30, 256], SIN: [1, 30, 256], MASK: [1, 1, 30, 30],
          QR: [1, 4, 30, 256], KR: [1, 1, 30, 256], KP: [1, 4, 30, 256], VP: [1, 4, 30, 256], RAW: [1, 4, 30, 30], SCALE: [1, 4, 30, 30], MASKED: [1, 4, 30, 30]}
SCOPE = "Fixed baseline decoder-index-2 local rotary/repeat-KV/QK/BF16-scale/mask slice i0074..i0079 from six verified reused boundaries. Includes value repetition as an observed pre-softmax companion, not a dependency of masked scores. Original operations are retained and stopped before target softmax; plain control uses function-boundary hooks/wrappers without dispatch observation or profiling and does not observe scaled scores. Not connected three-layer execution, native internal arithmetic, a fresh holdout, hardware or unrestricted qualification."
FALSE_FLAGS = ("qualified", "full_model_qualified", "hardware_semantics_established", "global_exactness_activation_allowed", "qualification_promotion_allowed", "shape_transfer_prequalified", "candidate_refitting_allowed", "connected_three_layers_independently_recomputed", "prefix_independently_recomputed", "internal_slice_predictions_reused", "fresh_prompt_holdout", "target_softmax_executed", "target_value_aggregation_executed", "target_mlp_executed", "later_layers_executed", "final_model_norm_executed", "logits_executed", "native_arithmetic_reconstructed", "native_registers_observed")
GUARDS = ("checkpoint_unchanged", "source_unchanged", "frozen_files_unchanged")


def _flags():
    return {**dict.fromkeys(FALSE_FLAGS, False), "prefix_boundary_reused": True, "stored_boundary_predictions_used": True, "empirical_rotary_specification_reused": True}


def _code_sha():
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Layer-2 rotary/score source changed after import")
    return _sha({"module": SOURCE_SHA256, "entry": entry._code_sha(), "arithmetic": first._code_sha()})


def instructions(program):
    entry.instructions(program)
    nodes = program["instructions"][74:80]
    if [node["id"] for node in nodes] != [f"i{i:04d}" for i in range(74, 80)]:
        raise ValueError("Wrong layer-2 rotary/score instruction range")
    available = set(ROOTS)
    for node in nodes:
        if node["layer"] != 2 or node["parameter_refs"] or any(name not in available for name in node["inputs"]) or available.intersection(node["outputs"]):
            raise ValueError("Wrong rotary/score dependencies or parameters")
        available.update(node["outputs"])
    if available != set(SHAPES) or [name for node in nodes for name in node["outputs"]] != list(OUTPUTS):
        raise ValueError("Incomplete rotary/score state coverage")
    return nodes


def _roots(root_bits):
    if set(root_bits) != set(ROOTS):
        raise ValueError("Exactly six verified rotary/score boundaries required")
    return {name: first._array(root_bits[name], SHAPES[name], "torch.bfloat16").copy() for name in ROOTS}


def _payload(node, states, hashes):
    return {"instruction_id": node["id"], "instruction_sha256": node["instruction_sha256"], "opcode": node["opcode"],
            "inputs": {name: {"descriptor": first._descriptor(states[name]), "producer_record_hash": hashes[name]} for name in node["inputs"]},
            "outputs": {name: first._descriptor(states[name]) for name in node["outputs"]},
            "provider": {"ROTARY_APPLY_PAIR": "unchanged_BF16_mul_mul_add", "REPEAT_KV": "integer_coordinate_repeat", "MATMUL_QK": "unchanged_operand_alignment_K256", "SCALE": "BF16_RNE_multiply_0x3d80", "ADD": "BF16_RNE_causal_mask_add"}[node["opcode"]]}


def execute_scores(program, root_bits, profiles, workers=4):
    code = _code_sha()
    first._profiles(profiles)
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Expected one to four CPU workers")
    states = _roots(root_bits)
    hashes, records = {name: "ROOT:" + first._descriptor(value)["sha256"] for name, value in states.items()}, []
    for node in instructions(program):
        inputs = [states[name] for name in node["inputs"]]
        opcode = node["opcode"]
        if opcode == "ROTARY_APPLY_PAIR":
            values = [first.rotate_bfloat16_bits(value, inputs[2], inputs[3]) for value in inputs[:2]]
        elif opcode == "REPEAT_KV":
            values = [np.repeat(inputs[0], node["attributes"]["repetitions"], axis=1)]
        elif opcode == "MATMUL_QK":
            values = [np.stack([first._project_rows(inputs[0][0, head], inputs[1][0, head], "serial", profiles["serial"], workers) for head in range(4)])[None, ...]]
        elif opcode == "SCALE":
            values = [np.asarray([first.bfloat16_multiply_bits(int(value), 0x3D80) for value in inputs[0].reshape(-1)], dtype=np.uint16).reshape(SHAPES[SCALE])]
        else:
            values = [first.mask_score_bits(*inputs)]
        for name, value in zip(node["outputs"], values):
            states[name] = first._array(value, SHAPES[name], "torch.bfloat16")
        record = append_chain_record(records, _payload(node, states, hashes))
        hashes.update({name: record["record_hash"] for name in node["outputs"]})
    result = {"state_bits": {name: value.tolist() for name, value in states.items()}, "records": records}
    check_execution(program, root_bits, result)
    if code != _code_sha():
        raise ValueError("Rotary/score source changed during prediction")
    return result


def check_execution(program, root_bits, execution):
    states = _roots(root_bits)
    if set(execution) != {"state_bits", "records"} or set(execution["state_bits"]) != set(SHAPES) or len(execution["records"]) != 6 or not verify_trace_chain(execution["records"])["valid"]:
        raise ValueError("Rotary/score state or ledger coverage mismatch")
    for name in ROOTS:
        if not np.array_equal(states[name], first._array(execution["state_bits"][name], SHAPES[name], "torch.bfloat16")):
            raise ValueError("Rotary/score reused boundary changed")
    hashes, records = {name: "ROOT:" + first._descriptor(value)["sha256"] for name, value in states.items()}, []
    for node, recorded in zip(instructions(program), execution["records"]):
        for name in node["outputs"]:
            states[name] = first._array(execution["state_bits"][name], SHAPES[name], "torch.bfloat16")
        expected = append_chain_record(records, _payload(node, states, hashes))
        if canonical_json(expected) != canonical_json(recorded):
            raise ValueError("Rotary/score producer or output ledger mismatch")
        hashes.update({name: expected["record_hash"] for name in node["outputs"]})


@dataclass(frozen=True)
class ScoreSources:
    entry_sources: entry.EntrySources
    entry_plan: dict[str, Any]
    entry_bundle: dict[str, Any]
    entry_report: dict[str, Any]

    @property
    def program(self):
        return self.entry_sources.program

    @property
    def runtime(self):
        return self.entry_sources.runtime

    def commitments(self):
        return {"entry_sources": self.entry_sources.commitments(), **{name + "_payload_sha256": _sha(getattr(self, name)) for name in ("entry_plan", "entry_bundle", "entry_report")}}

    def validate(self, model_path):
        code, guard = _code_sha(), _sha(self.commitments())
        for data, field, digest in ((self.entry_plan, "plan_sha256", ENTRY_PLAN), (self.entry_report, "report_sha256", ENTRY_REPORT)):
            _check_hash(data, field)
            if data[field] != digest:
                raise ValueError("Declared passing layer-2 entry evidence required")
        cached = entry.holdout._CachedTwoSources(self.entry_sources)
        verified = entry.verify_entry(cached, self.entry_plan, self.entry_bundle, self.entry_report, model_path)
        if not verified.get("valid") or not verified.get("entry_matches"):
            raise ValueError("Layer-2 entry must pass before rotary/scores")
        ids, providers, profiles, _, _, _ = cached.validate(model_path)
        baseline = self.entry_sources.baseline.two_layer_bundle["execution"]
        roots = {name: self.entry_bundle["execution"]["state_bits"][name] for name in (Q, K, V)}
        roots.update({name: baseline["state_bits"][name] for name in (COS, SIN, MASK)})
        values = _roots(roots)
        providers.validate(self.runtime)
        for name, expected in ((COS, providers.rotary_cosine), (SIN, providers.rotary_sine), (MASK, first.causal_mask_bits())):
            if not np.array_equal(values[name], expected):
                raise ValueError("Rotary/mask roots differ from checked primitive specification")
        bindings = {}
        for name in ROOTS:
            execution = self.entry_bundle["execution"] if name in (Q, K, V) else baseline
            found = [record for record in execution["records"] if name in record["payload"]["outputs"]]
            if len(found) != 1 or found[0]["payload"]["outputs"][name] != first._descriptor(values[name]):
                raise ValueError("Missing verified root producer")
            bindings[name] = {"descriptor": first._descriptor(values[name]), "producer_record_hash": found[0]["record_hash"], "instruction_id": found[0]["payload"]["instruction_id"]}
        prior = self.entry_sources.baseline.two_sources.second_report["kernel_trace"]
        if len(prior) != 3 or any(value != prior[0] for value in prior):
            raise ValueError("Prior score kernel evidence is unstable")
        kernels = {name: prior[0][name.replace(P, "layer.1.", 1)] for name in (RAW, SCALE, MASKED)}
        instructions(self.program)
        if code != _code_sha() or guard != _sha(self.commitments()):
            raise ValueError("Rotary/score sources changed during validation")
        return ids, providers, profiles, {name: value.tolist() for name, value in values.items()}, bindings, kernels


def _coverage():
    return {"layer_index": 2, "instruction_ids": [f"i{i:04d}" for i in range(74, 80)], "instruction_count": 6,
            "root_inputs": list(ROOTS), "new_state_count": 7, "state_count": 13,
            "states": {name: {"shape": shape, "dtype": "torch.bfloat16"} for name, shape in SHAPES.items()},
            "plain_observed_states": [name for name in SHAPES if name != SCALE], "rotated_value_count": 38400,
            "score_value_count_per_stage": 3600, "attention_type": "sliding_attention", "rotary_profile": "local", "sliding_window": 512,
            "value_repeat_is_companion_not_masked_score_dependency": True}


def _plan_body(sources, context, execution):
    ids, providers, profiles, roots, bindings, kernels = context
    return {"schema_version": 1, "scope": SCOPE, "sources": sources.commitments(), "code_sha256": _code_sha(), "runtime": sources.runtime,
            "input_token_ids": ids, "root_bindings": bindings, "root_bits_sha256": _sha(roots), "coverage": _coverage(),
            "profiles": profiles, "providers": providers.commitments(), "expected_score_kernel_sets": kernels,
            "rotary_kernel_scope": "observed whole original rotary-apply call; repeat-stable symbols, not native-internal proof",
            "plain_scope": "original function-boundary hooks/wrappers; no dispatch observation/profiling; scaled scores unobserved",
            "state_descriptors": {name: first._descriptor(first._array(value, SHAPES[name], "torch.bfloat16")) for name, value in execution["state_bits"].items()},
            "bundle_sha256": _sha(execution), "trace_root": execution["records"][-1]["record_hash"], "repetitions": 3, "required_forward_count": 6, **_flags()}


def build_score_plan(sources, model, model_path, workers=4):
    code, guard = _code_sha(), _sha(sources.commitments())
    context = sources.validate(model_path)
    snapshot = entry._snapshot_model(sources.entry_sources, model)
    if canonical_json(snapshot) != canonical_json(sources.entry_bundle["parameter_snapshots"]):
        raise ValueError("Original layer-2 parameters differ from entry roots")
    execution = execute_scores(sources.program, context[3], context[2], workers)
    if canonical_json(entry._snapshot_model(sources.entry_sources, model)) != canonical_json(snapshot) or code != _code_sha() or guard != _sha(sources.commitments()):
        raise ValueError("Checkpoint/source changed while freezing rotary/scores")
    return _seal(_plan_body(sources, context, execution), "plan_sha256"), execution


def check_score_plan(sources, plan, bundle, model_path):
    code, guard = _code_sha(), _sha(sources.commitments())
    _check_hash(plan, "plan_sha256")
    context = sources.validate(model_path)
    check_execution(sources.program, context[3], bundle)
    if canonical_json(plan) != canonical_json(_seal(_plan_body(sources, context, bundle), "plan_sha256")) or code != _code_sha() or guard != _sha(sources.commitments()):
        raise ValueError("Rotary/score plan or source commitments mismatch")
    return context


def capture_scores(model, token_ids, traced):
    import torch
    from torch.profiler import profile, ProfilerActivity
    from torch.utils._python_dispatch import TorchDispatchMode
    from transformers.models.gemma3 import modeling_gemma3 as gemma

    layer, attention = model.model.layers[2], model.model.layers[2].self_attn
    original_rotary, original_repeat, original_eager = gemma.apply_rotary_pos_emb, gemma.repeat_kv, gemma.eager_attention_forward
    original_matmul, original_softmax = torch.matmul, torch.nn.functional.softmax
    records, tensors, handles, calls = {"state_bits": {}, "geometry": {}, "kernels": {}}, {}, [], {}
    active = {"layer": False, "eager": False}
    before_code, before_runtime, ids_before = _code_sha(), first._runtime(), copy.deepcopy(token_ids)
    ids = torch.tensor(token_ids, dtype=torch.int64, device=layer.input_layernorm.weight.device)

    class Complete(Exception):
        pass

    def bits(value):
        with torch._C._DisableTorchDispatch():
            return first._bits(value)

    def once(name):
        calls[name] = calls.get(name, 0) + 1
        if calls[name] != 1:
            raise ValueError("Repeated native rotary/score operation: " + name)

    def store(name, value):
        if name in tensors or value.dtype != torch.bfloat16 or list(value.shape) != SHAPES[name]:
            raise ValueError("Unexpected native rotary/score state occurrence or geometry")
        tensors[name] = value
        records["state_bits"][name] = bits(value).tolist()
        records["geometry"][name] = {"shape": list(value.shape), "strides": list(value.stride()), "dtype": str(value.dtype), "device": str(value.device), "alignment_mod16": value.data_ptr() % 16}

    def link(value, name, identity=True):
        if (identity and value is not tensors.get(name)) or records["state_bits"].get(name) != bits(value).tolist():
            raise ValueError("Native rotary/score operand lineage mismatch: " + name)

    def run_profiled(name, call):
        once(name)
        if not traced:
            return call()
        activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if ids.is_cuda else [])
        with profile(activities=activities) as captured:
            output = call()
            if ids.is_cuda:
                torch.cuda.synchronize()
        records["kernels"][name] = _kernel_symbols(sorted({event.name for event in captured.events() if event.device_type == torch.autograd.DeviceType.CUDA}))
        return output

    def enter(module, args, kwargs):
        once("target_layer")
        if kwargs["position_ids"].tolist() != [list(range(30))] or kwargs.get("past_key_value") is not None or kwargs.get("use_cache") or kwargs.get("output_attentions"):
            raise ValueError("Unsupported target positions/cache/attention-output regime")
        if layer.layer_idx != 2 or not attention.is_sliding or layer.attention_type != "sliding_attention":
            raise ValueError("Wrong target local rotary/sliding regime")
        cosine, sine = kwargs["position_embeddings_local"]
        store(COS, cosine)
        store(SIN, sine)
        store(MASK, kwargs["attention_mask"])
        active["layer"] = True

    def norm_hook(name):
        def hook(module, args, output):
            store(name, output)
        return hook

    def rotary(q, k, cosine, sine, *args, **kwargs):
        if not active["layer"]:
            return original_rotary(q, k, cosine, sine, *args, **kwargs)
        if args or set(kwargs) - {"unsqueeze_dim"} or kwargs.get("unsqueeze_dim", 1) != 1:
            raise ValueError("Unsupported native rotary signature")
        for value, name in ((q, Q), (k, K), (cosine, COS), (sine, SIN)):
            link(value, name)
        output = run_profiled("rotary_apply", lambda: original_rotary(q, k, cosine, sine, **kwargs))
        if type(output) is not tuple or len(output) != 2:
            raise ValueError("Native rotary must return the original Q/K pair")
        store(QR, output[0])
        store(KR, output[1])
        return output

    def repeat(value, repetitions):
        if not active["eager"]:
            return original_repeat(value, repetitions)
        name, root = (KP, KR) if KP not in tensors else (VP, V)
        once(name)
        if repetitions != 4:
            raise ValueError("Unexpected native KV repetition count")
        link(value, root)
        output = original_repeat(value, repetitions)
        store(name, output)
        return output

    def matmul(left, right, *args, **kwargs):
        if not active["eager"]:
            return original_matmul(left, right, *args, **kwargs)
        if RAW in tensors or args or kwargs:
            raise ValueError("Repeated QK or forbidden value aggregation")
        link(left, QR)
        expected = np.asarray(records["state_bits"][KP], dtype=np.uint16).transpose(0, 1, 3, 2)
        if not np.array_equal(bits(right), expected) or right.data_ptr() != tensors[KP].data_ptr():
            raise ValueError("QK right operand is not the original repeated-key transpose")
        output = run_profiled(RAW, lambda: original_matmul(left, right))
        store(RAW, output)
        return output

    class ObserveScores(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func is torch.ops.aten.mul.Tensor and RAW in tensors:
                if len(args) != 2 or type(args[1]) not in (float, int) or args[1] != 0.0625 or kwargs:
                    raise ValueError("Unexpected native score scaling")
                link(args[0], RAW)
                name = SCALE
            elif func is torch.ops.aten.add.Tensor and SCALE in tensors:
                if len(args) != 2 or set(kwargs) - {"alpha"} or type(kwargs.get("alpha", 1)) not in (int, float) or kwargs.get("alpha", 1) != 1:
                    raise ValueError("Unexpected native mask addition")
                link(args[0], SCALE)
                link(args[1], MASK, False)
                if args[1].data_ptr() != tensors[MASK].data_ptr():
                    raise ValueError("Mask operand is not the original mask view")
                name = MASKED
            else:
                return func(*args, **kwargs)
            output = run_profiled(name, lambda: func(*args, **kwargs))
            store(name, output)
            return output

    def eager(module, query, key, value, mask, **kwargs):
        if module is not attention:
            return original_eager(module, query, key, value, mask, **kwargs)
        once("eager")
        if not active["layer"] or active["eager"] or kwargs.get("scaling") != 0.0625 or kwargs.get("dropout") != 0.0 or kwargs.get("softcap") is not None or module.training or module.num_key_value_groups != 4:
            raise ValueError("Unsupported target eager attention invocation")
        link(query, QR)
        link(key, KR)
        link(mask, MASK)
        store(V, value)
        active["eager"] = True
        try:
            if traced:
                with ObserveScores():
                    return original_eager(module, query, key, value, mask, **kwargs)
            return original_eager(module, query, key, value, mask, **kwargs)
        finally:
            active["eager"] = False

    def softmax(value, *args, **kwargs):
        if not active["eager"]:
            return original_softmax(value, *args, **kwargs)
        once("softmax_boundary")
        if args or kwargs != {"dim": -1, "dtype": torch.float32}:
            raise ValueError("Unexpected pre-softmax signature")
        if traced:
            link(value, MASKED)
        else:
            store(MASKED, value)
        raise Complete()

    def forbidden(module, args):
        raise ValueError("Execution escaped the pre-softmax slice")

    stopped = False
    try:
        with ExitStack() as stack, torch.no_grad():
            handles.append(layer.register_forward_pre_hook(enter, with_kwargs=True))
            handles.append(attention.q_norm.register_forward_hook(norm_hook(Q)))
            handles.append(attention.k_norm.register_forward_hook(norm_hook(K)))
            for module in (attention.o_proj, layer.mlp, model.model.layers[3], model.model.norm, model.lm_head):
                handles.append(module.register_forward_pre_hook(forbidden))
            for target, name, value in ((gemma, "apply_rotary_pos_emb", rotary), (gemma, "repeat_kv", repeat), (gemma, "eager_attention_forward", eager), (torch, "matmul", matmul), (torch.nn.functional, "softmax", softmax)):
                stack.enter_context(patch.object(target, name, new=value))
            model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, output_attentions=False, logits_to_keep=1)
    except Complete:
        stopped = True
    finally:
        for handle in handles:
            handle.remove()
    names = set(SHAPES) if traced else set(SHAPES) - {SCALE}
    expected_calls = {"target_layer", "rotary_apply", "eager", KP, VP, RAW, "softmax_boundary"} | ({SCALE, MASKED} if traced else set())
    if not stopped or set(records["state_bits"]) != names or set(calls) != expected_calls or any(value != 1 for value in calls.values()):
        raise ValueError("Incomplete or repeated native rotary/score coverage")
    for name, tensor in tensors.items():
        link(tensor, name)
    records.update(code_before=before_code, code_after=_code_sha(), runtime_before=before_runtime, runtime_after=first._runtime(),
                   token_commitment_unchanged=token_ids == ids_before and ids.cpu().tolist() == ids_before,
                   stopped_before_softmax=True, all_original_operands_unchanged=True,
                   plain_scope="no dispatch observation or profiling; original function-boundary wrappers only")
    return records


def score_report(plan, bundle, observations, guards):
    _check_hash(plan, "plan_sha256")
    if plan["scope"] != SCOPE or plan["bundle_sha256"] != _sha(bundle) or any(plan.get(key) is not value for key, value in _flags().items()):
        raise ValueError("Rotary/score comparison requires frozen bounded prediction")
    if not isinstance(observations, list) or len(observations) != 3:
        raise ValueError("Three traced/plain pairs required")
    comparisons, checks, kernels = [], [], []
    for pair in observations:
        if set(pair) != {"traced", "untraced"}:
            raise ValueError("Native pair coverage mismatch")
        compared, check, devices = {}, {}, set()
        for mode, record in pair.items():
            names = set(SHAPES) if mode == "traced" else set(SHAPES) - {SCALE}
            if set(record["state_bits"]) != names or set(record["geometry"]) != names:
                raise ValueError("Native rotary/score state coverage mismatch")
            for name in names:
                compared[mode + ":" + name] = first._comparison(bundle["state_bits"][name], record["state_bits"][name], SHAPES[name], "torch.bfloat16")
                devices.add(record["geometry"][name]["device"])
            check[mode + "_geometry"] = all(first._geometry(record["geometry"][name], SHAPES[name], "torch.bfloat16") for name in names)
            check[mode + "_code"] = record["code_before"] == record["code_after"] == plan["code_sha256"]
            check[mode + "_runtime"] = canonical_json(record["runtime_before"]) == canonical_json(plan["runtime"]) == canonical_json(record["runtime_after"])
            check[mode + "_stop_and_lineage"] = record["stopped_before_softmax"] is True and record["token_commitment_unchanged"] is True and record["all_original_operands_unchanged"] is True
        traced, plain = pair["traced"], pair["untraced"]
        if plain["kernels"]:
            raise ValueError("Plain observation must not claim profiling")
        check["plain_matches_traced"] = all(plain["state_bits"][name] == traced["state_bits"][name] for name in set(SHAPES) - {SCALE})
        symbols = {name: _kernel_symbols(value) for name, value in traced["kernels"].items()}
        check["kernel_scope"] = set(symbols) == {"rotary_apply", RAW, SCALE, MASKED}
        check["score_kernels_match"] = all(symbols.get(name) == plan["expected_score_kernel_sets"][name] for name in (RAW, SCALE, MASKED))
        check["same_cuda_device"] = len(devices) == 1
        comparisons.append(compared)
        checks.append(check)
        kernels.append(symbols)
    counts = {name: sum(value[name]["mismatch_count"] for value in comparisons) for name in comparisons[0]}
    stable = all(value == kernels[0] for value in kernels)
    guarded = set(guards) == set(GUARDS) and all(value is True for value in guards.values())
    order = [mode + ":" + name for name in SHAPES for mode in ("traced", "untraced") if mode == "traced" or name != SCALE]
    divergence = next(({"repetition": index, "state": name, **values[name]["first_divergence"]} for index, values in enumerate(comparisons) for name in order if values[name]["first_divergence"] is not None), None)
    if divergence is None:
        divergence = next(({"check": name} for values in checks for name, value in values.items() if not value), None)
    if divergence is None and (not guarded or not stable):
        divergence = {"check": "acquisition_guard_or_kernel_stability"}
    passed = guarded and stable and not any(counts.values()) and all(all(value.values()) for value in checks)
    return _seal({"schema_version": 1, "scope": SCOPE, "plan_sha256": plan["plan_sha256"], "observations": observations,
                  "comparisons": comparisons, "checks": checks, "mismatch_counts": counts, "aggregate_mismatch_count": sum(counts.values()),
                  "first_divergence": divergence, "rotary_scores_match": passed, "coverage": plan["coverage"],
                  "kernels_repeat_stable": stable, "acquisition_guards": guards, "original_forward_count": 6, **_flags()}, "report_sha256")


def acquire_scores(sources, model, plan, bundle, model_path, plan_path, bundle_path):
    for path, payload in ((plan_path, plan), (bundle_path, bundle)):
        entry.holdout.require_frozen(path, payload)
    context = check_score_plan(sources, plan, bundle, model_path)
    code, guard = _code_sha(), _sha(sources.commitments())
    snapshot = entry._snapshot_model(sources.entry_sources, model)
    if canonical_json(snapshot) != canonical_json(sources.entry_bundle["parameter_snapshots"]):
        raise ValueError("Native entry weights differ from frozen Q/K/V sources")
    observations = [{"traced": capture_scores(model, context[0], True), "untraced": capture_scores(model, context[0], False)} for _ in range(3)]
    guards = {"checkpoint_unchanged": canonical_json(entry._snapshot_model(sources.entry_sources, model)) == canonical_json(snapshot),
              "source_unchanged": code == _code_sha() and guard == _sha(sources.commitments()), "frozen_files_unchanged": True}
    for path, payload in ((plan_path, plan), (bundle_path, bundle)):
        entry.holdout.require_frozen(path, payload)
    return score_report(plan, bundle, observations, guards)


def verify_scores(sources, plan, bundle, report, model_path):
    try:
        code, guard = _code_sha(), _sha(sources.commitments())
        check_score_plan(sources, plan, bundle, model_path)
        _check_hash(report, "report_sha256")
        expected = score_report(plan, bundle, report["observations"], report["acquisition_guards"])
        valid = canonical_json(expected) == canonical_json(report) and code == _code_sha() and guard == _sha(sources.commitments())
        return {"valid": valid, "rotary_scores_match": valid and expected["rotary_scores_match"], "mode": "source_root_ledger_integrity_and_native_recount_only", "numerical_recomputation_performed": False, **_flags()}
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError, IndexError, OSError) as error:
        return {"valid": False, "rotary_scores_match": False, "reason": str(error), **_flags()}


def replay_scores(sources, model, plan, bundle, report, model_path, plan_path, bundle_path, workers=4):
    checked = verify_scores(sources, plan, bundle, report, model_path)
    if not checked["valid"]:
        return checked
    new_plan, new_bundle = build_score_plan(sources, model, model_path, workers)
    same = canonical_json(new_plan) == canonical_json(plan) and canonical_json(new_bundle) == canonical_json(bundle)
    observed = acquire_scores(sources, model, plan, bundle, model_path, plan_path, bundle_path) if same else None
    exact = same and canonical_json(observed) == canonical_json(report)
    return {"valid": exact, "rotary_scores_match": exact and observed["rotary_scores_match"], "predictions_recomputed_exact": same,
            "reexecution_exact": exact, "numerical_recomputation_performed": True, "mode": "fresh_rotary_repeat_QK_scale_mask_prediction_then_native_replay", **_flags()}


def score_summary(plan, report):
    return _seal({**{key: value for key, value in report.items() if key not in ("observations", "comparisons", "report_sha256")},
                  "source_report_sha256": report["report_sha256"], "sources": plan["sources"], "root_bindings": plan["root_bindings"],
                  "code_sha256": plan["code_sha256"], "trace_root": plan["trace_root"], "profiles": plan["profiles"],
                  "kernels": [pair["traced"]["kernels"] for pair in report["observations"]]}, "summary_sha256")
