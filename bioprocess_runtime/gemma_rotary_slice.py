from __future__ import annotations

import hashlib
import importlib
import inspect
from typing import Any
from unittest.mock import patch

import numpy as np

from .gemma_attention_entry import _model_parameters, _descriptor, _bits, _state_array, _entry_code_sha, build_attention_entry_plan, check_attention_entry_plan
from .gemma_ir import verify_gemma_ir
from .gemma_ir_interpreter import bind_model_tensors
from .gemma_rsqrt_lookup import CheckedRsqrtLookup, _runtime
from .operational_semantics import tensor_descriptor, append_chain_record, verify_trace_chain
from .serialization import canonical_json

TABLE_SCOPE = "Empirical fixed-position local RoPE table for one bound checkpoint, positions 0..29 and BF16 output; independently checked FP32 angles and casts, not reconstructed native trigonometric arithmetic."
SCOPE = "Connected fixed-input first-layer execution through RoPE using explicit compiled rotary tables and independent BF16 multiply/multiply/add; original eager-attention inputs compared before attention arithmetic; not full-layer/model qualification."
ROTATED = ("layer.0.query.rotary", "layer.0.key.rotary")


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _seal(body: dict[str, Any], field: str) -> dict[str, Any]:
    return {**body, field: _sha(body)}


def _check_hash(value: Any, field: str) -> None:
    if not isinstance(value, dict) or _sha({k: v for k, v in value.items() if k != field}) != value.get(field):
        raise ValueError("Rotary evidence commitment mismatch")


def _rotary_instructions(program: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not verify_gemma_ir(program)["valid"]:
        raise ValueError("Rotary slice requires a valid IR")
    position = next(item for item in program["instructions"] if item["outputs"] == ["position_ids"])
    table = next(item for item in program["instructions"] if item["outputs"] == ["rotary.local.cosine", "rotary.local.sine"])
    rotation = next(item for item in program["instructions"] if item["outputs"] == list(ROTATED))
    if position["opcode"] != "ARANGE" or position["inputs"] != ["input_ids"] or table["opcode"] != "ROTARY_TABLE" or table["inputs"] != ["hidden.0", "position_ids"] or table["parameter_refs"] != ["model.rotary_emb_local.inv_freq"] or table["attributes"] != {"attention_scaling": 1.0}:
        raise ValueError("Unsupported rotary position/table specification")
    if rotation["opcode"] != "ROTARY_APPLY_PAIR" or rotation["inputs"] != ["layer.0.query.normalized", "layer.0.key.normalized", "rotary.local.cosine", "rotary.local.sine"] or rotation["attributes"] != {"rotary_profile": "local"}:
        raise ValueError("Unsupported first-layer rotary binding")
    return position, table, rotation


def _f32_bits(value: Any) -> np.ndarray:
    import torch

    if value.dtype != torch.float32:
        raise ValueError("Rotary stage must be float32")
    with torch._C._DisableTorchDispatch():
        return value.detach().contiguous().cpu().view(torch.int32).numpy().astype(np.uint32)


def rotary_angle_bits(frequency_bits: list[int], positions: list[int]) -> np.ndarray:
    from fractions import Fraction
    from .reference_gemma import _rms_f32_mul, _rms_f32_round

    if len(frequency_bits) != 128 or positions != list(range(30)):
        raise ValueError("Rotary angles are restricted to the bound 128 frequencies and positions 0..29")
    half = [[_rms_f32_mul(bits, _rms_f32_round(Fraction(position))) for bits in frequency_bits] for position in positions]
    return np.concatenate((np.asarray(half, dtype=np.uint32), np.asarray(half, dtype=np.uint32)), axis=-1)[None, ...]


def _rotary_context(program: dict[str, Any], model: Any) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    parameters = _model_parameters(program, model)
    _rotary_instructions(program)
    module = model.model.rotary_emb_local
    frequency = parameters["model.rotary_emb_local.inv_freq"]
    if module.inv_freq is not frequency or frequency.dtype != torch.float32 or frequency.device != model.model.embed_tokens.weight.device or list(frequency.shape) != [128] or module.attention_scaling != 1.0 or module.rope_type != "default" or not model.model.layers[0].self_attn.is_sliding:
        raise ValueError("Unsupported original rotary module")
    binding = {"class": type(model).__module__ + "." + type(model).__name__, "config_sha256": _sha(model.config.to_dict()), "parameter_commitments_sha256": _sha(program["parameter_commitments"])}
    return module, frequency, binding


def build_rotary_table_plan(program: dict[str, Any], model: Any) -> dict[str, Any]:
    from .reference_gemma import _rms_code_commitment

    module, frequency, binding = _rotary_context(program, model)
    position, table, _ = _rotary_instructions(program)
    angles = rotary_angle_bits(_f32_bits(frequency).tolist(), list(range(30)))
    return _seal({"schema_version": 1, "scope": TABLE_SCOPE, "program_sha256": program["program_sha256"], "model_binding": binding,
                  "position_instruction_sha256": position["instruction_sha256"], "table_instruction_sha256": table["instruction_sha256"],
                  "frequency": tensor_descriptor(frequency), "positions": list(range(30)), "input_shape": [1, 30, 640], "output_shape": [1, 30, 256],
                  "angle_bits_sha256": _sha(angles.tolist()), "attention_scaling": 1.0,
                  "angle_code_sha256": hashlib.sha256(inspect.getsource(rotary_angle_bits).encode("utf-8")).hexdigest(),
                  "finite_arithmetic_sha256": _rms_code_commitment(), "rotary_module_source_sha256": hashlib.sha256(inspect.getsource(type(module).forward).encode("utf-8")).hexdigest(),
                  "runtime": _runtime(), "repetitions": 3, "acquisition_schedule": "one warm-up, one profiled/traced module call, and one untraced output control per repetition", "native_trigonometric_arithmetic_reconstructed": False,
                  "unseen_positions_allowed": False, "global_exactness_activation_allowed": False}, "plan_sha256")


def _observe_rotary(module: Any, value: Any, positions: Any) -> tuple[Any, dict[str, Any]]:
    from torch.utils._python_dispatch import TorchDispatchMode

    captured = {}

    class Capture(TorchDispatchMode):
        def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
            output = func(*args, **(kwargs or {}))
            if str(func) in ("aten.cos.default", "aten.sin.default"):
                name = "cosine" if str(func) == "aten.cos.default" else "sine"
                if name in captured:
                    raise ValueError("Repeated rotary transcendental occurrence")
                captured[name] = {"input_bits": _f32_bits(args[0]).tolist(), "output_bits": _f32_bits(output).tolist()}
            return output

    with Capture():
        output = module(value, positions)
    if set(captured) != {"cosine", "sine"}:
        raise ValueError("Original rotary module did not expose cosine and sine stages")
    return output, captured


def _table_manifest(plan: dict[str, Any], bundle: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    from .gemma_float_semantics import encode_bfloat16_rne
    from .gemma_reduction_semantics import decode_finite_float32

    if len(bundle["repetitions"]) != 3:
        raise ValueError("Missing rotary table repetitions")
    matches, casts, controls = True, True, True
    for record in bundle["repetitions"]:
        if not record["profiled_cuda_event_names"] or any(not isinstance(name, str) or not name for name in record["profiled_cuda_event_names"]):
            raise ValueError("Missing rotary CUDA activity provenance")
        for name in ("cosine", "sine"):
            stage = record["stages"][name]
            inputs = np.asarray(stage["input_bits"], dtype=object)
            outputs = np.asarray(stage["output_bits"], dtype=object)
            if inputs.shape != (1, 30, 256) or outputs.shape != inputs.shape or any(type(bits) is not int or not 0 <= bits <= 0xFFFFFFFF for bits in [*inputs.reshape(-1), *outputs.reshape(-1)]):
                raise ValueError("Malformed rotary float32 table")
            matches = matches and _sha(stage["input_bits"]) == plan["angle_bits_sha256"]
            expected = np.asarray([encode_bfloat16_rne(*decode_finite_float32(bits)) for bits in outputs.reshape(-1)], dtype=np.uint16).reshape(1, 30, 256)
            casts = casts and np.array_equal(expected, _state_array(record[name], [1, 30, 256]))
            controls = controls and np.array_equal(_state_array(record["untraced"][name], [1, 30, 256]), _state_array(record[name], [1, 30, 256]))
    repeated = all(canonical_json({k: v for k, v in record.items() if k != "profiled_cuda_event_names"}) == canonical_json({k: v for k, v in bundle["repetitions"][0].items() if k != "profiled_cuda_event_names"}) for record in bundle["repetitions"])
    return _seal({"schema_version": 1, "scope": TABLE_SCOPE, "plan_sha256": plan["plan_sha256"], "bundle_sha256": _sha(bundle),
                  "runtime": runtime, "independent_angles_match": matches, "independent_bfloat16_casts_match": casts,
                  "table_repetitions_identical": repeated, "untraced_output_controls_match": controls, "runtime_matches_plan": canonical_json(runtime) == canonical_json(plan["runtime"]),
                  "table_usable": matches and casts and repeated and controls and canonical_json(runtime) == canonical_json(plan["runtime"]),
                  "table_descriptors": {name: _descriptor(_state_array(bundle["repetitions"][0][name], [1, 30, 256])) for name in ("cosine", "sine")},
                  "profiled_cuda_event_names": [record["profiled_cuda_event_names"] for record in bundle["repetitions"]],
                  "native_trigonometric_arithmetic_reconstructed": False, "unseen_positions_allowed": False,
                  "global_exactness_activation_allowed": False}, "manifest_sha256")


def acquire_rotary_table(program: dict[str, Any], model: Any, plan: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    from .gemma_reduction_backend import _profile_call

    if canonical_json(plan) != canonical_json(build_rotary_table_plan(program, model)):
        raise ValueError("Frozen rotary table plan mismatch")
    module, frequency, _ = _rotary_context(program, model)
    positions = torch.arange(30, dtype=torch.int64, device=frequency.device).unsqueeze(0)
    value = torch.zeros((1, 30, 640), dtype=torch.bfloat16, device=frequency.device)
    repetitions, captured = [], []

    def invoke() -> Any:
        result, stages = _observe_rotary(module, value, positions)
        captured[:] = [stages]
        return result

    for _ in range(3):
        output, events = _profile_call(invoke)
        cosine, sine = output
        if cosine.dtype != torch.bfloat16 or sine.dtype != torch.bfloat16 or list(cosine.shape) != [1, 30, 256] or sine.shape != cosine.shape:
            raise ValueError("Original rotary output type mismatch")
        plain = module(value, positions)
        repetitions.append({"cosine": _bits(cosine).tolist(), "sine": _bits(sine).tolist(), "stages": captured[0],
                            "untraced": {name: _bits(tensor).tolist() for name, tensor in zip(("cosine", "sine"), plain)}, "profiled_cuda_event_names": events})
    bind_model_tensors(program, model, verify_hashes=True)
    bundle = {"repetitions": repetitions}
    return _table_manifest(plan, bundle, _runtime()), bundle


def check_rotary_table(program: dict[str, Any], plan: dict[str, Any], manifest: dict[str, Any], bundle: dict[str, Any]) -> None:
    from .reference_gemma import _rms_code_commitment

    _check_hash(plan, "plan_sha256")
    _check_hash(manifest, "manifest_sha256")
    position, table, _ = _rotary_instructions(program)
    if plan["program_sha256"] != program["program_sha256"] or plan["position_instruction_sha256"] != position["instruction_sha256"] or plan["table_instruction_sha256"] != table["instruction_sha256"] or plan["frequency"]["sha256"] != program["parameter_commitments"]["model.rotary_emb_local.inv_freq"]["sha256"]:
        raise ValueError("Rotary table IR/frequency binding mismatch")
    if plan["positions"] != list(range(30)) or plan["attention_scaling"] != 1.0 or plan["scope"] != TABLE_SCOPE or plan["output_shape"] != [1, 30, 256] or plan["input_shape"] != [1, 30, 640] or plan["repetitions"] != 3 or plan["finite_arithmetic_sha256"] != _rms_code_commitment() or plan["angle_code_sha256"] != hashlib.sha256(inspect.getsource(rotary_angle_bits).encode("utf-8")).hexdigest():
        raise ValueError("Rotary table domain/profile mismatch")
    if plan["model_binding"]["parameter_commitments_sha256"] != _sha(program["parameter_commitments"]) or plan["frequency"]["dtype"] != "torch.float32" or plan["frequency"]["shape"] != [128] or any(type(position) is not int for position in plan["positions"]):
        raise ValueError("Rotary table parameter/type commitment mismatch")
    if plan["acquisition_schedule"] != "one warm-up, one profiled/traced module call, and one untraced output control per repetition":
        raise ValueError("Rotary table acquisition protocol mismatch")
    if any(plan.get(key) is not False for key in ("native_trigonometric_arithmetic_reconstructed", "unseen_positions_allowed", "global_exactness_activation_allowed")):
        raise ValueError("Rotary table scope overclaim")
    expected = _table_manifest(plan, bundle, manifest["runtime"])
    if canonical_json(expected) != canonical_json(manifest) or manifest["table_usable"] is not True:
        raise ValueError("Rotary table evidence is inconsistent or unusable")


def rotate_bfloat16_bits(values: np.ndarray, cosine: np.ndarray, sine: np.ndarray) -> np.ndarray:
    from .gemma_float_semantics import bfloat16_add_bits, bfloat16_multiply_bits

    if values.dtype != np.uint16 or values.shape not in ((1, 4, 30, 256), (1, 1, 30, 256)) or cosine.dtype != np.uint16 or sine.dtype != np.uint16 or cosine.shape != (1, 30, 256) or sine.shape != cosine.shape:
        raise ValueError("Unsupported independent rotary tensor geometry")
    output = np.empty_like(values)
    for head in range(values.shape[1]):
        for position in range(30):
            for dimension in range(256):
                other = dimension + 128 if dimension < 128 else dimension - 128
                rotated = int(values[0, head, position, other]) ^ (0x8000 if dimension < 128 else 0)
                first = bfloat16_multiply_bits(int(values[0, head, position, dimension]), int(cosine[0, position, dimension]))
                second = bfloat16_multiply_bits(rotated, int(sine[0, position, dimension]))
                output[0, head, position, dimension] = bfloat16_add_bits(first, second)
    return output


def _rotation_code_sha() -> str:
    from . import gemma_float_semantics
    return _sha({"rotation": inspect.getsource(rotate_bfloat16_bits), "scalar_module": inspect.getsource(gemma_float_semantics), "entry": _entry_code_sha()})


def _rotary_records(program: dict[str, Any], entry_plan: dict[str, Any], tables: dict[str, np.ndarray], rotated: dict[str, np.ndarray], table_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    import torch

    position, table, rotation = _rotary_instructions(program)
    records = []
    append_chain_record(records, {"instruction_id": position["id"], "instruction_sha256": position["instruction_sha256"], "provider": "independent_position_indices_v1", "outputs": {"position_ids": tensor_descriptor(torch.arange(30, dtype=torch.int64).unsqueeze(0))}})
    for record in entry_plan["records"]:
        append_chain_record(records, record["payload"])
    append_chain_record(records, {"instruction_id": table["id"], "instruction_sha256": table["instruction_sha256"], "provider": "compiled_fixed_position_rotary_table_v1", "inputs": table["inputs"], "parameter_refs": table["parameter_refs"], "table_manifest_sha256": table_manifest["manifest_sha256"], "outputs": {"rotary.local." + name: _descriptor(value) for name, value in tables.items()}})
    append_chain_record(records, {"instruction_id": rotation["id"], "instruction_sha256": rotation["instruction_sha256"], "provider": "bfloat16_mul_mul_add_rne_rotary_v1", "inputs": rotation["inputs"], "outputs": {name: _descriptor(value) for name, value in rotated.items()}})
    return records


def build_rotary_slice(program: dict[str, Any], model: Any, fixture: dict[str, Any], lookup: CheckedRsqrtLookup, table_plan: dict[str, Any], table_manifest: dict[str, Any], table_bundle: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    code_sha256 = _rotation_code_sha()
    check_rotary_table(program, table_plan, table_manifest, table_bundle)
    if canonical_json(table_plan) != canonical_json(build_rotary_table_plan(program, model)):
        raise ValueError("Rotary table belongs to another model/runtime")
    entry_plan, entry_bundle = build_attention_entry_plan(program, model, fixture, lookup)
    states = check_attention_entry_plan(program, entry_plan, entry_bundle, lookup)
    _, _, rotation = _rotary_instructions(program)
    tables = {name: _state_array(table_bundle["repetitions"][0][name], [1, 30, 256]) for name in ("cosine", "sine")}
    rotated = {output: rotate_bfloat16_bits(states[input_name], tables["cosine"], tables["sine"]) for output, input_name in zip(ROTATED, rotation["inputs"][:2])}
    records = _rotary_records(program, entry_plan, tables, rotated, table_manifest)
    if _rotation_code_sha() != code_sha256:
        raise ValueError("Rotary numerical source changed during prediction; no plan may be frozen")
    bundle = {"entry_plan": entry_plan, "entry_bundle": entry_bundle, "rotated_bits": {name: value.tolist() for name, value in rotated.items()}}
    body = {"schema_version": 1, "scope": SCOPE, "program_sha256": program["program_sha256"], "entry_plan_sha256": entry_plan["plan_sha256"],
            "table_plan_sha256": table_plan["plan_sha256"], "table_manifest_sha256": table_manifest["manifest_sha256"],
            "records": records, "execution_root": records[-1]["record_hash"], "bundle_sha256": _sha(bundle),
            "rotation_code_sha256": code_sha256, "runtime": _runtime(), "target_names": [*ROTATED, "layer.0.value.heads"],
            "rotation_value_count": 38400, "target_value_count": 46080, "instruction_count": 14,
            "derivation_order": "positions, pre-RoPE slice, compiled table, rotation; dependency-consistent witness, not a native instruction timeline",
            "native_trigonometric_arithmetic_reconstructed": False, "attention_arithmetic_compared": False,
            "full_first_layer_qualified": False, "global_exactness_activation_allowed": False}
    return _seal(body, "plan_sha256"), bundle


def check_rotary_slice(program: dict[str, Any], plan: dict[str, Any], bundle: dict[str, Any], lookup: CheckedRsqrtLookup, table_plan: dict[str, Any], table_manifest: dict[str, Any], table_bundle: dict[str, Any]) -> dict[str, np.ndarray]:
    check_rotary_table(program, table_plan, table_manifest, table_bundle)
    _check_hash(plan, "plan_sha256")
    states = check_attention_entry_plan(program, bundle["entry_plan"], bundle["entry_bundle"], lookup)
    if plan["scope"] != SCOPE or plan["program_sha256"] != program["program_sha256"] or plan["entry_plan_sha256"] != bundle["entry_plan"]["plan_sha256"] or plan["table_plan_sha256"] != table_plan["plan_sha256"] or plan["table_manifest_sha256"] != table_manifest["manifest_sha256"] or plan["bundle_sha256"] != _sha(bundle) or plan["rotation_code_sha256"] != _rotation_code_sha():
        raise ValueError("Rotary slice source commitment mismatch")
    if any(plan.get(key) is not False for key in ("native_trigonometric_arithmetic_reconstructed", "attention_arithmetic_compared", "full_first_layer_qualified", "global_exactness_activation_allowed")) or plan["target_names"] != [*ROTATED, "layer.0.value.heads"] or plan["rotation_value_count"] != 38400 or plan["target_value_count"] != 46080 or plan["instruction_count"] != 14:
        raise ValueError("Rotary slice scope mismatch")
    if not verify_trace_chain(plan["records"], plan["execution_root"])["valid"] or len(plan["records"]) != 14 or set(bundle["rotated_bits"]) != set(ROTATED):
        raise ValueError("Rotary slice witness coverage mismatch")
    if any(canonical_json(runtime) != canonical_json(plan["runtime"]) for runtime in (bundle["entry_plan"]["runtime"], table_plan["runtime"], table_manifest["runtime"])) or canonical_json(bundle["entry_plan"]["model_binding"]) != canonical_json(table_plan["model_binding"]):
        raise ValueError("Rotary and pre-RoPE evidence disagree on model/runtime")
    cosine, sine = [_state_array(table_bundle["repetitions"][0][name], [1, 30, 256]) for name in ("cosine", "sine")]
    for output, input_name in zip(ROTATED, ("layer.0.query.normalized", "layer.0.key.normalized")):
        expected = rotate_bfloat16_bits(states[input_name], cosine, sine)
        actual = _state_array(bundle["rotated_bits"][output], list(expected.shape))
        if not np.array_equal(expected, actual):
            raise ValueError("Rotary bits differ from independent recomputation")
        states[output] = actual
    expected_records = _rotary_records(program, bundle["entry_plan"], {"cosine": cosine, "sine": sine}, {name: states[name] for name in ROTATED}, table_manifest)
    if canonical_json(plan["records"]) != canonical_json(expected_records) or plan["derivation_order"] != "positions, pre-RoPE slice, compiled table, rotation; dependency-consistent witness, not a native instruction timeline":
        raise ValueError("Rotary witness does not match its declared providers and dependencies")
    return states


def _rotary_report(plan: dict[str, Any], predicted: dict[str, np.ndarray], observations: list[dict[str, Any]], runtime: dict[str, Any], table_manifest: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(observations, list) or len(observations) != 3:
        raise ValueError("Expected three original rotary-prefix observations")
    targets = [*ROTATED, "layer.0.value.heads"]
    mismatches, table_match = [], True
    for repetition, observed in enumerate(observations):
        if set(observed["attention_inputs"]) != set(targets):
            raise ValueError("Incomplete original attention input capture")
        for name in targets:
            actual = _state_array(observed["attention_inputs"][name], list(predicted[name].shape))
            for coordinate in np.argwhere(actual != predicted[name]):
                mismatches.append({"repetition": repetition, "tensor": name, "coordinate": coordinate.tolist(), "predicted_bits": int(predicted[name][tuple(coordinate)]), "observed_bits": int(actual[tuple(coordinate)])})
        for name in ("cosine", "sine"):
            actual = _state_array(observed["tables"][name], [1, 30, 256])
            table_match = table_match and _descriptor(actual)["sha256"] == table_manifest["table_descriptors"][name]["sha256"]
    scope_match = canonical_json(runtime) == canonical_json(plan["runtime"])
    body = {"schema_version": 1, "scope": SCOPE, "plan_sha256": plan["plan_sha256"], "execution_root": plan["execution_root"],
            "observations": observations, "runtime": runtime, "runtime_matches_plan": scope_match, "original_tables_match_specification": table_match,
            "mismatch_count": len(mismatches), "mismatches": mismatches, "first_divergence": "rotary_table" if not table_match else mismatches[0] if mismatches else None,
            "rotated_attention_inputs_bit_exact": not mismatches and table_match and scope_match,
            "target_value_count": 46080, "attention_arithmetic_executed": False,
            "native_trigonometric_arithmetic_reconstructed": False, "full_first_layer_qualified": False,
            "hardware_semantics_established": False, "global_exactness_activation_allowed": False}
    return _seal(body, "report_sha256")


def acquire_rotary_slice(program: dict[str, Any], model: Any, plan: dict[str, Any], bundle: dict[str, Any], lookup: CheckedRsqrtLookup, table_plan: dict[str, Any], table_manifest: dict[str, Any], table_bundle: dict[str, Any]) -> dict[str, Any]:
    import torch

    predicted = check_rotary_slice(program, plan, bundle, lookup, table_plan, table_manifest, table_bundle)
    if canonical_json(table_plan) != canonical_json(build_rotary_table_plan(program, model)) or canonical_json(_runtime()) != canonical_json(plan["runtime"]):
        raise ValueError("Original rotary model/runtime differs from the prediction plan")
    target = model.model.layers[0].self_attn
    namespace = importlib.import_module(type(target).__module__)
    original_attention = namespace.eager_attention_forward
    ids = torch.tensor(bundle["entry_plan"]["input_token_ids"], dtype=torch.int64, device=model.model.embed_tokens.weight.device)
    observations = []

    class AttentionBoundary(Exception):
        pass

    for _ in range(3):
        captured, stopped = {}, False

        def capture_table(module: Any, args: Any, output: Any) -> None:
            if "tables" in captured or args[1].tolist() != [list(range(30))]:
                raise ValueError("Original rotary position/occurrence mismatch")
            captured["tables"] = {name: _bits(value).tolist() for name, value in zip(("cosine", "sine"), output)}

        def capture_attention(module: Any, query: Any, key: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
            if module is not target:
                return original_attention(module, query, key, value, *args, **kwargs)
            if "attention_inputs" in captured:
                raise ValueError("Repeated first-layer attention boundary")
            captured["attention_inputs"] = {name: _bits(tensor).tolist() for name, tensor in zip((*ROTATED, "layer.0.value.heads"), (query, key, value))}
            raise AttentionBoundary()

        handle = model.model.rotary_emb_local.register_forward_hook(capture_table)
        try:
            with patch.object(namespace, "eager_attention_forward", side_effect=capture_attention), torch.no_grad():
                model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1)
        except AttentionBoundary:
            stopped = True
        finally:
            handle.remove()
        if not stopped or set(captured) != {"tables", "attention_inputs"}:
            raise ValueError("Original forward did not stop at the complete attention input boundary")
        observations.append(captured)
    bind_model_tensors(program, model, verify_hashes=True)
    return _rotary_report(plan, predicted, observations, _runtime(), table_manifest)


def verify_rotary_slice(program: dict[str, Any], plan: dict[str, Any], bundle: dict[str, Any], report: dict[str, Any], lookup: CheckedRsqrtLookup, table_plan: dict[str, Any], table_manifest: dict[str, Any], table_bundle: dict[str, Any]) -> dict[str, Any]:
    try:
        predicted = check_rotary_slice(program, plan, bundle, lookup, table_plan, table_manifest, table_bundle)
        expected = _rotary_report(plan, predicted, report["observations"], report["runtime"], table_manifest)
        return {"valid": canonical_json(report) == canonical_json(expected), "mode": "integrity_only_with_rotary_arithmetic_recomputation",
                "rotated_attention_inputs_bit_exact": expected["rotated_attention_inputs_bit_exact"], "mismatch_count": expected["mismatch_count"],
                "first_divergence": expected["first_divergence"], "global_exactness_activation_allowed": False}
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError, StopIteration) as error:
        return {"valid": False, "reason": str(error)}


def rotary_slice_summary(plan: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    body = {k: v for k, v in report.items() if k not in ("observations", "mismatches", "report_sha256")}
    body.update({"source_report_sha256": report["report_sha256"], "program_sha256": plan["program_sha256"], "entry_plan_sha256": plan["entry_plan_sha256"],
                 "table_manifest_sha256": plan["table_manifest_sha256"], "instruction_count": 14,
                 "observed_target_hashes": {name: [_descriptor(_state_array(observed["attention_inputs"][name], [1, 4 if name == ROTATED[0] else 1, 30, 256]))["sha256"] for observed in report["observations"]] for name in plan["target_names"]}})
    return _seal(body, "summary_sha256")
