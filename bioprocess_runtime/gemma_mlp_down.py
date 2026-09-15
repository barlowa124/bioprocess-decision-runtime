from __future__ import annotations

import hashlib
import inspect
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from .gemma_attention_entry import _bits, _descriptor, _state_array
from .gemma_mlp_product import ProductSources, verify_product, product_summary, _model_context as product_model_context, _code_sha as product_code_sha
from .gemma_k2048_probes import candidates, verify_probes, _code_sha as probe_code_sha
from .gemma_float_semantics import decode_finite_bfloat16, encode_bfloat16_rne
from .gemma_wmma_candidate import _operand_aligned_accumulator, _merge_split_partials, OPERAND_ALIGNMENT_PROFILE
from .gemma_ir_interpreter import bind_model_tensors
from .gemma_rotary_slice import _sha, _seal, _check_hash
from .gemma_rsqrt_lookup import _runtime
from .serialization import canonical_json

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
WEIGHT = "model.layers.0.mlp.down_proj.weight"
INPUT_SHAPE = (1, 30, 2048)
WEIGHT_SHAPE = (640, 2048)
OUTPUT_SHAPE = (1, 30, 640)
SUPPORTED_ID = "k192:bfloat16_rne:sequential_float32_rne"
SUPPORTED_CANDIDATE_JSON = canonical_json(next(item for item in candidates() if item["id"] == SUPPORTED_ID))
SCOPE = "Fixed-case original eager MLP down projection from the reused verified product boundary; unchanged unique controlled K2048 candidate hypothesis, not hardware partition/accumulator proof, new-prompt holdout, or full-layer qualification. The original first down output was already observed; this v2 metadata correction is a development regression, not a fresh down holdout. Kernel provenance is distinct-symbol-set evidence only; launch order is not established. The prefix is not independently recomputed."
KERNEL_PROVENANCE = "distinct-symbol-set-only; launch order not established"
FALSE_FLAGS = ("candidate_refitting_allowed", "shape_transfer_prequalified", "hardware_partitioning_established", "hardware_semantics_established", "split_boundaries_observed", "intermediate_values_observed", "prefix_independently_recomputed", "fresh_prompt_holdout", "fresh_down_holdout", "full_first_layer_qualified", "global_exactness_activation_allowed", "post_feedforward_layernorm_executed")
ORDER = ["gate", "activation", "up", "down_input", "down_output"]
_WORKER_WEIGHTS: list[list[int]] = []
_WORKER_CANDIDATE: dict[str, Any] = {}


def _code_sha() -> str:
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("MLP-down source changed after import")
    return _sha({"module": SOURCE_SHA256, "product": product_code_sha(), "controlled_probe": probe_code_sha()})


def _supported_candidate(candidate: dict[str, Any]) -> None:
    if canonical_json(candidate) != SUPPORTED_CANDIDATE_JSON:
        raise ValueError("Unsupported unchanged K2048 down candidate")


@dataclass(frozen=True)
class DownSources:
    product: ProductSources
    product_plan: dict[str, Any]
    product_bundle: dict[str, Any]
    product_report: dict[str, Any]
    probe_binding: dict[str, Any]
    probe_plan: dict[str, Any]
    probe_bundle: dict[str, Any]
    probe_report: dict[str, Any]

    @property
    def program(self) -> dict[str, Any]:
        return self.product.entry.post.program

    def material(self) -> tuple[np.ndarray, dict[str, Any]]:
        checked = verify_product(self.product, self.product_plan, self.product_bundle, self.product_report)
        if not checked["valid"] or checked.get("activation_and_product_match") is not True:
            raise ValueError("MLP down requires passing full product lineage")
        summary = product_summary(self.product_plan, self.product_report)
        checked = verify_probes(self.probe_binding, summary, self.probe_plan, self.probe_bundle, self.probe_report)
        if not checked["valid"] or checked.get("survivors_supported_in_declared_scope") is not True or checked.get("unique_survivor_in_frozen_grid") is not True or checked.get("surviving_candidates") != [SUPPORTED_ID]:
            raise ValueError("MLP down requires one supported controlled-grid source survivor")
        if self.probe_plan["input_shape"] != list(INPUT_SHAPE) or self.probe_plan["weight_shape"] != list(WEIGHT_SHAPE) or self.probe_plan["matrix_value_count"] != 19200:
            raise ValueError("MLP down requires declared K2048 dimensions")
        down_instructions(self.program)
        candidate = next(item for item in self.probe_plan["candidates"] if item["id"] == SUPPORTED_ID)
        _supported_candidate(candidate)
        return _state_array(self.product_bundle["product"], list(INPUT_SHAPE)), candidate

    def commitments(self) -> dict[str, Any]:
        return {"program_sha256": self.program["program_sha256"], "product_sources": self.product.commitments(),
                "product_plan_sha256": self.product_plan["plan_sha256"], "product_bundle_sha256": _sha(self.product_bundle),
                "product_report_sha256": self.product_report["report_sha256"], "product_summary_sha256": product_summary(self.product_plan, self.product_report)["summary_sha256"],
                "probe_binding_sha256": self.probe_binding["binding_sha256"], "probe_plan_sha256": self.probe_plan["plan_sha256"],
                "probe_bundle_sha256": _sha(self.probe_bundle), "probe_report_sha256": self.probe_report["report_sha256"]}


def down_instructions(program: dict[str, Any]) -> list[dict[str, Any]]:
    found = [item for item in program["instructions"] if item["outputs"] == ["layer.0.mlp.down"]]
    if len(found) != 1:
        raise ValueError("Expected unique MLP-down IR node")
    node = found[0]
    if node["opcode"] != "LINEAR" or node["inputs"] != ["layer.0.mlp.product"] or node["parameter_refs"] != [WEIGHT] or node["attributes"] or type(node["layer"]) is not int or node["layer"] != 0:
        raise ValueError("Unsupported MLP-down IR binding/bias")
    _check_hash(node, "instruction_sha256")
    for name, shape in (("layer.0.mlp.product", INPUT_SHAPE), ("layer.0.mlp.down", OUTPUT_SHAPE)):
        declaration = program["tensors"][name]
        resolved = [{"B": 1, "S": 30}.get(value, value) for value in declaration["shape"]]
        if resolved != list(shape) or declaration["dtype"] != "torch.bfloat16":
            raise ValueError("Unsupported MLP-down IR tensor declaration")
    weight = program["parameter_commitments"][WEIGHT]
    if weight["shape"] != list(WEIGHT_SHAPE) or weight["dtype"] != "torch.bfloat16" or weight["numel"] != 1310720 or program["tensors"]["layer.0.mlp.down"]["producer"] != node["id"] or program["configuration"]["intermediate_size"] != 2048:
        raise ValueError("Unsupported MLP-down weight/output declaration")
    return found


def down_dot(left: list[int], right: list[int], candidate: dict[str, Any]) -> int:
    _supported_candidate(candidate)
    if len(left) != 2048 or len(right) != 2048:
        raise ValueError("MLP-down dot requires declared K2048 operands")
    partials = [_operand_aligned_accumulator(left[start:stop], right[start:stop]) for start, stop in candidate["partitions"]]
    rounded = [decode_finite_bfloat16(encode_bfloat16_rne(value))[0] for value in partials]
    return _merge_split_partials(rounded, candidate["merge"])


def _worker_init(weights: list[list[int]], candidate: dict[str, Any], expected_code: str) -> None:
    global _WORKER_WEIGHTS, _WORKER_CANDIDATE
    if _code_sha() != expected_code:
        raise ValueError("MLP-down prediction worker source mismatch")
    _supported_candidate(candidate)
    _WORKER_WEIGHTS, _WORKER_CANDIDATE = weights, candidate


def _worker_row(task: tuple[int, list[int]]) -> tuple[int, list[int]]:
    index, values = task
    return index, [down_dot(values, weight, _WORKER_CANDIDATE) for weight in _WORKER_WEIGHTS]


def project_down_bits(product: np.ndarray, weights: np.ndarray, candidate: dict[str, Any], workers: int = 1) -> np.ndarray:
    code = _code_sha()
    _supported_candidate(candidate)
    if type(workers) is not int or not 1 <= workers <= 4 or product.dtype != np.uint16 or product.shape != INPUT_SHAPE or weights.dtype != np.uint16 or weights.shape != WEIGHT_SHAPE:
        raise ValueError("Unsupported MLP-down projection geometry or worker count")
    rows = weights.tolist()
    tasks = list(enumerate(product[0].tolist()))
    output = np.empty(OUTPUT_SHAPE, dtype=np.uint16)
    if workers == 1:
        for index, values in tasks:
            output[0, index] = [down_dot(values, weight, candidate) for weight in rows]
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(rows, candidate, code)) as pool:
            for index, values in pool.map(_worker_row, tasks):
                output[0, index] = values
    if _code_sha() != code:
        raise ValueError("MLP-down source changed during prediction")
    return output


def _model_context(sources: DownSources, model: Any) -> dict[str, Any]:
    import torch
    from transformers.models.gemma3 import modeling_gemma3 as gemma
    if torch.nn.functional.linear is not torch._C._nn.linear:
        raise ValueError("Original MLP-down F.linear implementation was replaced")
    product_model_context(sources.product, model)
    parameters = bind_model_tensors(sources.program, model, verify_hashes=True)
    layer = model.model.layers[0]
    modules = [(model, gemma.Gemma3ForCausalLM), (model.model, gemma.Gemma3TextModel), (layer, gemma.Gemma3DecoderLayer), (layer.mlp, gemma.Gemma3MLP),
               (layer.mlp.down_proj, torch.nn.Linear), (layer.mlp.gate_proj, torch.nn.Linear), (layer.mlp.up_proj, torch.nn.Linear),
               (layer.post_feedforward_layernorm, gemma.Gemma3RMSNorm)]
    for module, cls in modules:
        if type(module) is not cls or getattr(module.forward, "__func__", None) is not cls.forward:
            raise ValueError("Original MLP-down model method implementation identity mismatch")
    down = layer.mlp.down_proj
    if down.bias is not None or down.weight is not parameters[WEIGHT] or tuple(down.weight.shape) != WEIGHT_SHAPE or down.weight.stride() != (2048, 1) or down.weight.dtype != torch.bfloat16 or down.weight.device != model.model.embed_tokens.weight.device or down.in_features != 2048 or down.out_features != 640:
        raise ValueError("Original MLP-down parameter binding mismatch")
    return parameters


def _implementation_commitment() -> dict[str, str]:
    import torch
    from transformers.models.gemma3 import modeling_gemma3 as gemma
    return {cls.__name__: hashlib.sha256(inspect.getsource(cls.forward).encode("utf-8")).hexdigest()
            for cls in (gemma.Gemma3ForCausalLM, gemma.Gemma3TextModel, gemma.Gemma3DecoderLayer, gemma.Gemma3MLP, torch.nn.Linear, gemma.Gemma3RMSNorm)}


def _kernel_symbols(names: Any) -> list[str]:
    if not isinstance(names, list) or not names or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Kernel provenance requires a nonempty list of unique nonempty symbol strings")
    return sorted(names)


def _plan_body(sources: DownSources, inputs: np.ndarray, candidate: dict[str, Any], weights: np.ndarray, prediction: np.ndarray, bundle: dict[str, Any]) -> dict[str, Any]:
    _kernel_symbols(sources.probe_plan["expected_kernel_names"])
    return {"schema_version": 1, "scope": SCOPE, "sources": sources.commitments(), "instructions": down_instructions(sources.program),
            "code_sha256": _code_sha(), "implementation_commitment": _implementation_commitment(), "runtime": sources.product_plan["runtime"],
            "model_binding": sources.product_plan["model_binding"], "input": _descriptor(inputs), "weight": _descriptor(weights), "prediction": _descriptor(prediction),
            "input_shape": list(INPUT_SHAPE), "weight_shape": list(WEIGHT_SHAPE), "output_shape": list(OUTPUT_SHAPE), "bias": None,
            "candidate": candidate, "partial_profile": dict(OPERAND_ALIGNMENT_PROFILE), "expected_kernel_names": sources.probe_plan["expected_kernel_names"],
            "bundle_sha256": _sha(bundle), "value_count": 19200, "repetitions": 3, "prefix_boundary_reused": True,
            "kernel_provenance": KERNEL_PROVENANCE, "metadata_correction_development_regression": True,
            "down_stage_previously_acquired_by_prefix_workflow": True, "uses_empirical_gelu_specification": True,
            **{key: False for key in FALSE_FLAGS}}


def build_down_plan(sources: DownSources, model: Any, workers: int = 1) -> tuple[dict[str, Any], dict[str, Any]]:
    code = _code_sha()
    inputs, candidate = sources.material()
    parameters = _model_context(sources, model)
    weights = _bits(parameters[WEIGHT])
    prediction = project_down_bits(inputs, weights, candidate, workers)
    bundle = {"weight_bits": weights.tolist(), "down_bits": prediction.tolist()}
    _model_context(sources, model)
    if _code_sha() != code:
        raise ValueError("MLP-down source changed during prediction")
    return _seal(_plan_body(sources, inputs, candidate, weights, prediction, bundle), "plan_sha256"), bundle


def check_down_plan(sources: DownSources, plan: dict[str, Any], bundle: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _check_hash(plan, "plan_sha256")
    inputs, candidate = sources.material()
    if set(bundle) != {"weight_bits", "down_bits"}:
        raise ValueError("Unexpected MLP-down prediction payload")
    weights = _state_array(bundle["weight_bits"], list(WEIGHT_SHAPE))
    prediction = _state_array(bundle["down_bits"], list(OUTPUT_SHAPE))
    if _descriptor(weights)["sha256"] != sources.program["parameter_commitments"][WEIGHT]["sha256"]:
        raise ValueError("MLP-down checkpoint weight commitment mismatch")
    expected = _seal(_plan_body(sources, inputs, candidate, weights, prediction, bundle), "plan_sha256")
    if canonical_json(plan) != canonical_json(expected):
        raise ValueError("MLP-down plan commitment/scope mismatch")
    return inputs, weights, prediction


def _capture_down(model: Any, token_ids: list[list[int]], traced: bool) -> dict[str, Any]:
    import torch
    from torch.profiler import profile, ProfilerActivity

    layer = model.model.layers[0]
    mlp, down = layer.mlp, layer.mlp.down_proj
    original_linear = torch.nn.functional.linear
    record, handles = {"order": [], "postnorm_executed": False}, []
    active, linear_input, linear_output = False, None, None

    class Complete(Exception):
        pass

    def geometry(value: Any) -> dict[str, Any]:
        return {"shape": list(value.shape), "dtype": str(value.dtype), "strides": list(value.stride()), "alignment_mod16": value.data_ptr() % 16, "device": str(value.device)}

    def boundary(role: str):
        def capture(module: Any, args: Any, output: Any) -> None:
            if record["order"] != ORDER[:ORDER.index(role)]:
                raise ValueError("Unexpected original down-prefix occurrence/order")
            record["order"].append(role)
        return capture

    def before_down(module: Any, args: Any) -> None:
        nonlocal active, linear_input
        if record["order"] != ORDER[:3] or active or len(args) != 1:
            raise ValueError("Unexpected original down input occurrence/order")
        active, linear_input = True, args[0]
        record["order"].append("down_input")
        record["input_bits"] = _bits(args[0]).tolist()
        record["input_geometry"] = geometry(args[0])
        record["weight"] = _descriptor(_bits(down.weight))
        record["weight_geometry"] = geometry(down.weight)

    def linear(input: Any, weight: Any, bias: Any = None) -> Any:
        nonlocal linear_output
        if not active:
            return original_linear(input, weight, bias)
        if "kernel_names" in record or record["order"] != ORDER[:4] or bias is not None:
            raise ValueError("Unexpected down F.linear occurrence/order/bias")
        record["weight_pointer_matches"] = weight is down.weight and weight.data_ptr() == down.weight.data_ptr()
        record["linear_input_matches_hook"] = input is linear_input and input.data_ptr() == linear_input.data_ptr()
        record["linear_input"] = _descriptor(_bits(input))
        record["linear_weight"] = _descriptor(_bits(weight))
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as captured:
            result = original_linear(input, weight, bias)
            torch.cuda.synchronize()
        record["kernel_names"] = sorted({event.name for event in captured.events() if event.device_type == torch.autograd.DeviceType.CUDA})
        record["linear_output"] = _descriptor(_bits(result))
        record["linear_geometry"] = {"input": geometry(input), "weight": geometry(weight), "output": geometry(result)}
        linear_output = result
        return result

    def after_down(module: Any, args: Any, output: Any) -> None:
        nonlocal active
        if record["order"] != ORDER[:4] or not active:
            raise ValueError("Unexpected original down output occurrence/order")
        if traced:
            record["linear_output_matches_hook"] = output is linear_output and output.data_ptr() == linear_output.data_ptr() if linear_output is not None else False
        record["output_bits"] = _bits(output).tolist()
        record["output_geometry"] = geometry(output)
        record["order"].append("down_output")
        active = False
        raise Complete()

    def postnorm(module: Any, args: Any) -> None:
        record["postnorm_executed"] = True
        raise ValueError("Post-feedforward layernorm must not execute")

    stopped = False
    try:
        handles.append(mlp.gate_proj.register_forward_hook(boundary("gate")))
        handles.append(mlp.act_fn.register_forward_hook(boundary("activation")))
        handles.append(mlp.up_proj.register_forward_hook(boundary("up")))
        handles.append(down.register_forward_pre_hook(before_down))
        handles.append(down.register_forward_hook(after_down))
        handles.append(layer.post_feedforward_layernorm.register_forward_pre_hook(postnorm))
        ids = torch.tensor(token_ids, dtype=torch.int64, device=down.weight.device)
        with torch.no_grad():
            if traced:
                with patch.object(torch.nn.functional, "linear", side_effect=linear):
                    model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1)
            else:
                model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1)
    except Complete:
        stopped = True
    finally:
        for handle in handles:
            handle.remove()
    if not stopped or record["order"] != ORDER or (traced and "kernel_names" not in record):
        raise ValueError("Original forward did not stop after the down output")
    return record


def _geometry_matches(geometry: dict[str, Any], shape: tuple[int, ...], strides: list[int]) -> bool:
    return geometry.get("shape") == list(shape) and geometry.get("dtype") == "torch.bfloat16" and geometry.get("strides") == strides and geometry.get("alignment_mod16") == 0 and str(geometry.get("device", "")).startswith("cuda:")


def down_report(plan: dict[str, Any], inputs: np.ndarray, weights: np.ndarray, prediction: np.ndarray, observations: list[dict[str, Any]], runtime: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(observations, list) or len(observations) != 3:
        raise ValueError("Expected three original down traced/untraced observations")
    if plan.get("kernel_provenance") != KERNEL_PROVENANCE:
        raise ValueError("MLP-down kernel provenance must not establish launch order")
    expected_symbols = _kernel_symbols(plan["expected_kernel_names"])
    mismatches, checks = [], []
    for repetition, pair in enumerate(observations):
        traced, plain = pair["traced"], pair["untraced"]
        observed_symbols = _kernel_symbols(traced["kernel_names"])
        actual = _state_array(traced["output_bits"], list(OUTPUT_SHAPE))
        for coordinate in np.argwhere(actual != prediction):
            index = tuple(coordinate)
            mismatches.append({"repetition": repetition, "coordinate": coordinate.tolist(), "predicted_bits": int(prediction[index]), "observed_bits": int(actual[index])})
        checks.append({"source_input_matches": all(np.array_equal(_state_array(record["input_bits"], list(INPUT_SHAPE)), inputs) for record in (traced, plain)),
                       "untraced_output_matches": np.array_equal(_state_array(plain["output_bits"], list(OUTPUT_SHAPE)), actual),
                       "checkpoint_weight_matches": all(canonical_json(record["weight"]) == canonical_json(_descriptor(weights)) for record in (traced, plain)),
                       "declared_order_matches": all(record["order"] == ORDER for record in (traced, plain)),
                       "postnorm_not_executed": all(record["postnorm_executed"] is False for record in (traced, plain)),
                       "boundary_geometry_matches": all(_geometry_matches(record[role + "_geometry"], shape, strides) for record in (traced, plain) for role, shape, strides in (("input", INPUT_SHAPE, [61440, 2048, 1]), ("weight", WEIGHT_SHAPE, [2048, 1]), ("output", OUTPUT_SHAPE, [19200, 640, 1]))),
                       "linear_geometry_matches_boundaries": all(canonical_json(traced["linear_geometry"][role]) == canonical_json(traced[role + "_geometry"]) for role in ("input", "weight", "output")),
                       "same_cuda_device": len({record[role + "_geometry"]["device"] for record in (traced, plain) for role in ("input", "weight", "output")}) == 1,
                       "weight_pointer_matches": traced["weight_pointer_matches"] is True,
                       "linear_input_lineage_matches": traced["linear_input_matches_hook"] is True and canonical_json(traced["linear_input"]) == canonical_json(_descriptor(inputs)),
                       "linear_weight_matches": canonical_json(traced["linear_weight"]) == canonical_json(_descriptor(weights)),
                       "linear_output_lineage_matches": traced["linear_output_matches_hook"] is True and canonical_json(traced["linear_output"]) == canonical_json(_descriptor(actual)),
                       "kernel_names_match": observed_symbols == expected_symbols})
    runtime_match = canonical_json(runtime) == canonical_json(plan["runtime"])
    first = next((key for key in checks[0] if not all(item[key] for item in checks)), None) or (mismatches[0] if mismatches else None) or (None if runtime_match else "runtime")
    return _seal({"schema_version": 1, "scope": SCOPE, "plan_sha256": plan["plan_sha256"], "runtime": runtime, "runtime_matches_plan": runtime_match,
                  "observations": observations, "checks": checks, "mismatches": mismatches,
                  "mismatch_counts": [sum(item["repetition"] == repetition for item in mismatches) for repetition in range(3)], "first_divergence": first,
                  "down_matches": not mismatches and runtime_match and all(all(item.values()) for item in checks), "value_count": 19200,
                  "kernel_provenance": KERNEL_PROVENANCE, "metadata_correction_development_regression": True,
                  "down_stage_previously_acquired_by_prefix_workflow": True,
                  "prefix_boundary_reused": True, "uses_empirical_gelu_specification": True, **{key: False for key in FALSE_FLAGS}}, "report_sha256")


def acquire_down(sources: DownSources, model: Any, plan: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    code = _code_sha()
    inputs, weights, prediction = check_down_plan(sources, plan, bundle)
    _model_context(sources, model)
    ids = sources.product.entry.post.survivor_context[0].softmax.scores.rotary_bundle["entry_plan"]["input_token_ids"]
    observations = [{"traced": _capture_down(model, ids, True), "untraced": _capture_down(model, ids, False)} for _ in range(3)]
    _model_context(sources, model)
    if _code_sha() != code:
        raise ValueError("MLP-down source changed during acquisition")
    return down_report(plan, inputs, weights, prediction, observations, _runtime())


def verify_down(sources: DownSources, plan: dict[str, Any], bundle: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    try:
        inputs, weights, prediction = check_down_plan(sources, plan, bundle)
        expected = down_report(plan, inputs, weights, prediction, report["observations"], report["runtime"])
        return {"valid": canonical_json(expected) == canonical_json(report), "mode": "integrity_inputs_parameter_commitments_geometry_only_no_down_numerical_recomputation",
                "down_predictions_recomputed": False, "prefix_independently_recomputed": False, "down_matches": expected["down_matches"],
                "mismatch_counts": expected["mismatch_counts"], "global_exactness_activation_allowed": False}
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError, StopIteration) as error:
        return {"valid": False, "reason": str(error)}


def down_summary(plan: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in report.items() if key not in ("observations", "mismatches", "report_sha256")}
    body.update({"source_report_sha256": report["report_sha256"], "sources": plan["sources"], "code_sha256": plan["code_sha256"], "candidate": plan["candidate"],
                 "kernel_names": [pair["traced"]["kernel_names"] for pair in report["observations"]],
                 "observed_output_hashes": [_descriptor(_state_array(pair["traced"]["output_bits"], list(OUTPUT_SHAPE)))["sha256"] for pair in report["observations"]]})
    return _seal(body, "summary_sha256")
