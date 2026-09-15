from __future__ import annotations

import argparse
import copy
import hashlib
import json
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import gemma_independent as engine
from . import gemma_first_layer as first
from . import gemma_second_layer as second
from . import gemma_third_layer_scores as scores
from .gemma_first_layer_holdout import require_frozen, write_new
from .gemma_rotary_slice import _sha, _seal, _check_hash
from .serialization import canonical_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SCORE_PLAN = "5f8ed5aa75f3b71f883cfd136faf215154ed78bf54c6081d0f57cc33074b5509"
SCORE_REPORT = "52323090f0596f91b0ee0ef3adcab64f70afb19d8ac9d7f7479f7797e7839e24"
PREDICTION_KIND = "independent_fixed_checkpoint_prediction_v3"
GEMV_PROBE = "916bb9cac8ceff1a3db1be776003a006e9a437c2772ee771431dd434d680901d"
GEMV_CASES = ("dense_narrow", "dense_wide", "cancellation", "rounding_boundary")
GEMV_PROTOCOL = "0d2b1f026cd9b4bc608a29850fddc3e5c115c57d502f2a3e80af2ac3df946d49"
PREDICTION_SCOPE = "Source-bound token-to-target computation with fresh checkpoint weights; a resumed invocation restores a disclosed same-run prefix. Empirical primitive data and unvalidated shape transfers remain; no native comparison or unrestricted proof is implied."


def code_sha256():
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Independent runner source changed after import")
    return _sha({"runner": SOURCE_SHA256, "engine": engine.code_sha256(), "source_evidence": scores._code_sha(), "native_implementation": second._implementation_commitment()})


@dataclass(frozen=True)
class Context:
    program: dict[str, Any]
    providers: first.Providers
    profiles: dict[str, Any]
    runtime: dict[str, Any]
    sources: dict[str, Any]


def file_hashes(paths):
    hashes = {}
    for name, path in paths.items():
        with path.open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return hashes


def check_vocabulary_evidence(plan, bundle, report, replay, probe, runtime):
    expected = engine.vocabulary_evidence()
    for value, key in ((plan, "plan_sha256"), (bundle, "bundle_sha256"), (report, "report_sha256"), (replay, "replay_sha256"), (probe, "probe_sha256")):
        _check_hash(value, key)
        if key in expected and value[key] != expected[key]:
            raise ValueError("Unpinned GEMV component evidence")
    if probe["probe_sha256"] != GEMV_PROBE or bundle["bundle_sha256"] != plan["bundle_sha256"] or report["bundle_sha256"] != bundle["bundle_sha256"] or report["plan_sha256"] != plan["plan_sha256"] or replay["plan_sha256"] != plan["plan_sha256"] or replay["report_sha256"] != report["report_sha256"]:
        raise ValueError("GEMV evidence cross-binding mismatch")
    if plan["code_sha256"] != expected["arithmetic_code_sha256"] or plan["profile"] != engine.gemv.PROFILE or plan["source_file_sha256"]["arithmetic"] != engine.gemv.SOURCE_SHA256 or canonical_json(probe["runtime"]) != canonical_json(runtime):
        raise ValueError("GEMV arithmetic/source/runtime mismatch")
    if plan["prediction_sha256"] != probe["prediction_sha256"] or plan["native_baseline_report_sha256"] != probe["report_sha256"]:
        raise ValueError("GEMV baseline binding mismatch")
    if report.get("match") is not True or replay.get("fresh_integer_prediction_match") is not True or replay.get("fresh_native_report_match") is not True or plan.get("all_synthetic_predictions_before_native") is not True or plan.get("stored_validated_hidden_state_used") is not True:
        raise ValueError("Passing source-bound GEMV component prediction/comparison/replay required")
    if any(value.get("qualified") is not False for value in (plan, report, replay, probe)) or report.get("end_to_end_holdouts_executed") is not False or report.get("hardware_semantics_established") is not False:
        raise ValueError("GEMV component qualification scope changed")
    names = ("baseline", *GEMV_CASES)
    if set(plan["cases"]) != set(GEMV_CASES) or set(bundle["predictions"]) != set(names) or len(report["records"]) != 15 or len(probe["records"]) != 3 or report["native_projection_call_count"] != 45 or report["full_model_forward_count"] != 0:
        raise ValueError("Incomplete GEMV case/native coverage")
    descriptors = {}
    for name in names:
        values = first._array(bundle["predictions"][name], [1, 262144 if name == "baseline" else 512], "torch.bfloat16")
        descriptor = plan["baseline_output"] if name == "baseline" else plan["cases"][name]["output"]
        if first._descriptor(values) != descriptor:
            raise ValueError("GEMV prediction descriptor mismatch")
        if name != "baseline":
            values = np.tile(values, (1, 512))
        descriptors[name] = first._descriptor(values.reshape(1, 1, 262144))
    kernels = probe["records"][0]["cuda_kernel_names"]
    if not kernels:
        raise ValueError("GEMV native kernel observation missing")
    for index, row in enumerate(probe["records"]):
        if row["repetition"] != index or row["plain_vs_saved_native_mismatches"] != 0 or row["profiled_vs_plain_mismatches"] != 0 or row["cuda_kernel_names"] != kernels or row["plain_descriptor"] != descriptors["baseline"] or row["profiled_descriptor"] != descriptors["baseline"]:
            raise ValueError("GEMV isolated probe differs from baseline")
    for index, row in enumerate(report["records"]):
        name = names[index // 3]
        if row["case"] != name or row["repetition"] != index % 3 or row["value_count"] != 262144 or row["mismatch_count"] != 0 or row["profiled_plain_mismatch_count"] != 0 or row["same_kernel"] is not True or row["cuda_kernel_names"] != kernels or row["native_descriptor"] != descriptors[name]:
            raise ValueError("GEMV native comparison/coverage mismatch")
    return {**expected, "probe_sha256": GEMV_PROBE, "bundle_sha256": bundle["bundle_sha256"], "profile": engine.gemv.PROFILE,
            "runtime": runtime, "scope": plan["scope"], "full_model_qualified": False}


def load_vocabulary_evidence(root, runtime):
    paths = {"plan": root / "artifacts/gemma_vocab_validation_plan_v1.json", "bundle": root / "artifacts/gemma_vocab_validation_bundle_v1.json",
             "component_report": root / "artifacts/gemma_vocab_validation_report_v1.json", "replay": root / "artifacts/gemma_vocab_replay_v1.json",
             "probe": root / "artifacts/gemma_vocab_native_probe_v1.json", "protocol": root / "results/gemma_vocab_validation_protocol_v1.json",
             "script": root / "artifacts/gemma_vocab_validation_v1.py", "arithmetic": Path(engine.gemv.__file__),
             "prediction": root / "artifacts/gemma3_270m_independent_baseline_prediction_v2.json", "report": root / "artifacts/gemma3_270m_independent_baseline_report_v2.json"}
    files = file_hashes(paths)
    if files["protocol"] != GEMV_PROTOCOL:
        raise ValueError("GEMV protocol differs from predeclared component cases")
    values = [json.loads(paths[name].read_text(encoding="utf-8")) for name in ("plan", "bundle", "component_report", "replay", "probe")]
    binding = check_vocabulary_evidence(*values, runtime)
    if any(files[name] != digest for name, digest in values[0]["source_file_sha256"].items()):
        raise ValueError("GEMV component source artifacts changed")
    if files != file_hashes(paths):
        raise ValueError("GEMV artifacts changed while validating")
    return {**binding, "source_file_sha256": files}


def load_context(root: Path, model_path: Path) -> Context:
    from . import cli
    args = cli.build_parser().parse_args(["gemma-third-layer-scores-verify", "results/gemma3_270m_execution_ir.json", "--report", "artifacts/gemma3_270m_third_layer_scores_report.json"])
    for name, value in vars(args).items():
        if isinstance(value, Path) and not value.is_absolute():
            setattr(args, name, root / value)
    args.model_path = model_path
    paths = {name: path for name, path in vars(args).items() if isinstance(path, Path) and path.is_file()}
    files = file_hashes(paths)
    source = cli._load_third_score_sources(args)
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    plan, bundle, report = load(args.plan), load(args.bundle), load(args.report)
    if plan["plan_sha256"] != SCORE_PLAN or report["report_sha256"] != SCORE_REPORT:
        raise ValueError("Declared passing score evidence required")
    cached = scores.entry.holdout._CachedTwoSources(source)
    verified = scores.verify_scores(cached, plan, bundle, report, model_path)
    if verified.get("valid") is not True or verified.get("rotary_scores_match") is not True:
        raise ValueError("Independent runner requires intact passing prerequisite evidence")
    _, providers, profiles, _, _, _ = cached.validate(model_path)
    vocabulary = load_vocabulary_evidence(root, source.runtime)
    if files != file_hashes(paths):
        raise ValueError("Source files changed during prerequisite validation")
    return Context(source.program, providers, profiles, source.runtime, {"score_plan_sha256": SCORE_PLAN, "score_report_sha256": SCORE_REPORT, "score_sources": source.commitments(), "source_file_sha256": files, "vocabulary_evidence": vocabulary})


def _original(module, cls):
    if type(module) is not cls or getattr(module.forward, "__func__", None) is not cls.forward or "forward" in module.__dict__ or module.training or module._forward_hooks or module._forward_pre_hooks:
        raise ValueError("Expected original evaluation-only native module without hooks")


def model_parameters(context: Context, model):
    import torch
    from transformers.models.gemma3 import modeling_gemma3 as gemma
    from transformers.activations import PytorchGELUTanh
    if canonical_json(first._runtime()) != canonical_json(context.runtime):
        raise ValueError("Native runtime differs from frozen arithmetic context")
    _original(model, gemma.Gemma3ForCausalLM)
    _original(model.model, gemma.Gemma3TextModel)
    configuration = model.config
    expected = {"hidden_size": 640, "intermediate_size": 2048, "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 256, "rms_norm_eps": 1e-6, "num_hidden_layers": 18, "vocab_size": 262144, "sliding_window": 512, "hidden_activation": "gelu_pytorch_tanh", "_attn_implementation": "eager", "final_logit_softcapping": None, "attn_logit_softcapping": None}
    if any(getattr(configuration, name) != value for name, value in expected.items()) or configuration.layer_types != context.program["configuration"]["layer_types"] or len(model.model.layers) != 18:
        raise ValueError("Native model configuration differs from fixed IR")
    parameters = first.bind_model_tensors(context.program, model, verify_hashes=True)
    if model.lm_head.weight is not model.model.embed_tokens.weight or model.lm_head.bias is not None:
        raise ValueError("Expected original tied, bias-free vocabulary projection")
    device = model.model.embed_tokens.weight.device
    for index, layer in enumerate(model.model.layers):
        for module, cls in ((layer, gemma.Gemma3DecoderLayer), (layer.self_attn, gemma.Gemma3Attention), (layer.mlp, gemma.Gemma3MLP), (layer.mlp.act_fn, PytorchGELUTanh)):
            _original(module, cls)
        attention = layer.self_attn
        sliding = configuration.layer_types[index] == "sliding_attention"
        if layer.layer_idx != index or attention.layer_idx != index or layer.attention_type != configuration.layer_types[index] or attention.is_sliding is not sliding or attention.sliding_window != (512 if sliding else None) or attention.head_dim != 256 or attention.num_key_value_groups != 4 or attention.scaling != 0.0625 or attention.attention_dropout != 0.0 or attention.attn_logit_softcapping is not None:
            raise ValueError("Native layer geometry/attention regime differs from IR")
    seen = set()
    for node in context.program["instructions"]:
        if node["opcode"] not in ("LINEAR", "RMS_NORM"):
            continue
        name = node["parameter_refs"][0]
        if name in seen:
            continue
        seen.add(name)
        module = model
        for component in name.split(".")[:-1]:
            module = module[int(component)] if component.isdigit() else getattr(module, component)
        _original(module, torch.nn.Linear if node["opcode"] == "LINEAR" else gemma.Gemma3RMSNorm)
        tensor = parameters[name]
        if module.weight is not tensor or tensor.device != device or tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
            raise ValueError("Native parameter identity/dtype/layout mismatch")
        if node["opcode"] == "LINEAR" and (module.bias is not None or [module.out_features, module.in_features] != list(tensor.shape)):
            raise ValueError("Native LINEAR geometry mismatch")
        if node["opcode"] == "RMS_NORM" and module.eps != node["attributes"]["epsilon"]:
            raise ValueError("Native RMS epsilon mismatch")
    return parameters


def _bits(tensor):
    import torch
    if tensor.dtype == torch.float32:
        return tensor.detach().contiguous().cpu().view(torch.int32).numpy().astype(np.uint32)
    return first._bits(tensor)


def snapshot_model(context: Context, model, token_ids, target):
    ids = first._tokens(context.program, token_ids)
    parameters = model_parameters(context, model)
    nodes = engine.dependency_cone(context.program, target)
    snapshots = {}
    for name in sorted({name for node in nodes for name in node["parameter_refs"]}):
        commitment = context.program["parameter_commitments"][name]
        if name == first.EMBEDDING:
            bits = np.stack([_bits(parameters[name][int(index)]) for index in ids[0]])
            snapshots[name] = {"format": "selected_token_rows_v1", "token_indices": ids[0].tolist(), "bits": bits, "descriptor": first._descriptor(bits),
                               "full_table_commitment": copy.deepcopy(commitment), "membership_attestation": "selected_by_index_during_fresh_hash_verified_model_binding"}
        else:
            bits = _bits(parameters[name])
            snapshots[name] = {"format": "full_parameter_bits_v1", "bits": bits, "descriptor": first._descriptor(bits), "commitment": copy.deepcopy(commitment)}
        bits.setflags(write=False)
    engine.validate_parameters(context.program, ids, snapshots, nodes)
    return snapshots


def calibrate_global_rotary(context: Context, model):
    import torch
    from transformers.models.gemma3 import modeling_gemma3 as gemma
    code = code_sha256()
    parameters = model_parameters(context, model)
    module = model.model.rotary_emb
    _original(module, gemma.Gemma3RotaryEmbedding)
    frequency = _bits(parameters[engine.GLOBAL_FREQUENCY])
    reference = None
    with torch.no_grad():
        zeros = torch.zeros((1, 30, 640), dtype=torch.bfloat16, device=model.model.embed_tokens.weight.device)
        positions = torch.arange(30, device=zeros.device, dtype=torch.int64)[None, :]
        for _ in range(3):
            cosine, sine = module(zeros, positions)
            observed = {"cosine": _bits(cosine), "sine": _bits(sine)}
            if reference is not None and any(not np.array_equal(observed[name], reference[name]) for name in observed):
                raise ValueError("Global rotary primitive calibration is not repeat-stable")
            reference = observed
    packet = _seal({"kind": "fixed_position_empirical_global_rotary_v1", "program_sha256": context.program["program_sha256"], "positions": list(range(30)),
                    "runtime": context.runtime, "frequency": first._descriptor(frequency), "cosine_bits": reference["cosine"].tolist(), "sine_bits": reference["sine"].tolist(),
                    "descriptors": {name: first._descriptor(value) for name, value in reference.items()}, "empirical_primitive_data": True, "hidden_states_used": False,
                    "native_repeat_count": 3, "native_repeat_exact": True, "calibration_code_sha256": code, "sources": context.sources,
                    "scope": "Original global rotary module on zero-valued shape carrier and fixed positions; empirical primitive specification, not model activation or hardware proof."}, "provider_sha256")
    model_parameters(context, model)
    if code != code_sha256() or not np.array_equal(frequency, _bits(parameters[engine.GLOBAL_FREQUENCY])):
        raise ValueError("Global calibration source or frequency changed")
    engine.check_global_rotary(packet, context.program, frequency, context.runtime)
    return packet


def _snapshot_metadata(snapshots):
    return {name: {key: value for key, value in snapshot.items() if key != "bits"} for name, snapshot in snapshots.items()}


def check_global_binding(context, global_rotary):
    if global_rotary is not None:
        _check_hash(global_rotary, "provider_sha256")
        if global_rotary.get("calibration_code_sha256") != code_sha256() or global_rotary.get("sources") != context.sources:
            raise ValueError("Global rotary calibration source/code differs from the runner context")


def predict(context, model, token_ids, global_rotary, *, target="selected_token_id", workers=4, vocabulary_candidate=False, retain_states=False, progress=None, checkpoint_dir=None, resume=False, checkpoint_every=8):
    code, source_guard = code_sha256(), _sha(context.sources)
    check_global_binding(context, global_rotary)
    snapshots = snapshot_model(context, model, token_ids, target)
    metadata = _snapshot_metadata(snapshots)
    result = engine.execute(context.program, token_ids, snapshots, context.providers, context.profiles, context.runtime,
                            target=target, global_rotary=global_rotary, workers=workers, vocabulary_candidate=vocabulary_candidate, retain_states=retain_states, progress=progress,
                            checkpoint_dir=checkpoint_dir, resume=resume, checkpoint_every=checkpoint_every, checkpoint_context={"runner_code_sha256": code, "sources_sha256": source_guard})
    engine.check_execution(context.program, result)
    after = snapshot_model(context, model, token_ids, target)
    if canonical_json(metadata) != canonical_json(_snapshot_metadata(after)) or code != code_sha256() or source_guard != _sha(context.sources):
        raise ValueError("Model/source changed during independent prediction")
    return _seal({"schema_version": 3, "kind": PREDICTION_KIND, "runner_code_sha256": code, "sources": context.sources,
                  "parameter_snapshots": metadata, "global_provider_sha256": global_rotary.get("provider_sha256") if global_rotary else None,
                  "execution": result, "scope": PREDICTION_SCOPE}, "prediction_sha256")


def check_prediction(context, prediction, global_rotary):
    _check_hash(prediction, "prediction_sha256")
    check_global_binding(context, global_rotary)
    if prediction["kind"] != PREDICTION_KIND or prediction["schema_version"] != 3 or prediction["scope"] != PREDICTION_SCOPE or prediction["runner_code_sha256"] != code_sha256() or prediction["sources"] != context.sources or prediction["global_provider_sha256"] != (global_rotary.get("provider_sha256") if global_rotary else None):
        raise ValueError("Prediction source/global-provider/code binding mismatch")
    execution = prediction["execution"]
    if execution["global_rotary_sha256"] != prediction["global_provider_sha256"]:
        raise ValueError("Execution global provider differs from prediction header")
    engine.check_execution(context.program, execution)
    if canonical_json(execution["runtime"]) != canonical_json(context.runtime) or canonical_json(execution["profiles"]) != canonical_json(context.profiles) or canonical_json(execution["providers"]) != canonical_json(context.providers.commitments()):
        raise ValueError("Prediction primitive providers differ from validated context")
    metadata = prediction["parameter_snapshots"]
    required = {name for node in engine.dependency_cone(context.program, execution["target"]) for name in node["parameter_refs"]}
    if set(metadata) != required:
        raise ValueError("Prediction snapshot coverage mismatch")
    for record in execution["records"]:
        for name, value in record["payload"]["parameters"].items():
            if value != metadata[name]:
                raise ValueError("Trace parameter differs from frozen snapshot metadata")


def same_numerical_prediction(left, right):
    def normalized(value):
        _check_hash(value, "prediction_sha256")
        result = copy.deepcopy(value)
        result.pop("prediction_sha256")
        result["execution"]["checkpoint_resume"] = None
        result["execution"]["stored_boundary_predictions_used"] = False
        return result
    return canonical_json(normalized(left)) == canonical_json(normalized(right))


def capture_boundaries(context, model, token_ids, target):
    import torch
    target_allowed = target == "selected_token_id" or target in {f"hidden.{index}" for index in range(1, 19)}
    if not target_allowed:
        raise ValueError("Native comparator supports decoder-boundary or selected-token targets only")
    tensors, handles, counts = {}, [], {}
    ids = torch.tensor(token_ids, dtype=torch.int64, device=model.model.embed_tokens.weight.device)
    before = code_sha256()

    class Complete(Exception):
        pass

    def store(name, value):
        if name in tensors:
            raise ValueError("Repeated native boundary")
        shape = first._shape(context.program, name)
        bits = first._array(_bits(value), shape, "torch.bfloat16")
        tensors[name] = bits.tolist()
        counts[name] = first._descriptor(bits)

    def layer_hook(index):
        def hook(module, args, output):
            if type(output) is not tuple or len(output) != 1:
                raise ValueError("Unexpected original decoder output tuple")
            name = "hidden." + str(index + 1)
            store(name, output[0])
            if name == target:
                raise Complete()
        return hook

    def final_hook(module, args, output):
        store("hidden.final", output)

    def logits_hook(module, args, output):
        store("hidden.last", args[0])
        store("logits.last", output)

    stopped, selected = False, None
    try:
        with torch.no_grad():
            for index, layer in enumerate(model.model.layers):
                handles.append(layer.register_forward_hook(layer_hook(index)))
            handles.append(model.model.norm.register_forward_hook(final_hook))
            handles.append(model.lm_head.register_forward_hook(logits_hook))
            output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, output_attentions=False, logits_to_keep=1)
            if target != "selected_token_id":
                raise ValueError("Native boundary stop was not reached")
            if first._bits(output.logits).tolist() != tensors["logits.last"]:
                raise ValueError("Returned logits differ from original vocabulary projection")
            selected = int(torch.argmax(output.logits[0, -1]).item())
    except Complete:
        stopped = True
    finally:
        for handle in handles:
            handle.remove()
    expected = {f"hidden.{index}" for index in range(1, 19 if target == "selected_token_id" else int(target.split(".")[1]) + 1)}
    if target == "selected_token_id":
        expected.update(("hidden.final", "hidden.last", "logits.last"))
    if set(tensors) != expected or (target != "selected_token_id" and not stopped):
        raise ValueError("Native comparison boundary coverage mismatch")
    if code_sha256() != before or ids.cpu().tolist() != token_ids:
        raise ValueError("Native input/source changed during capture")
    return {"state_bits": tensors, "descriptors": counts, "selected_token_id": selected, "runtime": first._runtime(), "runner_code_sha256": before,
            "native_scope": "original_forward_with_read_only_boundary_hooks_not_all_internal_states", "stopped_at_target": stopped}


def comparison_report(context, prediction, observations, guards):
    execution = prediction["execution"]
    if execution["status"] != "complete_uncompared":
        if observations:
            raise ValueError("Abstained prediction must not have native observations")
        return _seal({"prediction_sha256": prediction["prediction_sha256"], "status": "prediction_abstained", "match": False, "abstention": execution["abstention"],
                      "observations": [], "original_forward_count": 0, "checkpoint_resume": execution["checkpoint_resume"], "stored_boundary_predictions_used": execution["stored_boundary_predictions_used"], "qualified": False, "hardware_semantics_established": False}, "report_sha256")
    if len(observations) != 3:
        raise ValueError("Exactly three original native comparison forwards required")
    target = execution["target"]
    names = [f"hidden.{index}" for index in range(1, 19 if target == "selected_token_id" else int(target.split(".")[1]) + 1)]
    if target == "selected_token_id":
        names.extend(("hidden.final", "hidden.last", "logits.last"))
    comparisons, checks = [], []
    for observed in observations:
        if set(observed["state_bits"]) != set(names) or set(observed["descriptors"]) != set(names):
            raise ValueError("Native report boundary coverage mismatch")
        comparisons.append({name: first._comparison(execution["state_bits"][name], observed["state_bits"][name], first._shape(context.program, name), "torch.bfloat16") for name in names})
        checks.append({"runtime": observed["runtime"] == context.runtime, "code": observed["runner_code_sha256"] == prediction["runner_code_sha256"],
                       "token": observed["selected_token_id"] == execution["selected_token_id"],
                       "descriptors": all(first._descriptor(first._array(observed["state_bits"][name], first._shape(context.program, name), "torch.bfloat16")) == observed["descriptors"][name] for name in names),
                       "scope": observed["native_scope"] == "original_forward_with_read_only_boundary_hooks_not_all_internal_states" and observed["stopped_at_target"] is (target != "selected_token_id")})
    counts = {name: sum(values[name]["mismatch_count"] for values in comparisons) for name in names}
    divergence = next(({"repetition": index, "state": name, **values[name]["first_divergence"]} for index, values in enumerate(comparisons) for name in names if values[name]["first_divergence"] is not None), None)
    good_guards = set(guards) == {"source_unchanged", "checkpoint_unchanged", "frozen_files_unchanged"} and all(value is True for value in guards.values())
    matched = not any(counts.values()) and all(all(values.values()) for values in checks) and good_guards
    if divergence is None and not matched:
        divergence = {"check": "native_control_or_acquisition_guard"}
    return _seal({"prediction_sha256": prediction["prediction_sha256"], "status": "matched" if matched else "mismatched", "match": matched,
                  "scope": "Fixed-case complete target prediction versus three original forwards at decoder/final/logit boundaries; not all internal native states or hardware proof.",
                  "target": target, "selected_token_id": execution["selected_token_id"] if matched else None, "observations": observations, "comparisons": comparisons,
                  "checks": checks, "acquisition_guards": guards, "mismatch_counts": counts, "aggregate_mismatch_count": sum(counts.values()), "first_divergence": divergence,
                  "original_forward_count": 3, "checkpoint_resume": execution["checkpoint_resume"], "stored_boundary_predictions_used": execution["stored_boundary_predictions_used"], "qualified": False, "hardware_semantics_established": False, "full_model_qualified": False}, "report_sha256")


def acquire(context, model, prediction, global_rotary, prediction_path):
    require_frozen(prediction_path, prediction)
    check_prediction(context, prediction, global_rotary)
    execution = prediction["execution"]
    if execution["status"] != "complete_uncompared":
        return comparison_report(context, prediction, [], {})
    code = code_sha256()
    snapshot = snapshot_model(context, model, execution["input_token_ids"], execution["target"])
    if canonical_json(_snapshot_metadata(snapshot)) != canonical_json(prediction["parameter_snapshots"]):
        raise ValueError("Fresh native checkpoint snapshots differ from prediction")
    observations = [capture_boundaries(context, model, execution["input_token_ids"], execution["target"]) for _ in range(3)]
    after = snapshot_model(context, model, execution["input_token_ids"], execution["target"])
    require_frozen(prediction_path, prediction)
    return comparison_report(context, prediction, observations, {"source_unchanged": code == code_sha256(), "checkpoint_unchanged": canonical_json(_snapshot_metadata(snapshot)) == canonical_json(_snapshot_metadata(after)), "frozen_files_unchanged": True})


def build_parser():
    parser = argparse.ArgumentParser(description="Fixed-checkpoint independent IR execution and boundary comparison; unvalidated profile transfers never grant qualification.")
    parser.add_argument("operation", choices=("calibrate", "predict", "compare", "run", "verify"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tokens", type=Path)
    parser.add_argument("--global-provider", type=Path)
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target", default="selected_token_id")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--allow-vocabulary-candidate", action="store_true")
    parser.add_argument("--retain-states", action="store_true")
    parser.add_argument("--reexecute", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=8)
    return parser


def main():
    from .cli import _holdout_model
    args = build_parser().parse_args()
    root = args.root.resolve()
    model_path = args.model_path or root / ".models/gemma-3-270m-it"
    if args.resume and args.checkpoint_dir is None or not 1 <= args.checkpoint_every <= 533:
        raise ValueError("Resume requires a checkpoint directory; checkpoint interval must be 1..533")
    if args.checkpoint_dir is not None:
        directory = args.checkpoint_dir.resolve()
        if args.operation not in ("predict", "run", "verify") or args.operation == "verify" and not args.reexecute:
            raise ValueError("Checkpoints apply only to prediction or numerical reexecution")
        if directory == model_path.resolve() or model_path.resolve() in directory.parents or (directory == root or root in directory.parents) and root / "artifacts" not in directory.parents:
            raise ValueError("Checkpoint state must remain outside the model and under ignored artifacts inside the repository")
        if not args.resume and directory.exists():
            raise ValueError("Fresh checkpoint directory already exists; use explicit --resume or a new path")
        if args.resume and not directory.is_dir():
            raise ValueError("Resume checkpoint directory is missing")
    outputs = [args.output] if args.operation == "calibrate" else [args.prediction] if args.operation == "predict" else [args.report] if args.operation == "compare" else [args.prediction, args.report] if args.operation == "run" else []
    if any(path is None for path in outputs) or len({path.resolve() for path in outputs}) != len(outputs) or args.checkpoint_dir is not None and any(path.resolve() == args.checkpoint_dir.resolve() for path in outputs):
        raise ValueError("Operation requires distinct explicit output paths")
    inputs = [path.resolve() for path in (args.tokens, args.global_provider, args.model_path) if path is not None]
    if any(path.exists() or path.resolve() in inputs or model_path.resolve() in path.resolve().parents for path in outputs):
        raise ValueError("Outputs must be new and must not overwrite inputs or checkpoint")
    if any(root in path.resolve().parents and root / "artifacts" not in path.resolve().parents for path in outputs if args.operation != "calibrate"):
        raise ValueError("Tensor-rich prediction/report outputs must remain under ignored artifacts/")
    frozen_paths = {name: path for name, path in (("tokens", args.tokens), ("global_provider", args.global_provider), ("prediction", args.prediction), ("report", args.report)) if path is not None and path not in outputs}
    frozen_hashes = file_hashes(frozen_paths)
    context = None

    def check_files():
        if file_hashes(frozen_paths) != frozen_hashes:
            raise ValueError("Frozen runner input files changed")
        if context is not None and load_vocabulary_evidence(root, context.runtime) != context.sources["vocabulary_evidence"]:
            raise ValueError("Frozen GEMV evidence changed")

    context = load_context(root, model_path)
    check_files()
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    if args.operation == "calibrate":
        packet = calibrate_global_rotary(context, _holdout_model(model_path))
        check_files()
        write_new(args.output, packet)
        print(json.dumps({"provider_sha256": packet["provider_sha256"], "scope": packet["scope"]}, indent=2))
        return 0
    global_rotary = load(args.global_provider) if args.global_provider else None
    if args.global_provider:
        require_frozen(args.global_provider, global_rotary)
    if args.operation in ("predict", "run"):
        if args.tokens is None:
            raise ValueError("A frozen input-token JSON file is required")
        token_ids = load(args.tokens)
        first._tokens(context.program, token_ids)
        require_frozen(args.tokens, token_ids)
        model = _holdout_model(model_path)
        prediction = predict(context, model, token_ids, global_rotary, target=args.target, workers=args.workers, vocabulary_candidate=args.allow_vocabulary_candidate,
                             retain_states=args.retain_states, progress=lambda event: print(json.dumps(event), flush=True),
                             checkpoint_dir=args.checkpoint_dir, resume=args.resume, checkpoint_every=args.checkpoint_every)
        require_frozen(args.tokens, token_ids)
        check_files()
        write_new(args.prediction, prediction)
        print(json.dumps({"prediction_sha256": prediction["prediction_sha256"], "status": prediction["execution"]["status"], "abstention": prediction["execution"]["abstention"]}), flush=True)
        if args.operation == "predict":
            return 0 if prediction["execution"]["status"] == "complete_uncompared" else 1
    else:
        if args.prediction is None:
            raise ValueError("Frozen prediction path required")
        prediction = load(args.prediction)
        require_frozen(args.prediction, prediction)
        check_prediction(context, prediction, global_rotary)
    if args.operation in ("compare", "run"):
        if args.operation == "compare":
            model = _holdout_model(model_path) if prediction["execution"]["status"] == "complete_uncompared" else None
        check_files()
        report = acquire(context, model, prediction, global_rotary, args.prediction)
        check_files()
        write_new(args.report, report)
        print(json.dumps({"report_sha256": report["report_sha256"], "status": report["status"], "match": report["match"], "first_divergence": report.get("first_divergence"), "selected_token_id": report.get("selected_token_id")}, indent=2))
        return 0 if report["match"] else 1
    if args.report is None:
        raise ValueError("Frozen report path required")
    report = load(args.report)
    _check_hash(report, "report_sha256")
    expected = comparison_report(context, prediction, report["observations"], report.get("acquisition_guards", {}))
    valid = canonical_json(expected) == canonical_json(report)
    if args.reexecute and valid:
        model = _holdout_model(model_path)
        prior = prediction["execution"]
        fresh = predict(context, model, prior["input_token_ids"], global_rotary, target=prior["target"], workers=args.workers, vocabulary_candidate=prior["vocabulary_candidate_enabled"], retain_states=prior["state_retention"] == "all",
                        progress=lambda event: print(json.dumps(event), flush=True), checkpoint_dir=args.checkpoint_dir, resume=args.resume, checkpoint_every=args.checkpoint_every)
        check_prediction(context, fresh, global_rotary)
        valid = same_numerical_prediction(fresh, prediction)
        if valid:
            valid = canonical_json(acquire(context, model, prediction, global_rotary, args.prediction)) == canonical_json(report)
    check_files()
    print(json.dumps({"valid": valid, "match": valid and report["match"], "reexecute_requested": args.reexecute, "qualified": False}, indent=2))
    return 0 if valid and report["match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
