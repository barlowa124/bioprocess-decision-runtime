from __future__ import annotations

import hashlib
import inspect
import math
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

from . import reference_gemma
from .gemma_rsqrt_lookup import _runtime


STATE_SPECS = {
    "hidden.1": ((1, 30, 640), "torch.bfloat16"),
    "rotary.local.cosine": ((1, 30, 256), "torch.bfloat16"),
    "rotary.local.sine": ((1, 30, 256), "torch.bfloat16"),
    "mask.sliding": ((1, 1, 30, 30), "torch.bfloat16"),
    "layer.1.attention.normalized": ((1, 30, 640), "torch.bfloat16"),
    "layer.1.query.flat": ((1, 30, 1024), "torch.bfloat16"),
    "layer.1.query.heads": ((1, 4, 30, 256), "torch.bfloat16"),
    "layer.1.key.flat": ((1, 30, 256), "torch.bfloat16"),
    "layer.1.key.heads": ((1, 1, 30, 256), "torch.bfloat16"),
    "layer.1.value.flat": ((1, 30, 256), "torch.bfloat16"),
    "layer.1.value.heads": ((1, 1, 30, 256), "torch.bfloat16"),
    "layer.1.query.normalized": ((1, 4, 30, 256), "torch.bfloat16"),
    "layer.1.key.normalized": ((1, 1, 30, 256), "torch.bfloat16"),
    "layer.1.query.rotary": ((1, 4, 30, 256), "torch.bfloat16"),
    "layer.1.key.rotary": ((1, 1, 30, 256), "torch.bfloat16"),
    "layer.1.key.repeated": ((1, 4, 30, 256), "torch.bfloat16"),
    "layer.1.value.repeated": ((1, 4, 30, 256), "torch.bfloat16"),
    "layer.1.attention.unscaled_scores": ((1, 4, 30, 30), "torch.bfloat16"),
    "layer.1.attention.scaled_scores": ((1, 4, 30, 30), "torch.bfloat16"),
    "layer.1.attention.masked_scores": ((1, 4, 30, 30), "torch.bfloat16"),
    "layer.1.attention.probability": ((1, 4, 30, 30), "torch.bfloat16"),
    "layer.1.attention.head_output": ((1, 4, 30, 256), "torch.bfloat16"),
    "layer.1.attention.concatenated": ((1, 30, 1024), "torch.bfloat16"),
    "layer.1.attention.projected": ((1, 30, 640), "torch.bfloat16"),
    "layer.1.attention.post_normalized": ((1, 30, 640), "torch.bfloat16"),
    "layer.1.post_attention_residual": ((1, 30, 640), "torch.bfloat16"),
    "layer.1.mlp.normalized": ((1, 30, 640), "torch.bfloat16"),
    "layer.1.mlp.gate": ((1, 30, 2048), "torch.bfloat16"),
    "layer.1.mlp.up": ((1, 30, 2048), "torch.bfloat16"),
    "layer.1.mlp.activated_gate": ((1, 30, 2048), "torch.bfloat16"),
    "layer.1.mlp.product": ((1, 30, 2048), "torch.bfloat16"),
    "layer.1.mlp.down": ((1, 30, 640), "torch.bfloat16"),
    "layer.1.mlp.post_normalized": ((1, 30, 640), "torch.bfloat16"),
    "hidden.2": ((1, 30, 640), "torch.bfloat16"),
}


def _code_sha() -> str:
    pieces = (Path(__file__).read_bytes(), Path(reference_gemma.__file__).read_bytes(),
              inspect.getsource(reference_gemma._observe_rms_module).encode(),
              inspect.getsource(reference_gemma._rms_tensor_f32_bits).encode(),
              reference_gemma._rms_code_commitment().encode())
    return hashlib.sha256(b"\0".join(pieces)).hexdigest()


def _bit_tensor(value: Any) -> Any:
    import torch

    with torch._C._DisableTorchDispatch():
        dtypes = {torch.bfloat16: torch.uint16, torch.float32: torch.uint32, torch.int64: torch.int64}
        if not isinstance(value, torch.Tensor) or value.dtype not in dtypes:
            raise ValueError("Expected BF16, FP32 or int64 native tensor")
        return value.detach().cpu().contiguous().view(dtypes[value.dtype])


def _tensor_record(value: Any, shape: tuple[int, ...], dtype: str) -> tuple[list[Any], dict[str, Any]]:
    import torch

    with torch._C._DisableTorchDispatch():
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or str(value.dtype) != dtype:
            raise ValueError("Unexpected native state shape/dtype")
        return _bit_tensor(value).tolist(), {
            "shape": list(value.shape), "dtype": str(value.dtype), "strides": list(value.stride()),
            "device": str(value.device), "alignment_mod16": value.data_ptr() % 16,
        }


def _profile_call(function: Any, device: Any) -> tuple[Any, list[str]]:
    import torch
    from torch.profiler import profile, ProfilerActivity

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities) as profiled:
        output = function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    names = sorted({event.name for event in profiled.events() if event.device_type == torch.autograd.DeviceType.CUDA})
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Missing distinct native CUDA kernel symbols")
    return output, names


def capture_second_layer(model: Any, token_ids: list[list[int]], traced: bool) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional
    from torch.utils._python_dispatch import TorchDispatchMode
    from transformers.models.gemma3 import modeling_gemma3 as gemma

    if type(traced) is not bool or type(token_ids) is not list or len(token_ids) != 1 or type(token_ids[0]) is not list or len(token_ids[0]) != 30:
        raise ValueError("Expected one 30-token input and a boolean tracing flag")
    if type(model) is not gemma.Gemma3ForCausalLM or len(model.model.layers) < 3 or not hasattr(model.model, "rotary_emb_local"):
        raise ValueError("Expected original Gemma3 decoder with native prefix and later-layer guard")
    if any(type(token) is not int or not 0 <= token < model.config.vocab_size for token in token_ids[0]):
        raise ValueError("Invalid token IDs")
    prefix, layer, embed = model.model.layers[0], model.model.layers[1], model.model.embed_tokens
    attention, mlp = layer.self_attn, layer.mlp
    config = model.config
    if any(module.training for module in model.modules()) or config._attn_implementation != "eager":
        raise ValueError("Native capture requires eager evaluation")
    if (config.hidden_size, config.intermediate_size, config.head_dim, config.num_attention_heads, config.num_key_value_heads) != (640, 2048, 256, 4, 1):
        raise ValueError("Unsupported second-layer head geometry")
    for index, selected in enumerate((prefix, layer)):
        attn = selected.self_attn
        if (type(selected) is not gemma.Gemma3DecoderLayer or selected.layer_idx != index or attn.layer_idx != index
                or selected.attention_type != "sliding_attention" or config.layer_types[index] != "sliding_attention"
                or not attn.is_sliding or attn.sliding_window != 512 or attn.num_key_value_groups != 4
                or attn.scaling != 0.0625 or attn.attn_logit_softcapping is not None):
            raise ValueError("Unsupported native prefix/second-layer attention configuration or layer index")
    if set(prefix.modules()) & set(layer.modules()) or len({id(item) for item in model.model.layers}) != len(model.model.layers):
        raise ValueError("Native decoder layers must not alias")
    if embed.weight.dtype != torch.bfloat16 or any(parameter.dtype != torch.bfloat16 or parameter.device != embed.weight.device for selected in (prefix, layer) for parameter in selected.parameters()):
        raise ValueError("Native capture requires same-device BF16 parameters")
    device = embed.weight.device
    code_before, runtime_before = _code_sha(), _runtime()
    tokens_before = [list(row) for row in token_ids]
    ids = torch.tensor(token_ids, dtype=torch.int64, device=device)
    record: dict[str, Any] = {"state_bits": {}, "geometry": {}, "checks": {}, "scalar_stages": {}, "kernels": {}, "softmax_f32_bits": []}
    tensors, pending, counts, handles = {}, {}, {}, []
    active = {"layer": False, "attention": False, "eager": False}
    decoder_signature = inspect.signature(gemma.Gemma3DecoderLayer.forward)

    class Complete(Exception):
        pass

    def once(role: str) -> None:
        counts[role] = counts.get(role, 0) + 1
        if counts[role] != 1:
            raise ValueError("Repeated original operation: " + role)

    def capture(name: str, value: Any) -> None:
        once("state:" + name)
        bits, geometry = _tensor_record(value, *STATE_SPECS[name])
        if value.device != device:
            raise ValueError("Native state device changed")
        record["state_bits"][name], record["geometry"][name], tensors[name] = bits, geometry, value

    def link(value: Any, name: str, identity: bool = True) -> None:
        if name not in tensors or (identity and value is not tensors[name]):
            raise ValueError("Missing native operand identity: " + name)
        bits, geometry = _tensor_record(value, *STATE_SPECS[name])
        if bits != record["state_bits"][name] or geometry != record["geometry"][name]:
            raise ValueError("Native operand bits/geometry changed: " + name)
        record["checks"]["operand_link:" + name] = True

    def mapped(value: Any, name: str, transform: Any, shared: bool) -> None:
        link(tensors[name], name)
        with torch._C._DisableTorchDispatch():
            if value.dtype != tensors[name].dtype or value.device != tensors[name].device or (shared and value.data_ptr() != tensors[name].data_ptr()):
                raise ValueError("Native view storage/dtype mismatch: " + name)
            if _bit_tensor(value).tolist() != transform(_bit_tensor(tensors[name])).tolist():
                raise ValueError("Native integer coordinate mapping mismatch: " + name)
        record["checks"]["coordinate_link:" + name] = True

    def profiled(role: str, function: Any) -> Any:
        once("native:" + role)
        output, symbols = _profile_call(function, device)
        record["kernels"][role] = symbols
        return output

    def prefix_input(module: Any, args: Any) -> None:
        once("prefix_input")
        if active["layer"] or "prefix_output" in counts or "layer" in counts:
            raise ValueError("Native layer zero must precede target layer one")

    def prefix_output(module: Any, args: Any, output: Any) -> None:
        once("prefix_output")
        if counts.get("prefix_input") != 1 or "layer" in counts:
            raise ValueError("Native layer zero output order changed")
        if type(output) is not tuple or len(output) != 1:
            raise ValueError("Original prefix decoder must return a one-element hidden-state tuple")
        pending["prefix_hidden"] = output[0]
        pending["prefix_record"] = _tensor_record(output[0], *STATE_SPECS["hidden.1"])

    def layer_input(module: Any, args: Any, kwargs: Any) -> None:
        once("layer")
        if counts.get("prefix_input") != 1 or counts.get("prefix_output") != 1 or active["layer"]:
            raise ValueError("Native layer zero must finish before target layer one")
        bound = decoder_signature.bind(module, *args, **kwargs).arguments
        hidden = bound["hidden_states"]
        capture("hidden.1", hidden)
        if hidden is not pending["prefix_hidden"] or pending["prefix_record"] != (record["state_bits"]["hidden.1"], record["geometry"]["hidden.1"]):
            raise ValueError("Native layer zero output changed before layer one input")
        if bound.get("use_cache") or bound.get("past_key_value") is not None or bound.get("output_attentions"):
            raise ValueError("Native second-layer capture forbids cache/attention outputs")
        positions = bound.get("position_ids")
        position_bits, _ = _tensor_record(positions, (1, 30), "torch.int64")
        if position_bits != [list(range(30))] or positions.device != device:
            raise ValueError("Expected auxiliary position IDs zero through twenty-nine")
        cache_position = bound.get("cache_position")
        cache_bits, _ = _tensor_record(cache_position, (30,), "torch.int64")
        if cache_bits != list(range(30)) or cache_position.device != device:
            raise ValueError("Expected auxiliary cache positions zero through twenty-nine")
        local = bound.get("position_embeddings_local")
        if type(local) is not tuple or len(local) != 2:
            raise ValueError("Expected original local rotary table pair")
        roots = (("rotary.local.cosine", local[0]), ("rotary.local.sine", local[1]), ("mask.sliding", bound.get("attention_mask")))
        pending["root_records"] = {}
        for name, value in roots:
            bits, geometry = _tensor_record(value, *STATE_SPECS[name])
            if value.device != device:
                raise ValueError("Native root device changed")
            if name == "mask.sliding" and bits != [[[[0 if column <= row else 65407 for column in range(30)] for row in range(30)]]]:
                raise ValueError("Expected native causal sliding mask for thirty positions")
            pending["root_records"][name] = (value, bits, geometry)
            if traced:
                capture(name, value)
        pending["positions"] = (positions, cache_position)
        active["layer"] = True

    def layer_output(module: Any, args: Any, output: Any) -> None:
        once("layer_output")
        if counts.get("layer") != 1:
            raise ValueError("Missing target layer one entry")
        if type(output) is not tuple or len(output) != 1:
            raise ValueError("Original decoder must return a one-element hidden-state tuple")
        if traced and output[0] is not pending.get("hidden.2"):
            raise ValueError("Final original residual ADD did not reach decoder tuple")
        link(tensors["hidden.1"], "hidden.1")
        capture("hidden.2", output[0])
        active["layer"] = False
        raise Complete()

    def forbidden(module: Any, args: Any) -> None:
        raise ValueError("Later layer, final normalization or LM head must not execute")

    def attention_input(module: Any, args: Any, kwargs: Any) -> None:
        if not active["layer"]:
            return
        once("attention")
        if active["attention"]:
            raise ValueError("Unexpected original attention scope")
        bound = inspect.signature(gemma.Gemma3Attention.forward).bind(module, *args, **kwargs).arguments
        link(bound["hidden_states"], "layer.1.attention.normalized")
        local = bound["position_embeddings"]
        if type(local) is not tuple or len(local) != 2:
            raise ValueError("Expected original local rotary table pair")
        link(local[0], "rotary.local.cosine")
        link(local[1], "rotary.local.sine")
        link(bound["attention_mask"], "mask.sliding")
        active["attention"] = True

    def attention_output(module: Any, args: Any, output: Any) -> None:
        active["attention"] = False

    def rotary(q: Any, k: Any, cos: Any, sin: Any, *args: Any, **kwargs: Any) -> Any:
        if not active["attention"]:
            return original_rotary(q, k, cos, sin, *args, **kwargs)
        once("apply_rotary")
        for value, name in ((q, "layer.1.query.normalized"), (k, "layer.1.key.normalized"), (cos, "rotary.local.cosine"), (sin, "rotary.local.sine")):
            link(value, name)
        if args or kwargs.get("unsqueeze_dim", 1) != 1:
            raise ValueError("Unsupported original rotary broadcast")
        output = original_rotary(q, k, cos, sin, *args, **kwargs)
        capture("layer.1.query.rotary", output[0])
        capture("layer.1.key.rotary", output[1])
        return output

    def repeat(value: Any, n_rep: int) -> Any:
        if not active["eager"]:
            return original_repeat(value, n_rep)
        role = "key" if "layer.1.key.repeated" not in tensors else "value"
        once("repeat:" + role)
        if n_rep != 4:
            raise ValueError("Expected four native KV repetitions")
        if role == "value":
            mapped(value, "layer.1.value.flat", lambda bits: bits.reshape(1, 30, 1, 256).transpose(1, 2), True)
            capture("layer.1.value.heads", value)
        link(value, "layer.1." + role + (".rotary" if role == "key" else ".heads"))
        output = original_repeat(value, n_rep)
        capture("layer.1." + role + ".repeated", output)
        return output

    def matmul(left: Any, right: Any, *args: Any, **kwargs: Any) -> Any:
        if not active["eager"]:
            return original_matmul(left, right, *args, **kwargs)
        if args or kwargs:
            raise ValueError("Unsupported native attention matmul signature")
        if "layer.1.attention.unscaled_scores" not in tensors:
            link(left, "layer.1.query.rotary")
            mapped(right, "layer.1.key.repeated", lambda bits: bits.transpose(2, 3), True)
            name = "layer.1.attention.unscaled_scores"
        else:
            if left is not pending.get("dropout"):
                raise ValueError("AV left operand did not descend from original softmax/dropout")
            capture("layer.1.attention.probability", left)
            link(right, "layer.1.value.repeated")
            name = "layer.1.attention.head_output"
        output = profiled(name, lambda: original_matmul(left, right))
        capture(name, output)
        return output

    def softmax(input: Any, *args: Any, **kwargs: Any) -> Any:
        if not active["eager"]:
            return original_softmax(input, *args, **kwargs)
        link(input, "layer.1.attention.masked_scores")
        if args or kwargs != {"dim": -1, "dtype": torch.float32}:
            raise ValueError("Expected original FP32 softmax signature")
        output = original_softmax(input, **kwargs)
        bits, _ = _tensor_record(output, (1, 4, 30, 30), "torch.float32")
        if output is not pending.get("softmax_native") or bits != pending.get("softmax_native_bits"):
            raise ValueError("Softmax wrapper output lost original native FP32 output lineage")
        record["softmax_f32_bits"] = bits
        record["checks"]["softmax_native_output_link"] = True
        pending["softmax_f32"] = output
        return output

    def dropout(input: Any, *args: Any, **kwargs: Any) -> Any:
        if not active["eager"]:
            return original_dropout(input, *args, **kwargs)
        once("dropout")
        if args or kwargs != {"p": 0.0, "training": False} or input is not pending.get("softmax_bf16"):
            raise ValueError("Expected original evaluation zero-dropout softmax cast")
        output = original_dropout(input, **kwargs)
        if output is not input:
            raise ValueError("Zero-dropout changed original probability tensor")
        pending["dropout"] = output
        return output

    class AttentionCapture(TorchDispatchMode):
        def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
            kwargs = kwargs or {}
            if func is torch.ops.aten.mul.Tensor and "layer.1.attention.unscaled_scores" in tensors and "softmax_f32" not in pending:
                if len(args) != 2 or type(args[1]) not in (float, int) or args[1] != 0.0625 or kwargs:
                    raise ValueError("Unexpected native attention SCALE signature")
                link(args[0], "layer.1.attention.unscaled_scores")
                name = "layer.1.attention.scaled_scores"
            elif func is torch.ops.aten.add.Tensor and "layer.1.attention.scaled_scores" in tensors and "softmax_f32" not in pending:
                if len(args) != 2 or set(kwargs) - {"alpha"} or kwargs.get("alpha", 1) != 1:
                    raise ValueError("Unexpected native attention mask ADD signature")
                link(args[0], "layer.1.attention.scaled_scores")
                link(args[1], "mask.sliding", identity=False)
                with torch._C._DisableTorchDispatch():
                    if args[1].data_ptr() != tensors["mask.sliding"].data_ptr():
                        raise ValueError("Mask ADD is not the original mask view")
                name = "layer.1.attention.masked_scores"
            elif func is torch.ops.aten._softmax.default:
                if len(args) != 3 or kwargs or args[1] not in (-1, 3) or args[2] is not False:
                    raise ValueError("Unexpected original native FP32 softmax signature")
                input_bits, _ = _tensor_record(args[0], (1, 4, 30, 30), "torch.float32")
                masked = tensors["layer.1.attention.masked_scores"]
                link(masked, "layer.1.attention.masked_scores")
                with torch._C._DisableTorchDispatch():
                    expected_bits = (_bit_tensor(masked).to(torch.int64) << 16).tolist()
                if input_bits != expected_bits:
                    raise ValueError("Native FP32 softmax input is not the original masked BF16 upcast")
                record["checks"]["softmax_native_input_upcast_link"] = True
                output = profiled("softmax", lambda: func(*args, **kwargs))
                pending["softmax_native_bits"], _ = _tensor_record(output, (1, 4, 30, 30), "torch.float32")
                pending["softmax_native"] = output
                return output
            elif func is torch.ops.aten._to_copy.default and args[0] is pending.get("softmax_f32"):
                once("softmax_cast")
                if kwargs.get("dtype") != torch.bfloat16:
                    raise ValueError("Original softmax cast must produce BF16")
                output = func(*args, **kwargs)
                _tensor_record(output, (1, 4, 30, 30), "torch.bfloat16")
                pending["softmax_bf16"] = output
                return output
            else:
                return func(*args, **kwargs)
            output = profiled(name, lambda: func(*args, **kwargs))
            capture(name, output)
            return output

    def eager(module: Any, query: Any, key: Any, value: Any, mask: Any, **kwargs: Any) -> Any:
        if module is not attention or not active["layer"]:
            return original_eager(module, query, key, value, mask, **kwargs)
        if not active["attention"] or active["eager"]:
            raise ValueError("Unexpected original eager attention scope")
        once("eager")
        if kwargs.get("dropout") != 0.0 or module.training or kwargs.get("scaling") != 0.0625 or kwargs.get("softcap") is not None or kwargs.get("sliding_window") != 512 or module.num_key_value_groups != 4:
            raise ValueError("Unexpected eager scaling/dropout/softcap/groups/sliding window")
        link(query, "layer.1.query.rotary")
        link(key, "layer.1.key.rotary")
        link(mask, "mask.sliding")
        with torch._C._DisableTorchDispatch():
            if value.data_ptr() != tensors["layer.1.value.flat"].data_ptr():
                raise ValueError("Eager value lost projection storage lineage")
        active["eager"] = True
        try:
            with AttentionCapture():
                output = original_eager(module, query, key, value, mask, **kwargs)
            mapped(output[0], "layer.1.attention.head_output", lambda bits: bits.transpose(1, 2), False)
            link(output[1], "layer.1.attention.probability")
            return output
        finally:
            active["eager"] = False

    branch_ops = (
        ("layer.1.post_attention_residual", "hidden.1", "layer.1.attention.post_normalized", torch.ops.aten.add.Tensor),
        ("layer.1.mlp.product", "layer.1.mlp.activated_gate", "layer.1.mlp.up", torch.ops.aten.mul.Tensor),
        ("hidden.2", "layer.1.post_attention_residual", "layer.1.mlp.post_normalized", torch.ops.aten.add.Tensor),
    )

    class BranchCapture(TorchDispatchMode):
        def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
            kwargs = kwargs or {}
            for name, left, right, operation in branch_ops:
                if func is operation and right in tensors and len(args) >= 2 and (args[0] is tensors.get(left) or args[1] is tensors[right]):
                    if len(args) != 2 or set(kwargs) - {"alpha"} or kwargs.get("alpha", 1) != 1:
                        raise ValueError("Unexpected native branch operation signature")
                    link(args[0], left)
                    link(args[1], right)
                    output = profiled(name, lambda: func(*args, **kwargs))
                    pending[name] = output
                    link(args[0], left)
                    link(args[1], right)
                    return output
            return func(*args, **kwargs)

    def wrapped_layer(*args: Any, **kwargs: Any) -> Any:
        if not active["layer"]:
            raise ValueError("Target layer forward lacks its native entry hook")
        try:
            with BranchCapture():
                return original_layer(*args, **kwargs)
        finally:
            active.update(layer=False, attention=False, eager=False)

    def install_leaf(stack: ExitStack, module: Any, name: str, source: str, rms: bool = False) -> None:
        original = module.forward
        if rms and module.eps != 1e-6:
            raise ValueError("Unexpected original RMS epsilon")

        def prehook(module: Any, args: Any) -> None:
            if not active["layer"]:
                return
            once("input:" + name)
            value = args[0]
            if source in ("layer.1.query.heads", "layer.1.key.heads"):
                heads = 4 if "query" in source else 1
                mapped(value, source.replace(".heads", ".flat"), lambda bits: bits.reshape(1, 30, heads, 256).transpose(1, 2), True)
                capture(source, value)
            elif source == "layer.1.attention.concatenated":
                mapped(value, "layer.1.attention.head_output", lambda bits: bits.transpose(1, 2).reshape(1, 30, 1024), False)
                capture(source, value)
            elif source in ("layer.1.post_attention_residual", "layer.1.mlp.product"):
                if value is not pending.get(source):
                    raise ValueError("Native branch operation did not reach module input")
                capture(source, value)
            link(value, source)

        def forward(*args: Any, **kwargs: Any) -> Any:
            if not active["layer"]:
                return original(*args, **kwargs)
            if rms:
                if len(args) != 1 or kwargs or name in record["scalar_stages"]:
                    raise ValueError("Unexpected original RMS invocation")
                output, stages = profiled(name, lambda: reference_gemma._observe_rms_module(original, args[0]))
                rows = math.prod(STATE_SPECS[name][0][:-1])
                for stage in ("mean_bits", "denominator_bits", "rsqrt_bits"):
                    if len(stages[stage]) != rows or any(type(bits) is not int or not 0 <= bits <= 0xFFFFFFFF for bits in stages[stage]):
                        raise ValueError("Malformed native FP32 RMS scalar stage")
                metadata = stages["mean_input_metadata"]
                if metadata["input_shape"] != list(STATE_SPECS[name][0]) or metadata["input_dtype"] != "torch.float32" or metadata["axes"] != [-1] or metadata["keepdim"] is not True:
                    raise ValueError("Unexpected native RMS mean geometry")
                record["scalar_stages"][name] = stages
                return output
            return profiled(name, lambda: original(*args, **kwargs))

        def output_hook(module: Any, args: Any, output: Any) -> None:
            if not active["layer"]:
                return
            link(args[0], source)
            capture(name, output)

        handles.append(module.register_forward_pre_hook(prehook))
        handles.append(module.register_forward_hook(output_hook))
        stack.enter_context(patch.object(module, "forward", new=forward))

    stopped = False
    try:
        handles.append(prefix.register_forward_pre_hook(prefix_input))
        handles.append(prefix.register_forward_hook(prefix_output))
        handles.append(layer.register_forward_pre_hook(layer_input, with_kwargs=True))
        handles.append(layer.register_forward_hook(layer_output))
        for module in (*model.model.layers[2:], model.model.norm, model.lm_head):
            handles.append(module.register_forward_pre_hook(forbidden))
        with ExitStack() as stack, torch.no_grad():
            if traced:
                original_rotary, original_repeat = gemma.apply_rotary_pos_emb, gemma.repeat_kv
                original_matmul, original_softmax, original_dropout = torch.matmul, functional.softmax, functional.dropout
                original_eager, original_layer = gemma.eager_attention_forward, layer.forward
                for target, attr, replacement in ((gemma, "apply_rotary_pos_emb", rotary), (gemma, "repeat_kv", repeat),
                                                   (torch, "matmul", matmul), (functional, "softmax", softmax), (functional, "dropout", dropout),
                                                   (gemma, "eager_attention_forward", eager), (layer, "forward", wrapped_layer)):
                    stack.enter_context(patch.object(target, attr, new=replacement))
                handles.append(attention.register_forward_pre_hook(attention_input, with_kwargs=True))
                handles.append(attention.register_forward_hook(attention_output))
                leaves = (
                    (layer.input_layernorm, "layer.1.attention.normalized", "hidden.1", True),
                    (attention.q_proj, "layer.1.query.flat", "layer.1.attention.normalized", False),
                    (attention.k_proj, "layer.1.key.flat", "layer.1.attention.normalized", False),
                    (attention.v_proj, "layer.1.value.flat", "layer.1.attention.normalized", False),
                    (attention.q_norm, "layer.1.query.normalized", "layer.1.query.heads", True),
                    (attention.k_norm, "layer.1.key.normalized", "layer.1.key.heads", True),
                    (attention.o_proj, "layer.1.attention.projected", "layer.1.attention.concatenated", False),
                    (layer.post_attention_layernorm, "layer.1.attention.post_normalized", "layer.1.attention.projected", True),
                    (layer.pre_feedforward_layernorm, "layer.1.mlp.normalized", "layer.1.post_attention_residual", True),
                    (mlp.gate_proj, "layer.1.mlp.gate", "layer.1.mlp.normalized", False),
                    (mlp.up_proj, "layer.1.mlp.up", "layer.1.mlp.normalized", False),
                    (mlp.act_fn, "layer.1.mlp.activated_gate", "layer.1.mlp.gate", False),
                    (mlp.down_proj, "layer.1.mlp.down", "layer.1.mlp.product", False),
                    (layer.post_feedforward_layernorm, "layer.1.mlp.post_normalized", "layer.1.mlp.down", True),
                )
                for module, name, source, rms in leaves:
                    install_leaf(stack, module, name, source, rms)
            model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, output_attentions=False, output_hidden_states=False, logits_to_keep=1)
    except Complete:
        stopped = True
    finally:
        active.update(layer=False, attention=False, eager=False)
        for handle in reversed(handles):
            handle.remove()
    expected = set(STATE_SPECS) if traced else {"hidden.1", "hidden.2"}
    if not stopped or set(record["state_bits"]) != expected or set(record["geometry"]) != expected:
        raise ValueError("Original forward did not stop with complete second-layer state coverage")
    needed = {"prefix_input", "prefix_output", "layer", "layer_output"} | {"state:" + name for name in expected}
    if traced:
        needed.update({"attention", "apply_rotary", "repeat:key", "repeat:value", "eager", "dropout", "softmax_cast"})
        needed.update("input:" + name for _, name, _, _ in leaves)
        native = {name for _, name, _, _ in leaves} | {name for name, _, _, _ in branch_ops} | {
            "layer.1.attention.unscaled_scores", "layer.1.attention.scaled_scores", "layer.1.attention.masked_scores", "layer.1.attention.head_output", "softmax"}
        needed.update("native:" + name for name in native)
        if set(record["kernels"]) != native or len(record["scalar_stages"]) != 6 or not record["softmax_f32_bits"]:
            raise ValueError("Incomplete original native operation occurrence coverage")
        record["checks"].update(original_operations_once=True, all_operand_links_unchanged=True, rms_scalar_stages_complete=True, softmax_f32_complete=True)
    if set(counts) != needed or any(count != 1 for count in counts.values()):
        raise ValueError("Incomplete original native operation occurrence coverage")
    for name in expected:
        link(tensors[name], name)
    for name, (value, bits, geometry) in pending["root_records"].items():
        if _tensor_record(value, *STATE_SPECS[name]) != (bits, geometry):
            raise ValueError("Native local rotary/sliding mask root changed")
    positions, cache_position = pending["positions"]
    if _bit_tensor(positions).tolist() != [list(range(30))] or _bit_tensor(cache_position).tolist() != list(range(30)):
        raise ValueError("Auxiliary position contract changed")
    code_after, runtime_after = _code_sha(), _runtime()
    record.update(code_before=code_before, code_after=code_after, runtime_before=runtime_before, runtime_after=runtime_after)
    record["checks"].update(
        token_commitment_unchanged=token_ids == tokens_before and _bit_tensor(ids).tolist() == tokens_before,
        stopped_at_decoder_tuple=stopped, decoder_tuple_length_one=True, no_later_layer=True,
        no_final_norm=True, no_lm_head=True, all_needed_states_present=True,
        code_unchanged=code_before == code_after, runtime_unchanged=runtime_before == runtime_after,
        target_layer_one=True, local_rotary_and_sliding_mask=True, auxiliary_positions_unchanged=True,
        native_prefix_executed=True, native_prefix_output_is_target_input=True,
        prefix_observed_not_independently_recomputed=True,
    )
    if not traced:
        record["checks"]["plain_minimal_control"] = set(record["state_bits"]) == {"hidden.1", "hidden.2"} and not record["kernels"] and not record["scalar_stages"] and not record["softmax_f32_bits"]
    if not all(record["checks"].values()):
        raise ValueError("Native second-layer capture integrity check failed")
    return record
