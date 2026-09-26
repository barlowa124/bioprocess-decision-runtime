"""Execution smoke tests for the generalized IR compiler.

Compile synthetic (no-weights) manifests for the standard and Gemma
decoder layouts, run the interpreter against randomly initialized
tensors, and compare logits bit-for-bit against a reference forward
pass written directly from the declared opcode equations. This is an
interpreter/reference agreement check on synthetic parameters — not a
checkpoint certificate.
"""

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from torch.nn import functional as F
except ImportError:
    raise unittest.SkipTest("requires .[gemma] extras")

from bioprocess_runtime.gemma_ir import compile_gemma_ir, verify_gemma_ir
from bioprocess_runtime.gemma_ir_interpreter import execute_gemma_ir
from bioprocess_runtime.serialization import canonical_json


ZERO_HASH = "0" * 64
BUFFERS = {"model.rotary_emb.inv_freq", "model.rotary_emb_local.inv_freq",
           "model.embed_tokens.embed_scale"}


def _manifest(config, shapes, model_class):
    parameters, buffers = [], []
    for name, shape in sorted(shapes.items()):
        entry = {"name": name, "shape": list(shape), "dtype": "torch.float32",
                 "numel": int(torch.tensor(shape).prod().item()) if shape else 1,
                 "sha256": ZERO_HASH}
        (buffers if name in BUFFERS else parameters).append(entry)
    body = {"model": {"class": model_class, "config": config,
                      "parameter_tensors": parameters, "buffer_tensors": buffers},
            "tokenizer": {}, "modules": [], "runtime": {}}
    body["manifest_sha256"] = hashlib.sha256(
        canonical_json(body).encode("utf-8")).hexdigest()
    return body


def _fill(shapes, family):
    gen = torch.Generator().manual_seed(7)
    tensors = {}
    for name, shape in shapes.items():
        if "inv_freq" in name:
            half = shape[0]
            theta = 10000.0 if "local" in name else 1000000.0
            tensors[name] = theta ** (-torch.arange(half, dtype=torch.float32) * 2.0 / (half * 2))
        elif name.endswith("embed_scale"):
            tensors[name] = torch.full(shape, 8.0)
        elif "layernorm" in name or name == "model.norm.weight" or name.endswith("_norm.weight"):
            base = torch.zeros(shape) if family == "gemma" else torch.ones(shape)
            tensors[name] = base + torch.randn(shape, generator=gen) * 0.02
        else:
            tensors[name] = torch.randn(shape, generator=gen) * 0.05
    return tensors


def _rms(x, w, eps, offset):
    normalized = x.float() * torch.rsqrt(
        x.float().pow(2).mean(-1, keepdim=True) + eps)
    if offset:
        # Gemma ordering: multiply in float32, cast once.
        return (normalized * (1.0 + w.float())).type_as(x)
    # Qwen/Llama ordering: cast first, then multiply by the bf16 weight.
    return w * normalized.type_as(x)


def _rotary_table(inv_freq, positions, batch, scale=1.0, position_scale=None):
    freq = inv_freq[None, :, None].float().expand(batch, -1, 1)
    pos = positions[:, None, :].float()
    if position_scale is not None:
        pos = pos / float(position_scale)
    embedding = torch.cat(((freq @ pos).transpose(1, 2),) * 2, dim=-1)
    return (embedding.cos() * scale), (embedding.sin() * scale)


def _rotate_half(value):
    half = value.shape[-1] // 2
    return torch.cat((-value[..., half:], value[..., :half]), dim=-1)


def _repeat_kv(value, repetitions):
    batch, heads, sequence, dimension = value.shape
    return value[:, :, None, :, :].expand(
        batch, heads, repetitions, sequence, dimension
    ).reshape(batch, heads * repetitions, sequence, dimension)


def _mask(sequence, window):
    row = torch.arange(sequence)[:, None]
    column = torch.arange(sequence)[None, :]
    allowed = column <= row
    if window is not None:
        allowed = allowed & (column > row - window)
    return torch.where(
        allowed,
        torch.zeros(sequence, sequence),
        torch.full((sequence, sequence), torch.finfo(torch.float32).min),
    )[None, None]


def _attention(params, prefix, x, cosine, sine, mask, heads, kv_heads, head_dim,
               scale, biased, norm_fn=None):
    batch, sequence, _ = x.shape
    def project(name, count):
        weight = params[f"{prefix}.self_attn.{name}_proj.weight"]
        bias = params.get(f"{prefix}.self_attn.{name}_proj.bias") if biased else None
        return F.linear(x, weight, bias).view(
            batch, sequence, count, head_dim).transpose(1, 2)
    query, key, value = project("q", heads), project("k", kv_heads), project("v", kv_heads)
    if norm_fn is not None:
        query = norm_fn(query, params[f"{prefix}.self_attn.q_norm.weight"])
        key = norm_fn(key, params[f"{prefix}.self_attn.k_norm.weight"])
    c, s = cosine.unsqueeze(1), sine.unsqueeze(1)
    query = query * c + _rotate_half(query) * s
    key = key * c + _rotate_half(key) * s
    repetitions = heads // kv_heads
    key = _repeat_kv(key, repetitions)
    value = _repeat_kv(value, repetitions)
    scores = query @ key.transpose(2, 3) * scale + mask
    probability = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
    output = (probability @ value).transpose(1, 2).reshape(batch, sequence, -1)
    return F.linear(output, params[f"{prefix}.self_attn.o_proj.weight"])


def _standard_forward(params, ids, layers, heads, kv_heads, head_dim, eps):
    hidden = F.embedding(ids, params["model.embed_tokens.weight"])
    batch, sequence = ids.shape
    positions = torch.arange(sequence).unsqueeze(0)
    cosine, sine = _rotary_table(params["model.rotary_emb.inv_freq"], positions, batch)
    mask = _mask(sequence, None)
    scale = float(head_dim) ** -0.5
    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        normalized = _rms(hidden, params[f"{prefix}.input_layernorm.weight"], eps, False)
        attended = _attention(params, prefix, normalized, cosine, sine, mask,
                              heads, kv_heads, head_dim, scale, biased=True)
        hidden = hidden + attended
        normalized = _rms(hidden, params[f"{prefix}.post_attention_layernorm.weight"], eps, False)
        product = F.silu(F.linear(normalized, params[f"{prefix}.mlp.gate_proj.weight"])) * \
            F.linear(normalized, params[f"{prefix}.mlp.up_proj.weight"])
        hidden = hidden + F.linear(product, params[f"{prefix}.mlp.down_proj.weight"])
    hidden = _rms(hidden, params["model.norm.weight"], eps, False)
    return F.linear(hidden[:, -1:], params["lm_head.weight"])


def _gemma_forward(params, ids, layers, heads, kv_heads, head_dim, eps,
                   layer_types, sliding_window, scale, position_scale):
    hidden = F.embedding(ids, params["model.embed_tokens.weight"]) * \
        params["model.embed_tokens.embed_scale"]
    batch, sequence = ids.shape
    positions = torch.arange(sequence).unsqueeze(0)
    g_cos, g_sin = _rotary_table(params["model.rotary_emb.inv_freq"], positions,
                                 batch, position_scale=position_scale)
    l_cos, l_sin = _rotary_table(params["model.rotary_emb_local.inv_freq"], positions, batch)
    mask_full = _mask(sequence, None)
    mask_sliding = _mask(sequence, sliding_window)
    norm = lambda value, weight: _rms(value, weight, eps, True)
    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        normalized = norm(hidden, params[f"{prefix}.input_layernorm.weight"])
        sliding = layer_types[layer] == "sliding_attention"
        cosine, sine = (l_cos, l_sin) if sliding else (g_cos, g_sin)
        attended = _attention(params, prefix, normalized, cosine, sine,
                              mask_sliding if sliding else mask_full,
                              heads, kv_heads, head_dim, scale,
                              biased=False, norm_fn=norm)
        hidden = hidden + norm(attended, params[f"{prefix}.post_attention_layernorm.weight"])
        normalized = norm(hidden, params[f"{prefix}.pre_feedforward_layernorm.weight"])
        product = F.gelu(F.linear(normalized, params[f"{prefix}.mlp.gate_proj.weight"]), approximate="tanh") * \
            F.linear(normalized, params[f"{prefix}.mlp.up_proj.weight"])
        down = F.linear(product, params[f"{prefix}.mlp.down_proj.weight"])
        hidden = hidden + norm(down, params[f"{prefix}.post_feedforward_layernorm.weight"])
    hidden = norm(hidden, params["model.norm.weight"])
    return F.linear(hidden[:, -1:], params["lm_head.weight"])


def _qwen_fixture():
    config = {
        "model_type": "qwen2", "num_hidden_layers": 2, "hidden_size": 64,
        "intermediate_size": 128, "num_attention_heads": 4,
        "num_key_value_heads": 2, "vocab_size": 64, "rms_norm_eps": 1e-6,
        "torch_dtype": "float32", "use_sliding_window": False,
        "sliding_window": 32768, "hidden_act": "silu", "rope_theta": 1000000.0,
    }
    shapes = {"model.embed_tokens.weight": [64, 64], "lm_head.weight": [64, 64],
              "model.norm.weight": [64], "model.rotary_emb.inv_freq": [8]}
    for layer in range(2):
        prefix = f"model.layers.{layer}"
        shapes.update({
            f"{prefix}.input_layernorm.weight": [64],
            f"{prefix}.post_attention_layernorm.weight": [64],
            f"{prefix}.self_attn.q_proj.weight": [64, 64],
            f"{prefix}.self_attn.q_proj.bias": [64],
            f"{prefix}.self_attn.k_proj.weight": [32, 64],
            f"{prefix}.self_attn.k_proj.bias": [32],
            f"{prefix}.self_attn.v_proj.weight": [32, 64],
            f"{prefix}.self_attn.v_proj.bias": [32],
            f"{prefix}.self_attn.o_proj.weight": [64, 64],
            f"{prefix}.mlp.gate_proj.weight": [128, 64],
            f"{prefix}.mlp.up_proj.weight": [128, 64],
            f"{prefix}.mlp.down_proj.weight": [64, 128],
        })
    return config, shapes


def _gemma_fixture():
    config = {
        "model_type": "gemma3_text", "num_hidden_layers": 2, "hidden_size": 64,
        "intermediate_size": 128, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 64,
        "rms_norm_eps": 1e-6, "torch_dtype": "float32", "sliding_window": 4,
        "layer_types": ["sliding_attention", "full_attention"],
        "hidden_act": "gelu_pytorch_tanh", "query_pre_attn_scalar": 64.0,
        "rope_scaling": {"rope_type": "linear", "factor": 2.0},
    }
    shapes = {"model.embed_tokens.weight": [64, 64],
              "model.embed_tokens.embed_scale": [],
              "lm_head.weight": [64, 64], "model.norm.weight": [64],
              "model.rotary_emb.inv_freq": [8],
              "model.rotary_emb_local.inv_freq": [8]}
    for layer in range(2):
        prefix = f"model.layers.{layer}"
        shapes.update({
            f"{prefix}.input_layernorm.weight": [64],
            f"{prefix}.post_attention_layernorm.weight": [64],
            f"{prefix}.pre_feedforward_layernorm.weight": [64],
            f"{prefix}.post_feedforward_layernorm.weight": [64],
            f"{prefix}.self_attn.q_proj.weight": [64, 64],
            f"{prefix}.self_attn.k_proj.weight": [32, 64],
            f"{prefix}.self_attn.v_proj.weight": [32, 64],
            f"{prefix}.self_attn.q_norm.weight": [16],
            f"{prefix}.self_attn.k_norm.weight": [16],
            f"{prefix}.self_attn.o_proj.weight": [64, 64],
            f"{prefix}.mlp.gate_proj.weight": [128, 64],
            f"{prefix}.mlp.up_proj.weight": [128, 64],
            f"{prefix}.mlp.down_proj.weight": [64, 128],
        })
    return config, shapes


class SyntheticExecutionTests(unittest.TestCase):

    def _check(self, config, shapes, model_class, family, reference):
        manifest = _manifest(config, shapes, model_class)
        program = compile_gemma_ir(manifest)
        self.assertTrue(verify_gemma_ir(program, manifest)["valid"])
        params = _fill(shapes, family)
        ids = torch.tensor([[3, 11, 7, 40, 12, 55]], dtype=torch.int64)
        execution = execute_gemma_ir(program, params, ids)
        expected = reference(params, ids, config)
        self.assertTrue(torch.equal(execution.logits, expected),
                        (execution.logits - expected).abs().max().item())
        self.assertTrue(torch.equal(
            execution.selected_token_id, expected[:, -1, :].argmax(dim=-1)))

    def test_standard_family_executes_and_matches_reference(self):
        config, shapes = _qwen_fixture()
        reference = lambda params, ids, cfg: _standard_forward(
            params, ids, cfg["num_hidden_layers"], cfg["num_attention_heads"],
            cfg["num_key_value_heads"], cfg["hidden_size"] // cfg["num_attention_heads"],
            cfg["rms_norm_eps"])
        self._check(config, shapes, "Qwen2ForCausalLM", "standard", reference)

    def test_gemma_family_executes_and_matches_reference(self):
        config, shapes = _gemma_fixture()
        reference = lambda params, ids, cfg: _gemma_forward(
            params, ids, cfg["num_hidden_layers"], cfg["num_attention_heads"],
            cfg["num_key_value_heads"], cfg["head_dim"], cfg["rms_norm_eps"],
            cfg["layer_types"], cfg["sliding_window"],
            float(cfg["query_pre_attn_scalar"]) ** -0.5,
            cfg["rope_scaling"]["factor"])
        self._check(config, shapes, "Gemma3ForCausalLM", "gemma", reference)


@unittest.skipUnless(
    importlib.util.find_spec("transformers") is not None,
    "transformers is not installed",
)
class HuggingFaceEquivalenceTests(unittest.TestCase):
    """Randomly initialized Qwen2 checkpoint: IR interpreter output vs HF
    eager forward. At bf16 — the dtype real Qwen checkpoints ship in — the
    agreement is bit-identical (torch.equal). At fp32, HF's internal op
    ordering diverges by one ulp (observed max abs diff 1.19e-07), so fp32
    is asserted against a tight bound instead of equality. The fused SDPA
    path is excluded entirely: it is a different reduction schedule and
    legitimately diverges in bf16 (~0.55 observed)."""

    def _check(self, dtype, exact):
        import tempfile as _tempfile
        from transformers import Qwen2Config, Qwen2ForCausalLM

        config = Qwen2Config(
            vocab_size=99, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            use_sliding_window=False, tie_word_embeddings=True,
            max_position_embeddings=64, rope_theta=1000000.0,
            rms_norm_eps=1e-6, torch_dtype=dtype,
        )
        model = Qwen2ForCausalLM(config).to(dtype).eval()
        model.config._attn_implementation = "eager"
        tokenizer = type("Tokenizer", (), {
            "vocab_size": 99, "bos_token_id": 1,
            "eos_token_id": 2, "pad_token_id": 0})()
        with _tempfile.TemporaryDirectory() as directory:
            from bioprocess_runtime.operational_semantics import (
                build_architecture_manifest)
            manifest = build_architecture_manifest(
                model, tokenizer, Path(directory))
        program = compile_gemma_ir(manifest)
        self.assertTrue(verify_gemma_ir(program, manifest)["valid"])
        from bioprocess_runtime.gemma_ir_interpreter import bind_model_tensors
        params = bind_model_tensors(program, model)
        ids = torch.tensor([[3, 11, 7, 40, 12]], dtype=torch.int64)
        with torch.no_grad():
            expected = model(ids).logits[:, -1:, :]
        execution = execute_gemma_ir(program, params, ids)
        if exact:
            self.assertTrue(torch.equal(execution.logits, expected))
        else:
            self.assertTrue(torch.allclose(
                execution.logits.float(), expected.float(), atol=1e-5))
        self.assertTrue(torch.equal(
            execution.selected_token_id, expected[:, -1, :].argmax(dim=-1)))

    def test_random_qwen2_bf16_logits_equal_eager(self):
        self._check(torch.bfloat16, exact=True)

    def test_random_qwen2_fp32_logits_match_eager_within_ulp(self):
        self._check(torch.float32, exact=False)


if __name__ == "__main__":
    unittest.main()
