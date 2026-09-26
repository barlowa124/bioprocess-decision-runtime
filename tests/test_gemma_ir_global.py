import hashlib
import unittest

from bioprocess_runtime.gemma_ir import compile_gemma_ir, verify_gemma_ir
from bioprocess_runtime.serialization import canonical_json


ZERO_HASH = "0" * 64
BASE_CONFIG = {
    "num_hidden_layers": 2,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 256,
    "sliding_window": 8,
    "rms_norm_eps": 1e-6,
    "torch_dtype": "bfloat16",
    "layer_types": ["sliding_attention", "full_attention"],
}


def _params(layers, qk_norm=True, local_rotary=True, embed_scale=True,
            lm_head=True):
    names = {
        "model.embed_tokens.weight",
        "model.rotary_emb.inv_freq",
        "model.norm.weight",
    }
    if embed_scale:
        names.add("model.embed_tokens.embed_scale")
    if local_rotary:
        names.add("model.rotary_emb_local.inv_freq")
    if lm_head:
        names.add("lm_head.weight")
    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        names.update({
            f"{prefix}.self_attn.q_proj.weight",
            f"{prefix}.self_attn.k_proj.weight",
            f"{prefix}.self_attn.v_proj.weight",
            f"{prefix}.self_attn.o_proj.weight",
            f"{prefix}.mlp.gate_proj.weight",
            f"{prefix}.mlp.up_proj.weight",
            f"{prefix}.mlp.down_proj.weight",
            f"{prefix}.input_layernorm.weight",
            f"{prefix}.post_attention_layernorm.weight",
            f"{prefix}.pre_feedforward_layernorm.weight",
            f"{prefix}.post_feedforward_layernorm.weight",
        })
        if qk_norm:
            names.update({
                f"{prefix}.self_attn.q_norm.weight",
                f"{prefix}.self_attn.k_norm.weight",
            })
    return sorted(names)


def _manifest(config, parameter_names, model_class="Gemma3ForCausalLM"):
    body = {
        "model": {
            "class": model_class,
            "config": config,
            "parameter_tensors": [
                {"name": name, "shape": [1, 1], "dtype": "torch.bfloat16",
                 "numel": 1, "sha256": ZERO_HASH}
                for name in parameter_names
            ],
            "buffer_tensors": [],
        },
        "tokenizer": {}, "modules": [], "runtime": {},
    }
    body["manifest_sha256"] = hashlib.sha256(
        canonical_json(body).encode("utf-8")).hexdigest()
    return body


def _checks(result):
    return {key: value for key, value in result.items() if key.endswith("_valid") or key == "valid"}


class GemmaIrGlobalTests(unittest.TestCase):

    def test_gemma3_270m_layout_unchanged(self):
        # The pinned-270m structure: 18 layers, dual rotary, q/k norms,
        # embed scale, separate lm_head, 5:1 sliding pattern.
        config = dict(BASE_CONFIG, **{
            "num_hidden_layers": 18,
            "hidden_size": 640,
            "intermediate_size": 2048,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "vocab_size": 262144,
            "sliding_window": 512,
            "layer_types": ["sliding_attention"] * 5 + ["full_attention"] * 13,
        })
        config["layer_types"] = [
            "full_attention" if (i + 1) % 6 == 0 else "sliding_attention"
            for i in range(18)
        ]
        manifest = _manifest(config, _params(18))
        program = compile_gemma_ir(manifest)
        # 29 instructions/layer * 18 + 7 prologue + 4 epilogue = 533 — the
        # documented pinned-270m program size (no softcap: Gemma-3 dropped
        # it; the SOFTCAP emit serves Gemma-2-family checkpoints).
        # documented pinned-270m program size. Regression: the generalized
        # compiler must not change this structure.
        self.assertEqual(len(program["instructions"]), 533)
        checks = _checks(verify_gemma_ir(program, manifest))
        self.assertTrue(all(checks.values()), checks)

    def test_rope_scaling_linear_global_rotary(self):
        # Gemma-3 4b+ style: linear rope_scaling on the global rotary.
        config = dict(BASE_CONFIG,
                      rope_scaling={"factor": 8.0, "rope_type": "linear"})
        manifest = _manifest(config, _params(2))
        program = compile_gemma_ir(manifest)
        rotary = [i for i in program["instructions"]
                  if i["opcode"] == "ROTARY_TABLE"]
        self.assertEqual(len(rotary), 2)
        global_table = next(i for i in rotary
                            if "rotary.global.cosine" in i["outputs"])
        self.assertEqual(global_table["attributes"]["position_scaling"], 8.0)
        local_table = next(i for i in rotary
                           if "rotary.local.cosine" in i["outputs"])
        self.assertNotIn("position_scaling", local_table["attributes"])
        self.assertEqual(
            program["configuration"]["rotary_position_scaling"], 8.0)
        checks = _checks(verify_gemma_ir(program, manifest))
        self.assertTrue(all(checks.values()), checks)

    def test_unsupported_rope_type_rejected(self):
        config = dict(BASE_CONFIG,
                      rope_scaling={"factor": 8.0, "rope_type": "yarn"})
        manifest = _manifest(config, _params(2))
        with self.assertRaises(ValueError):
            compile_gemma_ir(manifest)

    def test_gemma2_style_no_qknorm_single_rotary(self):
        # Gemma-2 shape: no layer_types (pattern-derived), no q/k norms,
        # single rotary, softcaps on both ends, tied head (no lm_head).
        config = {
            "num_hidden_layers": 4,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 256,
            "sliding_window": 8,
            "sliding_window_pattern": 2,
            "rms_norm_eps": 1e-6,
            "torch_dtype": "bfloat16",
            "attn_logit_softcapping": 50.0,
            "final_logit_softcapping": 30.0,
        }
        manifest = _manifest(
            config, _params(4, qk_norm=False, local_rotary=False,
                            embed_scale=False, lm_head=False))
        program = compile_gemma_ir(manifest)
        self.assertEqual(
            program["configuration"]["layer_types"],
            ["sliding_attention", "full_attention"] * 2)
        ops = [i["opcode"] for i in program["instructions"]]
        self.assertIn("SOFTCAP", ops)
        rotaries = [i for i in program["instructions"]
                    if i["opcode"] == "ROTARY_TABLE"]
        self.assertEqual(len(rotaries), 1)   # single global table
        head = [i for i in program["instructions"]
                if "logits.last" in i["outputs"]][0]
        self.assertEqual(head["parameter_refs"],
                         ["model.embed_tokens.weight"])
        checks = _checks(verify_gemma_ir(program, manifest))
        self.assertTrue(all(checks.values()), checks)

    def test_uniform_full_attention_no_sliding(self):
        config = dict(BASE_CONFIG, sliding_window=None,
                      layer_types=["full_attention", "full_attention"])
        manifest = _manifest(config, _params(2, local_rotary=False))
        program = compile_gemma_ir(manifest)
        masks = [i for i in program["instructions"]
                 if i["opcode"] == "CAUSAL_MASK"]
        self.assertEqual(len(masks), 1)  # only mask.full
        checks = _checks(verify_gemma_ir(program, manifest))
        self.assertTrue(all(checks.values()), checks)

    def test_sliding_layer_without_window_rejected(self):
        config = dict(BASE_CONFIG, sliding_window=None)
        manifest = _manifest(config, _params(2))
        with self.assertRaises(ValueError):
            compile_gemma_ir(manifest)

    def test_scalar_embed_scale_when_buffer_absent(self):
        config = dict(BASE_CONFIG)
        manifest = _manifest(config, _params(2, embed_scale=False))
        program = compile_gemma_ir(manifest)
        scale = [i for i in program["instructions"]
                 if i["opcode"] == "SCALE"
                 and "hidden.0" in i["outputs"]][0]
        self.assertEqual(scale["parameter_refs"], [])
        self.assertEqual(scale["attributes"]["scalar"], 8.0)  # sqrt(64)


if __name__ == "__main__":
    unittest.main()
