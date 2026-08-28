from __future__ import annotations

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None


@unittest.skipUnless(TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE, "Gemma optional dependencies are not installed")
class OperationalSemanticsTests(unittest.TestCase):
    def setUp(self) -> None:
        import torch

        from bioprocess_runtime.operational_semantics import tensor_sha256

        class ToyConfig:
            hidden_size = 4
            num_hidden_layers = 1
            num_attention_heads = 1
            head_dim = 4
            intermediate_size = 4
            vocab_size = 8
            torch_dtype = "float32"

            def to_dict(self):
                return {
                    "hidden_size": self.hidden_size,
                    "num_hidden_layers": self.num_hidden_layers,
                    "num_attention_heads": self.num_attention_heads,
                    "head_dim": self.head_dim,
                    "intermediate_size": self.intermediate_size,
                    "vocab_size": self.vocab_size,
                    "torch_dtype": self.torch_dtype,
                    "model_type": "toy",
                }

        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = ToyConfig()
                self.embedding = torch.nn.Embedding(8, 4)
                self.projection = torch.nn.Linear(4, 8, bias=False)
                self.projection.weight = self.embedding.weight
                with torch.no_grad():
                    self.embedding.weight.copy_(torch.arange(32, dtype=torch.float32).reshape(8, 4) / 32)

            def forward(self, input_ids, attention_mask=None, use_cache=False):
                hidden = self.embedding(input_ids)
                return SimpleNamespace(logits=self.projection(hidden))

            def generate(self, input_ids, attention_mask=None, max_new_tokens=1, **kwargs):
                generated = input_ids.clone()
                for _ in range(max_new_tokens):
                    logits = self(input_ids=generated, attention_mask=attention_mask, use_cache=False).logits[0, -1]
                    token = torch.argmax(logits).reshape(1, 1)
                    generated = torch.cat((generated, token), dim=1)
                    if attention_mask is not None:
                        attention_mask = torch.cat((attention_mask, torch.ones_like(token)), dim=1)
                return generated

        class ToyTokenizer:
            chat_template = None
            vocab_size = 8
            bos_token_id = 1
            eos_token_id = 7
            pad_token_id = 0

            def __call__(self, prompt, return_tensors="pt"):
                ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
                return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

            def decode(self, token_ids, skip_special_tokens=False):
                return " ".join(f"token-{token}" for token in token_ids)

        self.torch = torch
        self.tensor_sha256 = tensor_sha256
        self.model = ToyModel()
        self.tokenizer = ToyTokenizer()

    def test_tensor_hash_is_value_sensitive(self) -> None:
        first = self.torch.tensor([1.0, 2.0])
        second = self.torch.tensor([1.0, 3.0])
        self.assertNotEqual(self.tensor_sha256(first), self.tensor_sha256(second))
        self.assertEqual(self.tensor_sha256(first), self.tensor_sha256(first.clone()))

    def test_non_finite_scalars_are_encoded_for_canonical_traces(self) -> None:
        from bioprocess_runtime.operational_semantics import _value_descriptor, append_chain_record

        descriptor = _value_descriptor(float("inf"), {})
        records = []
        append_chain_record(records, {"value": descriptor})
        self.assertEqual(descriptor, {"kind": "non_finite_float", "value": "Infinity"})

    def test_trace_chain_detects_tampering(self) -> None:
        from bioprocess_runtime.operational_semantics import append_chain_record, verify_trace_chain

        records = []
        append_chain_record(records, {"operator": "first"})
        append_chain_record(records, {"operator": "second"})
        root = records[-1]["record_hash"]
        self.assertTrue(verify_trace_chain(records, root)["valid"])
        damaged = copy.deepcopy(records)
        damaged[0]["payload"]["operator"] = "changed"
        self.assertFalse(verify_trace_chain(damaged, root)["valid"])

    def test_architecture_manifest_records_tied_parameters(self) -> None:
        from bioprocess_runtime.operational_semantics import build_architecture_manifest

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "model.bin").write_bytes(b"toy")
            manifest = build_architecture_manifest(self.model, self.tokenizer, path)
        tied = manifest["model"]["tied_parameter_names"]
        self.assertTrue(any({"embedding.weight", "projection.weight"}.issubset(set(names)) for names in tied))
        self.assertEqual(len(manifest["manifest_sha256"]), 64)

    def test_module_trace_predicts_same_token_as_generate(self) -> None:
        from bioprocess_runtime.operational_semantics import predict_with_provenance

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "model.bin").write_bytes(b"toy")
            result = predict_with_provenance(self.model, self.tokenizer, path, "test", trace_level="module")
        self.assertTrue(result["trace"]["chain_verification"]["valid"])
        self.assertTrue(result["reference_generate_comparison"]["exact_match"])
        self.assertGreater(result["trace"]["record_count"], 0)

    def test_aten_trace_predicts_same_token_as_generate(self) -> None:
        from bioprocess_runtime.operational_semantics import predict_with_provenance

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "model.bin").write_bytes(b"toy")
            result = predict_with_provenance(self.model, self.tokenizer, path, "test", trace_level="aten")
        self.assertTrue(result["trace"]["chain_verification"]["valid"])
        self.assertTrue(result["reference_generate_comparison"]["exact_match"])
        operators = {record["payload"]["operator"] for record in result["trace"]["records"]}
        self.assertTrue(any("argmax" in operator for operator in operators))

    def test_compact_summary_is_derived_from_full_artifacts(self) -> None:
        from bioprocess_runtime.operational_semantics import (
            build_architecture_manifest,
            predict_with_provenance,
            summarize_operational_evidence,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "model.bin").write_bytes(b"toy")
            manifest = build_architecture_manifest(self.model, self.tokenizer, path)
            module = predict_with_provenance(self.model, self.tokenizer, path, "test", trace_level="module")
            aten = predict_with_provenance(self.model, self.tokenizer, path, "test", trace_level="aten")
            summary = summarize_operational_evidence(manifest, module, aten, "toy/model")
        self.assertEqual(summary["prediction"]["output_commitment_sha256"], aten["prediction"]["output_commitment_sha256"])
        self.assertTrue(summary["prediction"]["transformers_generate_exact_match"])
        self.assertGreater(summary["aten_trace"]["record_count"], 0)


if __name__ == "__main__":
    unittest.main()
