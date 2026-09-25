"""Checks for offline copy/drift acquisition-binding recovery."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bioprocess_runtime import gemma_prompt_tokenization as recovery


ROOT = Path(__file__).resolve().parent.parent

try:
    import jinja2  # noqa: F401
    import tokenizers
    OPTIONAL_DEPS = True
except ImportError:
    OPTIONAL_DEPS = False


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


ASSET_NAMES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
               "added_tokens.json", "chat_template.jinja", "generation_config.json",
               "config.json"]

SYNTHETIC_TEMPLATE = ("{{ bos_token }} <start_of_turn>user\n"
                      "{{ messages[0]['content'] | trim }}<end_of_turn>\n"
                      "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}")


def _plain_asset_dir(root: Path) -> dict[str, str]:
    for name in ASSET_NAMES:
        (root / name).write_text(f"synthetic {name}", encoding="utf-8")
    return {name: _sha((root / name).read_bytes()) for name in ASSET_NAMES}


def _synthetic_model_dir(root: Path) -> dict[str, str]:
    """Build a tiny but real tokenizer-asset directory; return its hash map."""
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(
        {"<bos>": 0, "<start_of_turn>": 1, "<end_of_turn>": 2, "user": 3,
         "model": 4, "[UNK]": 5}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(root / "tokenizer.json"))
    files = {
        "tokenizer_config.json": json.dumps(
            {"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>"}),
        "special_tokens_map.json": json.dumps({"bos_token": "<bos>"}),
        "added_tokens.json": json.dumps({}),
        "chat_template.jinja": SYNTHETIC_TEMPLATE,
        "generation_config.json": json.dumps(
            {"bos_token_id": 0, "eos_token_id": [2], "pad_token_id": 5,
             "cache_implementation": "hybrid", "do_sample": True,
             "top_k": 64, "top_p": 0.95}),
        "config.json": json.dumps(
            {"use_cache": True, "sliding_window": 512,
             "_sliding_window_pattern": 6,
             "layer_types": ["sliding_attention"] * 5 + ["full_attention"]}),
    }
    for name, text in files.items():
        (root / name).write_text(text, encoding="utf-8")
    return {name: _sha((root / name).read_bytes()) for name in ASSET_NAMES}


def _synthetic_sources(root: Path, prompt: str) -> dict[str, Path]:
    heldout = [{"level": "L1", "target": ["m", "k"], "set": "heldout", "prompt": prompt}]
    heldout_path = root / "heldout.json"
    heldout_path.write_text(json.dumps(heldout), encoding="utf-8")
    protocol_path = root / "protocol.md"
    protocol_path.write_text("synthetic protocol", encoding="utf-8")
    probe = {
        "protocol_sha256": _sha(protocol_path.read_bytes()),
        "heldout_sha256": _sha(heldout_path.read_bytes()),
        "environment": {"decode": "greedy"},
        "cases": [{"level": "L1", "target": ["m", "k"], "set": "heldout",
                   "prompt": prompt, "score_class": "misattribution",
                   "generation": {"generated_token_ids": [10, 11, 12],
                                   "digit_token_top5": [
                                       {"step": 1, "token_id": 11, "token": "4",
                                        "top5": [{"token": "4", "logit": 1.0}] * 5}]}}],
    }
    probe_path = root / "probe.json"
    probe_path.write_text(json.dumps(probe), encoding="utf-8")
    return {"probe": probe_path, "protocol": protocol_path, "heldout": heldout_path}


class AssetVerificationTests(unittest.TestCase):
    """Standard-library-only checks over pinned asset bytes."""

    def test_matching_assets_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            pinned = _plain_asset_dir(Path(tmp))
            report = recovery.verify_tokenizer_assets(Path(tmp), pinned)
            self.assertTrue(all(entry["matches_pinned"] for entry in report.values()))

    def test_tampered_asset_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            pinned = _plain_asset_dir(Path(tmp))
            (Path(tmp) / "tokenizer_config.json").write_text("tampered", encoding="utf-8")
            with self.assertRaises(ValueError):
                recovery.verify_tokenizer_assets(Path(tmp), pinned)

    def test_missing_asset_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            pinned = _plain_asset_dir(Path(tmp))
            (Path(tmp) / "chat_template.jinja").unlink()
            with self.assertRaises(ValueError):
                recovery.verify_tokenizer_assets(Path(tmp), pinned)


class PureFunctionTests(unittest.TestCase):
    """Checks that need neither tokenizer nor template dependencies."""

    def test_tokenize_uses_no_added_specials(self):
        class Stub:
            def encode(self, text, add_special_tokens):
                assert add_special_tokens is False
                return type("E", (), {"ids": [1, 2, 3]})()
        self.assertEqual(recovery.tokenize_prompt(Stub(), "anything"), [1, 2, 3])

    def test_decode_settings_reflect_hybrid_cache(self):
        settings = recovery.decode_settings(
            {"cache_implementation": "hybrid", "do_sample": True, "top_k": 64,
             "top_p": 0.95, "eos_token_id": [1, 106], "bos_token_id": 2,
             "pad_token_id": 0},
            {"use_cache": True, "sliding_window": 512,
             "_sliding_window_pattern": 6,
             "layer_types": ["sliding_attention"] * 5 + ["full_attention"]},
            "h" * 64)
        self.assertEqual(settings["effective"]["cache_implementation"], "hybrid")
        self.assertEqual(settings["effective"]["full_attention_layers"], [5])
        self.assertEqual(settings["overridden_by_harness"], {"do_sample": False})
        self.assertFalse(settings["paths_established_equivalent"])

    def test_no_torch_or_transformers_import(self):
        source = (ROOT / "bioprocess_runtime/gemma_prompt_tokenization.py").read_text(
            encoding="utf-8")
        self.assertNotIn("import torch", source)
        self.assertNotIn("import transformers", source)


@unittest.skipUnless(OPTIONAL_DEPS, "requires tokenizers and jinja2")
class OptionalDependencyTests(unittest.TestCase):

    def test_render_single_user_message(self):
        rendered = recovery.render_chat_prompt(
            SYNTHETIC_TEMPLATE, {"bos_token": "<bos>"}, "  body  ")
        self.assertEqual(
            rendered, "<bos> <start_of_turn>user\nbody<end_of_turn>\n<start_of_turn>model\n")

    def test_generation_prompt_flag_flows_through(self):
        rendered = recovery.render_chat_prompt(
            "{{ messages[0]['content'] }}"
            "{% if add_generation_prompt %}GEN{% endif %}", {}, "x")
        self.assertEqual(rendered, "xGEN")

    def _build(self, tmp: str):
        model_dir = Path(tmp) / "model"
        model_dir.mkdir()
        pinned = _synthetic_model_dir(model_dir)
        sources = _synthetic_sources(Path(tmp), "State the value for m. Answer only.")
        tokenizer = recovery._load_tokenizer(model_dir)
        specials = recovery._special_tokens(json.loads(
            (model_dir / "tokenizer_config.json").read_text(encoding="utf-8")))
        rendered = recovery.render_chat_prompt(SYNTHETIC_TEMPLATE, specials,
                                               recovery.BASELINE_PROMPT)
        baseline_path = Path(tmp) / "baseline.json"
        baseline_path.write_text(json.dumps(
            [recovery.tokenize_prompt(tokenizer, rendered)]), encoding="utf-8")
        return model_dir, pinned, sources, baseline_path

    def test_report_recovers_ids_and_contexts(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir, pinned, sources, baseline_path = self._build(tmp)
            report = recovery.build_report(
                model_dir, sources["probe"], sources["protocol"],
                sources["heldout"], baseline_path, pinned)
            self.assertTrue(report["baseline_conformance"]["token_ids_reproduced"])
            self.assertFalse(report["decode_settings"]["paths_established_equivalent"])
            case = report["cases"][0]
            expected_context = case["prompt_token_count"] + 1
            self.assertEqual(case["digit_decision_contexts"][0]
                             ["context_token_count_at_decision"], expected_context)
            self.assertFalse(report["model_inference_performed"])
            repeat = recovery.build_report(
                model_dir, sources["probe"], sources["protocol"],
                sources["heldout"], baseline_path, pinned)
            self.assertEqual(report["report_sha256"], repeat["report_sha256"])

    def test_baseline_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir, pinned, sources, baseline_path = self._build(tmp)
            baseline_path.write_text(json.dumps([[0] * 30]), encoding="utf-8")
            with self.assertRaises(ValueError):
                recovery.build_report(
                    model_dir, sources["probe"], sources["protocol"],
                    sources["heldout"], baseline_path, pinned)

    def test_heldout_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir, pinned, sources, baseline_path = self._build(tmp)
            sealed = json.loads(sources["heldout"].read_text(encoding="utf-8"))
            sealed[0]["prompt"] = "changed after sealing"
            sources["heldout"].write_text(json.dumps(sealed), encoding="utf-8")
            with self.assertRaises(ValueError):
                recovery.build_report(
                    model_dir, sources["probe"], sources["protocol"],
                    sources["heldout"], baseline_path, pinned)

    def test_cli_refuses_to_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir, pinned, sources, baseline_path = self._build(tmp)
            output = Path(tmp) / "out.json"
            output.write_text("existing", encoding="utf-8")
            argv = ["prog", "--model-dir", str(model_dir),
                    "--probe", str(sources["probe"]),
                    "--protocol", str(sources["protocol"]),
                    "--heldout", str(sources["heldout"]),
                    "--baseline-tokens", str(baseline_path),
                    "--output", str(output)]
            with mock.patch.dict(recovery.PINNED_ASSET_SHA256, pinned, clear=True), \
                    mock.patch.object(sys, "argv", argv), \
                    self.assertRaises(SystemExit) as caught:
                recovery.main()
            self.assertNotEqual(caught.exception.code, 0)
            self.assertEqual(output.read_text(encoding="utf-8"), "existing")


if __name__ == "__main__":
    unittest.main()
