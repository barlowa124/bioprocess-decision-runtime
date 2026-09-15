from __future__ import annotations

import json
import math
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig
    from transformers.models.gemma3 import modeling_gemma3 as gemma
    from bioprocess_runtime import gemma_first_layer_capture as capture
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional torch/transformers dependencies unavailable")
class FirstLayerCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1731)
            config = Gemma3TextConfig(
                vocab_size=32, hidden_size=640, intermediate_size=2048, head_dim=256,
                num_attention_heads=4, num_key_value_heads=1, num_hidden_layers=2,
                layer_types=["sliding_attention", "full_attention"], sliding_window=512,
                max_position_embeddings=512, query_pre_attn_scalar=256,
                hidden_activation="gelu_pytorch_tanh", attention_dropout=0.0,
                attn_implementation="eager", pad_token_id=0, bos_token_id=1, eos_token_id=2,
            )
            cls.model = Gemma3ForCausalLM(config).to(device="cpu", dtype=torch.bfloat16).eval()
        if not hasattr(cls.model.model, "rotary_emb_local"):
            torch.set_num_threads(cls.old_threads)
            raise unittest.SkipTest("Requires pinned HF local-RoPE/decoder-tuple API (transformers 4.53.3)")
        cls.ids = [[1 + index % 31 for index in range(30)]]

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(capture, "_runtime", return_value={"device": "cpu", "synthetic_unit_test": True}))
        self.sync = self.stack.enter_context(patch.object(torch.cuda, "synchronize", side_effect=AssertionError("No CUDA in synthetic tests")))
        self.profile_depth = 0
        self.profile_count = 0
        owner = self

        class Profile:
            def __init__(self, **kwargs):
                owner.assertEqual(kwargs["activities"], [torch.profiler.ProfilerActivity.CPU])

            def __enter__(self):
                owner.assertEqual(owner.profile_depth, 0, "Nested profiler is forbidden")
                owner.profile_depth += 1
                owner.profile_count += 1
                return self

            def __exit__(self, *args):
                owner.profile_depth -= 1

            def events(self):
                return [SimpleNamespace(name=name, device_type=torch.autograd.DeviceType.CUDA)
                        for name in ("synthetic_native_z", "synthetic_native_a", "synthetic_native_z")]

        self.stack.enter_context(patch.object(torch.profiler, "profile", new=Profile))

    def snapshot(self):
        return {
            "functions": (gemma.eager_attention_forward, gemma.apply_rotary_pos_emb, gemma.repeat_kv,
                          torch.matmul, torch.nn.functional.embedding, torch.nn.functional.softmax, torch.nn.functional.dropout),
            "modules": [(module, module.forward, dict(module._forward_hooks), dict(module._forward_pre_hooks), "forward" in module.__dict__)
                        for module in self.model.modules()],
        }

    def assert_restored(self, before):
        after = self.snapshot()
        self.assertEqual(before, after)
        self.assertEqual(self.profile_depth, 0)
        self.sync.assert_not_called()

    def test_traced_all_ir_states_and_native_auxiliary_stages(self):
        before = self.snapshot()
        observed = capture.capture_first_layer(self.model, self.ids, True)
        self.assert_restored(before)
        self.assertEqual(set(observed), {"state_bits", "geometry", "scalar_stages", "softmax_f32_bits", "kernels", "checks",
                                         "code_before", "code_after", "runtime_before", "runtime_after"})
        self.assertEqual(len(observed["state_bits"]), 36)
        self.assertEqual(set(observed["state_bits"]), set(capture.STATE_SPECS))
        self.assertNotIn("input_ids", observed["state_bits"])
        self.assertEqual(observed["state_bits"]["position_ids"], [list(range(30))])
        self.assertTrue(all(observed["checks"].values()))
        self.assertEqual(observed["code_before"], observed["code_after"])
        self.assertEqual(observed["runtime_before"], observed["runtime_after"])
        ir = json.loads((Path(__file__).resolve().parents[1] / "results" / "gemma3_270m_execution_ir.json").read_text(encoding="utf-8"))
        cone = [item for item in ir["instructions"] if int(item["id"][1:]) <= 35 and item["id"] not in ("i0003", "i0005")]
        self.assertEqual(len(cone), 34)
        self.assertEqual(set(observed["state_bits"]), {name for item in cone for name in item["outputs"]})
        for name, bits in observed["state_bits"].items():
            declaration = ir["tensors"][name]
            shape = [{"B": 1, "S": 30}.get(size, size) for size in declaration["shape"]]
            values = torch.tensor(bits, dtype=torch.int64)
            self.assertEqual(list(values.shape), shape)
            self.assertTrue(bool((values >= 0).all()))
            self.assertTrue(bool((values <= (29 if name == "position_ids" else 65535)).all()))
            self.assertEqual(observed["geometry"][name]["shape"], shape)
            self.assertEqual(observed["geometry"][name]["dtype"], declaration["dtype"])
            self.assertEqual(observed["geometry"][name]["device"], "cpu")
            self.assertEqual(len(observed["geometry"][name]["strides"]), len(shape))
        expected_rms = {item["outputs"][0] for item in cone if item["opcode"] == "RMS_NORM"}
        self.assertEqual(set(observed["scalar_stages"]), expected_rms)
        positions = 0
        for name, stages in observed["scalar_stages"].items():
            self.assertEqual(set(stages), {"mean_bits", "denominator_bits", "rsqrt_bits", "mean_input_metadata"})
            rows = math.prod(capture.STATE_SPECS[name][0][:-1])
            self.assertEqual(rows, 120 if name == "layer.0.query.normalized" else 30)
            for stage in ("mean_bits", "denominator_bits", "rsqrt_bits"):
                self.assertEqual(len(stages[stage]), rows)
                self.assertTrue(all(type(bits) is int and 0 <= bits <= 0xFFFFFFFF for bits in stages[stage]))
                positions += len(stages[stage])
        self.assertEqual(positions, 810)
        softmax = torch.tensor(observed["softmax_f32_bits"], dtype=torch.int64)
        self.assertEqual(tuple(softmax.shape), (1, 4, 30, 30))
        self.assertEqual(softmax.numel(), 3600)
        self.assertTrue(bool(((softmax >= 0) & (softmax <= 0xFFFFFFFF)).all()))
        self.assertEqual(len(observed["kernels"]), 22)
        self.assertEqual(self.profile_count, 22)
        for names in observed["kernels"].values():
            self.assertEqual(names, ["synthetic_native_a", "synthetic_native_z"])
        self.assertEqual(json.loads(json.dumps(observed)), observed)

    def test_softmax_profile_contains_only_native_fp32_softmax_not_wrapper_or_casts(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        before = self.snapshot()
        original = torch.nn.functional.softmax
        events = []
        owner = self

        def softmax(*args, **kwargs):
            events.append(("wrapper_enter", owner.profile_depth))
            output = original(*args, **kwargs)
            events.append(("wrapper_exit", owner.profile_depth))
            return output

        class Spy(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                if func is torch.ops.aten._softmax.default:
                    owner.assertEqual(len(args), 3)
                    owner.assertEqual(kwargs, {})
                    owner.assertEqual(args[0].dtype, torch.float32)
                    owner.assertEqual(tuple(args[0].shape), (1, 4, 30, 30))
                    owner.assertIn(args[1], (-1, 3))
                    owner.assertIs(args[2], False)
                    events.append(("aten._softmax.default", owner.profile_depth))
                elif func is torch.ops.aten._to_copy.default and tuple(args[0].shape) == (1, 4, 30, 30):
                    events.append(("cast", str(args[0].dtype), str(kwargs.get("dtype")), owner.profile_depth))
                return func(*args, **kwargs)

        with patch.object(torch.nn.functional, "softmax", new=softmax), Spy():
            observed = capture.capture_first_layer(self.model, self.ids, True)
        self.assertEqual(events, [
            ("wrapper_enter", 0),
            ("cast", "torch.bfloat16", "torch.float32", 0),
            ("aten._softmax.default", 1),
            ("wrapper_exit", 0),
            ("cast", "torch.float32", "torch.bfloat16", 0),
        ])
        self.assertEqual(self.profile_count, 22)
        self.assertEqual(len(observed["kernels"]), 22)
        self.assertEqual(observed["kernels"]["softmax"], ["synthetic_native_a", "synthetic_native_z"])
        self.assertTrue(observed["checks"]["softmax_native_input_upcast_link"])
        self.assertTrue(observed["checks"]["softmax_native_output_link"])
        self.assert_restored(before)

    def test_softmax_native_input_and_wrapper_output_lineage_guards(self):
        original = torch.nn.functional.softmax

        def changed_input(input, **kwargs):
            return original(input.float().add_(1.0), **kwargs)

        def changed_output(input, **kwargs):
            return original(input, **kwargs).clone()

        for replacement, message in ((changed_input, "original masked BF16 upcast"),
                                     (changed_output, "native FP32 output lineage")):
            with patch.object(torch.nn.functional, "softmax", new=replacement):
                before = self.snapshot()
                with self.assertRaisesRegex(ValueError, message):
                    capture.capture_first_layer(self.model, self.ids, True)
                self.assert_restored(before)

    def test_plain_is_minimal_and_repeated_calls_restore_every_patch(self):
        before = self.snapshot()
        traced = capture.capture_first_layer(self.model, self.ids, True)
        profile_count = self.profile_count
        with patch.object(capture, "_profile_call", side_effect=AssertionError("Plain must not profile")):
            plain = capture.capture_first_layer(self.model, self.ids, False)
            again = capture.capture_first_layer(self.model, self.ids, False)
        self.assertEqual(self.profile_count, profile_count)
        self.assertEqual(set(plain), {"state_bits", "geometry", "checks", "code_before", "code_after", "runtime_before", "runtime_after"})
        self.assertEqual(set(plain["state_bits"]), {"hidden.1"})
        self.assertEqual(set(plain["geometry"]), {"hidden.1"})
        self.assertEqual(plain["state_bits"], again["state_bits"])
        self.assertEqual(plain["state_bits"]["hidden.1"], traced["state_bits"]["hidden.1"])
        self.assertTrue(plain["checks"]["plain_minimal_control"])
        self.assertTrue(all(plain["checks"].values()))
        self.assert_restored(before)

    def test_bad_decoder_tuple_is_rejected_and_cleaned_up(self):
        layer = self.model.model.layers[0]
        original = layer.forward
        for traced in (False, True):
            for malformed in (lambda result: result[0], lambda result: result + (None,)):
                with patch.object(layer, "forward", new=lambda *args, **kwargs: malformed(original(*args, **kwargs))):
                    before = self.snapshot()
                    with self.assertRaisesRegex(ValueError, "one-element"):
                        capture.capture_first_layer(self.model, self.ids, traced)
                    self.assert_restored(before)

    def test_later_layer_final_norm_and_lm_head_guards(self):
        layer = self.model.model.layers[0]
        value = torch.zeros((1, 30, 640), dtype=torch.bfloat16)
        for target in (self.model.model.layers[1], self.model.model.norm, self.model.lm_head):
            with patch.object(layer, "forward", new=lambda *args, **kwargs: target(value)):
                before = self.snapshot()
                with self.assertRaisesRegex(ValueError, "must not execute"):
                    capture.capture_first_layer(self.model, self.ids, False)
                self.assert_restored(before)

    def test_native_failure_restores_hooks_and_functions(self):
        original = torch.nn.functional.softmax

        def failing(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("synthetic softmax failure")

        with patch.object(torch.nn.functional, "softmax", new=failing):
            before = self.snapshot()
            with self.assertRaisesRegex(RuntimeError, "synthetic softmax failure"):
                capture.capture_first_layer(self.model, self.ids, True)
            self.assert_restored(before)

    def test_dtype_geometry_and_input_guards(self):
        before = self.snapshot()
        with patch.object(self.model.config, "head_dim", 128):
            with self.assertRaisesRegex(ValueError, "head geometry"):
                capture.capture_first_layer(self.model, self.ids, True)
        embed = self.model.model.embed_tokens
        with patch.object(embed, "weight", torch.nn.Parameter(embed.weight.float())):
            with self.assertRaisesRegex(ValueError, "BF16"):
                capture.capture_first_layer(self.model, self.ids, False)
        for ids in ([[1] * 29], [[32] * 30], [[True] * 30], [[-1] * 30]):
            with self.assertRaises(ValueError):
                capture.capture_first_layer(self.model, ids, True)
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            capture._tensor_record(torch.zeros((1, 30, 640)), (1, 30, 640), "torch.bfloat16")
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            capture._tensor_record(torch.zeros((1, 30, 641), dtype=torch.bfloat16), (1, 30, 640), "torch.bfloat16")
        with self.assertRaisesRegex(ValueError, "BF16, FP32 or int64"):
            capture._bit_tensor(torch.zeros(1, dtype=torch.float64))
        self.assert_restored(before)

    def test_runtime_source_and_token_commitment_changes_are_rejected(self):
        before = self.snapshot()
        for attribute, values in (("_runtime", [{"device": "before"}, {"device": "after"}]),
                                  ("_code_sha", ["before", "after"])):
            with patch.object(capture, attribute, side_effect=values):
                with self.assertRaisesRegex(ValueError, "integrity check"):
                    capture.capture_first_layer(self.model, self.ids, False)
            self.assert_restored(before)
        original = self.model.forward

        def mutated(*args, **kwargs):
            try:
                return original(*args, **kwargs)
            finally:
                kwargs["input_ids"][0, 0] = 7

        with patch.object(self.model, "forward", new=mutated):
            patched = self.snapshot()
            with self.assertRaisesRegex(ValueError, "integrity check"):
                capture.capture_first_layer(self.model, self.ids, False)
            self.assert_restored(patched)
        self.assert_restored(before)

    def test_plain_does_not_install_math_wrappers_or_dispatch_modes(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        before = self.snapshot()
        original = self.model.forward

        def checked(*args, **kwargs):
            self.assertEqual(self.snapshot()["functions"], before["functions"])
            for module, forward, hooks, prehooks, own_forward in before["modules"]:
                if module is not self.model:
                    self.assertEqual(module.forward, forward)
                expected_outputs = len(hooks) + int(module is self.model.model.layers[0])
                guards = (self.model.model.layers[1], self.model.model.norm, self.model.lm_head)
                expected_inputs = len(prehooks) + int(any(module is guard for guard in guards))
                self.assertEqual(len(module._forward_hooks), expected_outputs)
                self.assertEqual(len(module._forward_pre_hooks), expected_inputs)
            return original(*args, **kwargs)

        with patch.object(self.model, "forward", new=checked), patch.object(TorchDispatchMode, "__enter__", side_effect=AssertionError("Plain dispatch mode")):
            capture.capture_first_layer(self.model, self.ids, False)
        self.assert_restored(before)

    def test_repeated_native_leaf_is_rejected(self):
        layer = self.model.model.layers[0]
        original = layer.self_attn.q_proj.forward

        def twice(value):
            layer.self_attn.q_proj(value)
            return original(value)

        with patch.object(layer.self_attn.q_proj, "forward", new=twice):
            before = self.snapshot()
            with self.assertRaisesRegex(ValueError, "Repeated original operation"):
                capture.capture_first_layer(self.model, self.ids, True)
            self.assert_restored(before)

    def test_kernel_symbols_required_and_code_hash_covers_observer(self):
        self.assertEqual(len(capture._code_sha()), 64)
        original_hash = capture._code_sha()
        original = capture.reference_gemma._observe_rms_module

        def observer(module, value):
            return original(module, value)

        with patch.object(capture.reference_gemma, "_observe_rms_module", new=observer):
            self.assertNotEqual(capture._code_sha(), original_hash)
        with patch.object(torch.profiler, "profile") as profiler:
            profiler.return_value.__enter__.return_value.events.return_value = []
            with self.assertRaisesRegex(ValueError, "kernel symbols"):
                capture._profile_call(lambda: None, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
