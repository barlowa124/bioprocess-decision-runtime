from __future__ import annotations

import copy
import hashlib
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
    from bioprocess_runtime import gemma_second_layer_capture as capture
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional torch/transformers dependencies unavailable")
class SecondLayerCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1732)
            config = Gemma3TextConfig(
                vocab_size=32, hidden_size=640, intermediate_size=2048, head_dim=256,
                num_attention_heads=4, num_key_value_heads=1, num_hidden_layers=3,
                layer_types=["sliding_attention", "sliding_attention", "full_attention"], sliding_window=512,
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
            "layers": tuple(self.model.model.layers),
            "config": copy.deepcopy(self.model.config.to_dict()),
            "parameters": [(name, id(value), value._version, tuple(value.shape), value.dtype, value.device)
                           for name, value in self.model.named_parameters()],
        }

    def assert_restored(self, before):
        self.assertEqual(before, self.snapshot())
        self.assertEqual(self.profile_depth, 0)
        self.sync.assert_not_called()

    def test_traced_exact_layer_one_ir_states_and_auxiliary_stages(self):
        before = self.snapshot()
        observed = capture.capture_second_layer(self.model, self.ids, True)
        self.assert_restored(before)
        self.assertEqual(set(observed), {"state_bits", "geometry", "scalar_stages", "softmax_f32_bits", "kernels", "checks",
                                         "code_before", "code_after", "runtime_before", "runtime_after"})
        self.assertEqual(len(observed["state_bits"]), 34)
        self.assertEqual(set(observed["state_bits"]), set(capture.STATE_SPECS))
        self.assertEqual(set(observed["geometry"]), set(capture.STATE_SPECS))
        self.assertNotIn("position_ids", observed["state_bits"])
        self.assertNotIn("hidden.0", observed["state_bits"])
        self.assertNotIn("embedding_unscaled", observed["state_bits"])
        self.assertTrue(all(observed["checks"].values()))
        required = {"token_commitment_unchanged", "stopped_at_decoder_tuple", "decoder_tuple_length_one", "no_later_layer",
                    "no_final_norm", "no_lm_head", "all_needed_states_present", "code_unchanged", "runtime_unchanged",
                    "target_layer_one", "local_rotary_and_sliding_mask", "original_operations_once", "all_operand_links_unchanged",
                    "rms_scalar_stages_complete", "softmax_f32_complete", "native_prefix_executed",
                    "prefix_observed_not_independently_recomputed"}
        self.assertLessEqual(required, set(observed["checks"]))
        self.assertEqual(observed["code_before"], observed["code_after"])
        self.assertEqual(observed["runtime_before"], observed["runtime_after"])
        ir = json.loads((Path(__file__).resolve().parents[1] / "results" / "gemma3_270m_execution_ir.json").read_text(encoding="utf-8"))
        self.assertEqual(ir["configuration"]["layer_types"][:6], ["sliding_attention"] * 5 + ["full_attention"])
        cone = [item for item in ir["instructions"] if 36 <= int(item["id"][1:]) <= 64]
        roots = {"hidden.1", "rotary.local.cosine", "rotary.local.sine", "mask.sliding"}
        outputs = {name for item in cone for name in item["outputs"]}
        self.assertEqual(len(cone), 29)
        self.assertEqual(len(outputs), 30)
        self.assertEqual(set(observed["state_bits"]), outputs | roots)
        for name, bits in observed["state_bits"].items():
            declaration = ir["tensors"][name]
            shape = [{"B": 1, "S": 30}.get(size, size) for size in declaration["shape"]]
            values = torch.tensor(bits, dtype=torch.int64)
            self.assertEqual(list(values.shape), shape)
            self.assertTrue(bool(((values >= 0) & (values <= 65535)).all()))
            geometry = observed["geometry"][name]
            self.assertEqual(set(geometry), {"shape", "dtype", "strides", "device", "alignment_mod16"})
            self.assertEqual(geometry["shape"], shape)
            self.assertEqual(geometry["dtype"], declaration["dtype"])
            self.assertEqual(geometry["device"], "cpu")
            self.assertEqual(len(geometry["strides"]), len(shape))
        self.assertEqual(observed["geometry"]["layer.1.query.heads"]["strides"], [30720, 256, 1024, 1])
        expected_rms = {item["outputs"][0] for item in cone if item["opcode"] == "RMS_NORM"}
        self.assertEqual(set(observed["scalar_stages"]), expected_rms)
        positions = 0
        for name, stages in observed["scalar_stages"].items():
            self.assertEqual(set(stages), {"mean_bits", "denominator_bits", "rsqrt_bits", "mean_input_metadata"})
            rows = math.prod(capture.STATE_SPECS[name][0][:-1])
            self.assertEqual(rows, 120 if name == "layer.1.query.normalized" else 30)
            for stage in ("mean_bits", "denominator_bits", "rsqrt_bits"):
                self.assertEqual(len(stages[stage]), rows)
                self.assertTrue(all(type(bits) is int and 0 <= bits <= 0xFFFFFFFF for bits in stages[stage]))
                positions += len(stages[stage])
        self.assertEqual(positions, 810)
        softmax = torch.tensor(observed["softmax_f32_bits"], dtype=torch.int64)
        self.assertEqual(tuple(softmax.shape), (1, 4, 30, 30))
        self.assertEqual(softmax.numel(), 3600)
        self.assertTrue(bool(((softmax >= 0) & (softmax <= 0xFFFFFFFF)).all()))
        kernel_opcodes = {"RMS_NORM", "LINEAR", "GELU_TANH", "ADD", "MUL", "SCALE", "MATMUL_QK", "MATMUL_AV"}
        expected_kernels = {item["outputs"][0] for item in cone if item["opcode"] in kernel_opcodes} | {"softmax"}
        self.assertEqual(set(observed["kernels"]), expected_kernels)
        self.assertEqual(len(observed["kernels"]), 22)
        self.assertEqual(self.profile_count, 22)
        for names in observed["kernels"].values():
            self.assertEqual(names, ["synthetic_native_a", "synthetic_native_z"])
        self.assertEqual(json.loads(json.dumps(observed)), observed)

    def test_native_prefix_local_roots_order_and_no_mutation(self):
        before = self.snapshot()
        weights = {name: parameter.detach().clone() for name, parameter in self.model.named_parameters()}
        events, roots = [], {}

        def prefix_pre(module, args):
            events.append("layer0_pre")
            self.assertEqual(self.profile_count, 0)

        def prefix_post(module, args, output):
            events.append("layer0_post")
            self.assertEqual(self.profile_count, 0)
            roots["hidden.1"] = capture._tensor_record(output[0], *capture.STATE_SPECS["hidden.1"])

        def local(module, args, output):
            roots["rotary.local.cosine"] = capture._tensor_record(output[0], *capture.STATE_SPECS["rotary.local.cosine"])
            roots["rotary.local.sine"] = capture._tensor_record(output[1], *capture.STATE_SPECS["rotary.local.sine"])

        def target_pre(module, args, kwargs):
            events.append("layer1_pre")
            self.assertEqual(self.profile_count, 0)
            self.assertEqual(capture._tensor_record(args[0], *capture.STATE_SPECS["hidden.1"]), roots["hidden.1"])
            roots["mask.sliding"] = capture._tensor_record(kwargs["attention_mask"], *capture.STATE_SPECS["mask.sliding"])

        with ExitStack() as stack:
            for handle in (self.model.model.layers[0].register_forward_pre_hook(prefix_pre),
                           self.model.model.layers[0].register_forward_hook(prefix_post),
                           self.model.model.rotary_emb_local.register_forward_hook(local),
                           self.model.model.layers[1].register_forward_pre_hook(target_pre, with_kwargs=True)):
                stack.callback(handle.remove)
            observed = capture.capture_second_layer(self.model, self.ids, True)
        self.assertEqual(events, ["layer0_pre", "layer0_post", "layer1_pre"])
        for name, (bits, geometry) in roots.items():
            self.assertEqual(observed["state_bits"][name], bits)
            self.assertEqual(observed["geometry"][name], geometry)
        for name, parameter in self.model.named_parameters():
            self.assertTrue(torch.equal(parameter, weights[name]))
        self.assert_restored(before)

    def test_softmax_profile_leaf_only_and_layer_zero_passthrough(self):
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
            capture.capture_second_layer(self.model, self.ids, True)
        expected = []
        for depth in (0, 1):
            expected.extend([("wrapper_enter", 0), ("cast", "torch.bfloat16", "torch.float32", 0),
                             ("aten._softmax.default", depth), ("wrapper_exit", 0),
                             ("cast", "torch.float32", "torch.bfloat16", 0)])
        self.assertEqual(events, expected)
        self.assertEqual(self.profile_count, 22)
        self.assert_restored(before)

    def test_plain_minimal_repeated_calls_and_trace_equality(self):
        before = self.snapshot()
        traced = capture.capture_second_layer(self.model, self.ids, True)
        traced_again = capture.capture_second_layer(self.model, self.ids, True)
        self.assertEqual(traced["state_bits"], traced_again["state_bits"])
        self.assertEqual(traced["scalar_stages"], traced_again["scalar_stages"])
        self.assertEqual(traced["softmax_f32_bits"], traced_again["softmax_f32_bits"])
        profile_count = self.profile_count
        with patch.object(capture, "_profile_call", side_effect=AssertionError("Plain must not profile")):
            plain = capture.capture_second_layer(self.model, self.ids, False)
            again = capture.capture_second_layer(self.model, self.ids, False)
        self.assertEqual(self.profile_count, profile_count)
        self.assertEqual(set(plain["state_bits"]), {"hidden.1", "hidden.2"})
        self.assertEqual(set(plain["geometry"]), {"hidden.1", "hidden.2"})
        self.assertEqual(plain["state_bits"], again["state_bits"])
        for name in ("hidden.1", "hidden.2"):
            self.assertEqual(plain["state_bits"][name], traced["state_bits"][name])
            self.assertEqual(plain["geometry"][name], traced["geometry"][name])
        self.assertEqual(plain["kernels"], {})
        self.assertEqual(plain["scalar_stages"], {})
        self.assertEqual(plain["softmax_f32_bits"], [])
        self.assertTrue(plain["checks"]["plain_minimal_control"])
        self.assertTrue(all(plain["checks"].values()))
        self.assert_restored(before)

    def test_plain_installs_only_boundary_and_guard_hooks(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        before = self.snapshot()
        original = self.model.forward

        def checked(*args, **kwargs):
            self.assertEqual(self.snapshot()["functions"], before["functions"])
            for module, forward, hooks, prehooks, own_forward in before["modules"]:
                if module is not self.model:
                    self.assertEqual(module.forward, forward)
                boundaries = (self.model.model.layers[0], self.model.model.layers[1])
                guards = (self.model.model.layers[2], self.model.model.norm, self.model.lm_head)
                expected_outputs = len(hooks) + int(any(module is item for item in boundaries))
                expected_inputs = len(prehooks) + int(any(module is item for item in boundaries + guards))
                self.assertEqual(len(module._forward_hooks), expected_outputs)
                self.assertEqual(len(module._forward_pre_hooks), expected_inputs)
            return original(*args, **kwargs)

        with patch.object(self.model, "forward", new=checked), patch.object(TorchDispatchMode, "__enter__", side_effect=AssertionError("Plain dispatch mode")):
            capture.capture_second_layer(self.model, self.ids, False)
        self.assert_restored(before)

    def test_decoder_tuple_and_later_boundary_guards(self):
        layer = self.model.model.layers[1]
        original = layer.forward
        for traced in (False, True):
            for malformed in (lambda result: result[0], lambda result: result + (None,)):
                with patch.object(layer, "forward", new=lambda *args, **kwargs: malformed(original(*args, **kwargs))):
                    before = self.snapshot()
                    with self.assertRaisesRegex(ValueError, "one-element"):
                        capture.capture_second_layer(self.model, self.ids, traced)
                    self.assert_restored(before)
            value = torch.zeros((1, 30, 640), dtype=torch.bfloat16)
            for target in (self.model.model.layers[2], self.model.model.norm, self.model.lm_head):
                with patch.object(layer, "forward", new=lambda *args, **kwargs: target(value)):
                    before = self.snapshot()
                    with self.assertRaisesRegex(ValueError, "must not execute"):
                        capture.capture_second_layer(self.model, self.ids, traced)
                    self.assert_restored(before)

    def test_prefix_order_and_repeated_occurrence_guards(self):
        prefix, layer = self.model.model.layers[:2]
        original = prefix.forward

        def out_of_order(*args, **kwargs):
            return layer(*args, **kwargs)

        for traced in (False, True):
            with patch.object(prefix, "forward", new=out_of_order):
                before = self.snapshot()
                with self.assertRaisesRegex(ValueError, "layer zero must finish"):
                    capture.capture_second_layer(self.model, self.ids, traced)
                self.assert_restored(before)
            with patch.object(prefix, "forward", new=lambda *args, **kwargs: prefix(*args, **kwargs)):
                before = self.snapshot()
                with self.assertRaisesRegex(ValueError, "Repeated original operation: prefix_input"):
                    capture.capture_second_layer(self.model, self.ids, traced)
                self.assert_restored(before)
        leaf = layer.self_attn.q_proj
        with patch.object(leaf, "forward", new=lambda value: leaf(value)):
            before = self.snapshot()
            with self.assertRaisesRegex(ValueError, "Repeated original operation"):
                capture.capture_second_layer(self.model, self.ids, True)
            self.assert_restored(before)
        with patch.object(prefix, "forward", new=lambda *args, **kwargs: original(*args, **kwargs)[0]):
            with self.assertRaisesRegex(ValueError, "prefix decoder.*one-element"):
                capture.capture_second_layer(self.model, self.ids, False)

    def test_target_entry_kwargs_and_auxiliary_contracts(self):
        layer = self.model.model.layers[1]

        def keyword_hidden(module, args, kwargs):
            return (), {**kwargs, "hidden_states": args[0]}

        handle = layer.register_forward_pre_hook(keyword_hidden, with_kwargs=True)
        try:
            observed = capture.capture_second_layer(self.model, self.ids, True)
            self.assertEqual(len(observed["state_bits"]), 34)
        finally:
            handle.remove()
        for key, replacement, message in (
            ("position_ids", lambda value: value + 1, "position IDs"),
            ("cache_position", lambda value: value + 1, "cache positions"),
            ("attention_mask", lambda value: torch.zeros_like(value), "causal sliding mask"),
            ("position_embeddings_local", lambda value: (value[0],), "local rotary table pair"),
            ("use_cache", lambda value: True, "forbids cache"),
        ):
            def changed(module, args, kwargs):
                return args, {**kwargs, key: replacement(kwargs[key])}

            handle = layer.register_forward_pre_hook(changed, with_kwargs=True)
            try:
                before = self.snapshot()
                with self.assertRaisesRegex(ValueError, message):
                    capture.capture_second_layer(self.model, self.ids, False)
                self.assert_restored(before)
            finally:
                handle.remove()

    def test_target_local_table_mask_and_prefix_identity_links(self):
        layer = self.model.model.layers[1]
        for field in ("position_embeddings", "attention_mask"):
            def changed(module, args, kwargs):
                value = kwargs[field]
                replacement = tuple(item.clone() for item in value) if type(value) is tuple else value.clone()
                return args, {**kwargs, field: replacement}

            handle = layer.self_attn.register_forward_pre_hook(changed, with_kwargs=True)
            try:
                before = self.snapshot()
                with self.assertRaisesRegex(ValueError, "operand identity"):
                    capture.capture_second_layer(self.model, self.ids, True)
                self.assert_restored(before)
            finally:
                handle.remove()
        for traced in (False, True):
            handle = layer.register_forward_pre_hook(lambda module, args, kwargs: ((args[0].clone(),), kwargs), with_kwargs=True)
            try:
                before = self.snapshot()
                with self.assertRaisesRegex(ValueError, "layer zero output changed"):
                    capture.capture_second_layer(self.model, self.ids, traced)
                self.assert_restored(before)
            finally:
                handle.remove()

    def test_softmax_lineage_failure_cleanup_and_prefix_passthrough(self):
        original = torch.nn.functional.softmax
        for mutation, message in ((lambda value, kwargs: original(value.float().add_(1.0), **kwargs), "original masked BF16 upcast"),
                                  (lambda value, kwargs: original(value, **kwargs).clone(), "native FP32 output lineage")):
            calls = []

            def changed(input, **kwargs):
                calls.append(1)
                if len(calls) == 1:
                    return original(input, **kwargs)
                return mutation(input, kwargs)

            with patch.object(torch.nn.functional, "softmax", new=changed):
                before = self.snapshot()
                with self.assertRaisesRegex(ValueError, message):
                    capture.capture_second_layer(self.model, self.ids, True)
                self.assertEqual(len(calls), 2)
                self.assert_restored(before)
        calls = []

        def failing(*args, **kwargs):
            result = original(*args, **kwargs)
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("synthetic target softmax failure")
            return result

        with patch.object(torch.nn.functional, "softmax", new=failing):
            before = self.snapshot()
            with self.assertRaisesRegex(RuntimeError, "synthetic target softmax failure"):
                capture.capture_second_layer(self.model, self.ids, True)
            self.assert_restored(before)

    def test_regime_dtype_geometry_and_input_guards(self):
        before = self.snapshot()
        layer = self.model.model.layers[1]
        cases = ((self.model.config, "head_dim", 128, "head geometry"),
                 (self.model.config, "_attn_implementation_internal", "sdpa", "eager evaluation"),
                 (layer, "layer_idx", 0, "layer index"),
                 (layer.self_attn, "layer_idx", 0, "layer index"),
                 (layer.self_attn, "is_sliding", False, "attention configuration"),
                 (layer, "attention_type", "full_attention", "attention configuration"),
                 (self.model.config, "layer_types", ["sliding_attention", "full_attention", "full_attention"], "attention configuration"),
                 (layer.self_attn, "sliding_window", 1024, "attention configuration"),
                 (layer, "training", True, "eager evaluation"))
        for target, attribute, value, message in cases:
            with patch.object(target, attribute, value):
                with self.assertRaisesRegex(ValueError, message):
                    capture.capture_second_layer(self.model, self.ids, True)
        for module in (self.model.model.embed_tokens, self.model.model.layers[0].self_attn.q_proj, layer.self_attn.q_proj):
            with patch.object(module, "weight", torch.nn.Parameter(module.weight.float())):
                with self.assertRaisesRegex(ValueError, "BF16"):
                    capture.capture_second_layer(self.model, self.ids, False)
        for ids in ([[1] * 29], [[32] * 30], [[True] * 30], [[-1] * 30]):
            with self.assertRaises(ValueError):
                capture.capture_second_layer(self.model, ids, True)
        with self.assertRaisesRegex(ValueError, "boolean tracing flag"):
            capture.capture_second_layer(self.model, self.ids, 1)
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            capture._tensor_record(torch.zeros((1, 30, 640)), (1, 30, 640), "torch.bfloat16")
        with self.assertRaisesRegex(ValueError, "BF16, FP32 or int64"):
            capture._bit_tensor(torch.zeros(1, dtype=torch.float64))
        self.assert_restored(before)

    def test_runtime_source_and_token_stability(self):
        before = self.snapshot()
        for attribute, values in (("_runtime", [{"device": "before"}, {"device": "after"}]),
                                  ("_code_sha", ["before", "after"])):
            with patch.object(capture, attribute, side_effect=values):
                with self.assertRaisesRegex(ValueError, "integrity check"):
                    capture.capture_second_layer(self.model, self.ids, False)
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
                capture.capture_second_layer(self.model, self.ids, False)
            self.assert_restored(patched)
        ids = copy.deepcopy(self.ids)

        def changed_list(*args, **kwargs):
            ids[0][0] = 7
            return original(*args, **kwargs)

        with patch.object(self.model, "forward", new=changed_list):
            with self.assertRaisesRegex(ValueError, "integrity check"):
                capture.capture_second_layer(self.model, ids, False)
        self.assert_restored(before)

    def test_code_hash_own_source_rms_dependencies_and_missing_symbols(self):
        original_hash = capture._code_sha()
        self.assertEqual(len(original_hash), 64)
        original = capture.reference_gemma._observe_rms_module

        def observer(module, value):
            return original(module, value)

        with patch.object(capture.reference_gemma, "_observe_rms_module", new=observer):
            self.assertNotEqual(capture._code_sha(), original_hash)
        read_bytes = Path.read_bytes
        own_path = Path(capture.__file__)

        def source_bytes(path):
            content = read_bytes(path)
            return content + b"\n" if path == own_path else content

        with patch.object(Path, "read_bytes", new=source_bytes):
            self.assertNotEqual(capture._code_sha(), original_hash)
        older = own_path.with_name("gemma_first_layer_capture.py")
        frozen = hashlib.sha256(older.read_bytes()).hexdigest()
        capture.capture_second_layer(self.model, self.ids, False)
        self.assertEqual(hashlib.sha256(older.read_bytes()).hexdigest(), frozen)
        with patch.object(torch.profiler, "profile") as profiler:
            profiler.return_value.__enter__.return_value.events.return_value = []
            with self.assertRaisesRegex(ValueError, "kernel symbols"):
                capture._profile_call(lambda: None, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
