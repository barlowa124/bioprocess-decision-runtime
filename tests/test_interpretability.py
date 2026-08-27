from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bioprocess_runtime.interpretability import (
    ActivationCapture,
    ProjectionInputCapture,
    _effect_size,
    _normalized,
    _null_controls,
    default_low_oxygen_prompts,
    hash_model_snapshot,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "Gemma optional dependencies are not installed")
class ActivationCaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        import torch

        class ToyAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.o_proj = torch.nn.Identity()

            def forward(self, hidden):
                return (self.o_proj(hidden + 0.25), None)

        class ToyMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.down_proj = torch.nn.Identity()

            def forward(self, hidden):
                return self.down_proj(hidden + 0.5)

        class ToyLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = ToyAttention()
                self.mlp = ToyMLP()

            def forward(self, hidden):
                return (self.mlp(self.self_attn(hidden)[0]) + 0.25,)

        class ToyBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([ToyLayer(), ToyLayer()])

        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.model = ToyBackbone()

            def forward(self, hidden):
                output = hidden
                for layer in self.model.layers:
                    output = layer(output)[0]
                return output

        self.torch = torch
        self.model = ToyModel()
        self.capture = ActivationCapture(self.model)

    def test_captures_selected_residual_outputs(self) -> None:
        hidden = self.torch.zeros((1, 3, 4))
        with self.capture.capture((0, 1)) as activations:
            output = self.model(hidden)
        self.assertTrue(self.torch.equal(output, self.torch.full_like(output, 2.0)))
        self.assertTrue(self.torch.equal(activations[0], self.torch.ones(4)))
        self.assertTrue(self.torch.equal(activations[1], self.torch.full((4,), 2.0)))

    def test_captures_attention_and_mlp_outputs_separately(self) -> None:
        hidden = self.torch.zeros((1, 3, 4))
        attention_capture = ActivationCapture(self.model, "attention")
        mlp_capture = ActivationCapture(self.model, "mlp")
        with attention_capture.capture((0,)) as attention:
            self.model(hidden)
        with mlp_capture.capture((0,)) as mlp:
            self.model(hidden)
        self.assertTrue(self.torch.equal(attention[0], self.torch.full((4,), 0.25)))
        self.assertTrue(self.torch.equal(mlp[0], self.torch.full((4,), 0.75)))

    def test_projection_input_capture_and_replacement(self) -> None:
        hidden = self.torch.zeros((1, 3, 4))
        capture = ProjectionInputCapture(self.model, "attention_heads")
        with capture.capture(0) as activation:
            self.model(hidden)
        self.assertTrue(self.torch.equal(activation["activation"], self.torch.full((4,), 0.25)))
        replacement = self.torch.tensor([2.0, 3.0])
        baseline = self.model(hidden)
        with capture.replace(0, 0, 2, replacement):
            replaced = self.model(hidden)
        self.assertTrue(self.torch.equal((replaced - baseline)[0, -1], self.torch.tensor([1.75, 2.75, 0.0, 0.0])))
        indices = np.arange(4)[::-1][:2].copy()
        with capture.replace_indices(0, indices, self.torch.tensor([4.0, 5.0])):
            indexed = self.model(hidden)
        self.assertTrue(self.torch.equal((indexed - baseline)[0, -1], self.torch.tensor([0.0, 0.0, 4.75, 3.75])))

    def test_intervention_changes_only_last_token(self) -> None:
        hidden = self.torch.zeros((1, 3, 4))
        direction = self.torch.tensor([1.0, 0.0, 0.0, 0.0])
        baseline = self.model(hidden)
        with self.capture.intervene(0, direction, 2.0):
            intervened = self.model(hidden)
        difference = intervened - baseline
        self.assertEqual(float(difference[0, -1, 0]), 2.0)
        self.assertEqual(float(difference[0, 0, 0]), 0.0)


class InterpretabilityMathTests(unittest.TestCase):
    def test_normalizes_direction(self) -> None:
        result = _normalized(np.array([3.0, 4.0]))
        self.assertAlmostEqual(float(np.linalg.norm(result)), 1.0)

    def test_effect_size_preserves_direction(self) -> None:
        positive = np.array([3.0, 4.0])
        negative = np.array([0.0, 1.0])
        self.assertGreater(_effect_size(positive, negative), 0)

    def test_prompt_splits_are_disjoint(self) -> None:
        prompts = default_low_oxygen_prompts()
        groups = [
            set(prompts.positive_training + prompts.negative_training),
            set(prompts.positive_validation + prompts.negative_validation),
            set(prompts.positive_test + prompts.negative_test),
        ]
        self.assertFalse(groups[0] & groups[1])
        self.assertFalse(groups[0] & groups[2])
        self.assertFalse(groups[1] & groups[2])
        self.assertEqual(len(prompts.positive_training), 6)
        self.assertEqual(len(prompts.positive_validation), 4)
        self.assertEqual(len(prompts.positive_test), 12)
        self.assertEqual(len(prompts.negative_test), 12)

    def test_null_controls_are_seeded_and_report_empirical_p_values(self) -> None:
        positive_training = np.array([[2.0, 1.0], [3.0, 1.0], [4.0, 1.0]])
        negative_training = np.array([[-2.0, 1.0], [-3.0, 1.0], [-4.0, 1.0]])
        positive_test = np.array([[2.5, 1.0], [3.5, 1.0]])
        negative_test = np.array([[-2.5, 1.0], [-3.5, 1.0]])
        layers = {0: positive_training}
        negative_layers = {0: negative_training}
        positive_test_layers = {0: positive_test}
        negative_test_layers = {0: negative_test}
        first = _null_controls(
            layers, negative_layers, positive_test_layers, negative_test_layers, positive_test_layers, negative_test_layers, 1.0, iterations=20, seed=9
        )
        second = _null_controls(
            layers, negative_layers, positive_test_layers, negative_test_layers, positive_test_layers, negative_test_layers, 1.0, iterations=20, seed=9
        )
        self.assertEqual(first, second)
        self.assertGreater(first["random_directions"]["empirical_right_tail_p_value"], 0)

    def test_snapshot_hash_changes_with_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            file = path / "weights.bin"
            file.write_bytes(b"first")
            first = hash_model_snapshot(path)
            file.write_bytes(b"second")
            second = hash_model_snapshot(path)
            self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
