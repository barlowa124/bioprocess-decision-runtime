from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bioprocess_runtime.interpretability import ActivationCapture, _effect_size, _normalized, hash_model_snapshot


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "Gemma optional dependencies are not installed")
class ActivationCaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        import torch

        class ToyLayer(torch.nn.Module):
            def forward(self, hidden):
                return (hidden + 1.0,)

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
