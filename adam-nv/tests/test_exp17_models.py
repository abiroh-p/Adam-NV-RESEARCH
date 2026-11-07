"""Exp. 17 tests: alternative architectures (CPU).

  * output shapes for 32x32 inputs, for 10 and 100 classes;
  * parameter counts in the documented ranges;
  * a fixed seed gives identical weights and outputs;
  * every parameter receives a gradient in one backward pass;
  * unknown architectures are rejected.
"""

import importlib.util
import sys
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "e2_noisy_gradient" / "src"


def _load():
    spec = importlib.util.spec_from_file_location("models_alt_exp17_under_test", SRC / "models_alt.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class ModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load()

    def test_shapes(self):
        x = torch.randn(4, 3, 32, 32)
        for arch in self.m.ARCHS:
            for nc in (10, 100):
                self.assertEqual(tuple(self.m.build(arch, nc).eval()(x).shape), (4, nc), (arch, nc))

    def test_parameter_counts(self):
        vit = sum(p.numel() for p in self.m.build("vit_tiny").parameters())
        vgg = sum(p.numel() for p in self.m.build("vgg11_bn").parameters())
        self.assertTrue(1.5e6 < vit < 2.2e6, vit)
        self.assertTrue(9.0e6 < vgg < 9.5e6, vgg)

    def test_seed_reproducibility(self):
        x = torch.randn(2, 3, 32, 32)
        for arch in self.m.ARCHS:
            torch.manual_seed(5); a = self.m.build(arch).eval()
            torch.manual_seed(5); b = self.m.build(arch).eval()
            self.assertTrue(all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters())), arch)
            self.assertTrue(torch.equal(a(x), b(x)), arch)

    def test_every_parameter_gets_a_gradient(self):
        x, y = torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))
        for arch in self.m.ARCHS:
            model = self.m.build(arch).train()
            torch.nn.functional.cross_entropy(model(x), y).backward()
            missing = [n for n, p in model.named_parameters() if p.grad is None]
            self.assertEqual(missing, [], arch)

    def test_unknown_arch_rejected(self):
        with self.assertRaises(ValueError):
            self.m.build("resnet50")


if __name__ == "__main__":
    unittest.main()
