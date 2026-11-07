"""Exp. 14 tests: AdamNVSubset (controller on a subset of tensors).

  * tensors in the plain group follow torch Adam (float64, 300 steps, tolerance 1e-6);
  * tensors in the adaptive group are bit-identical to a stand-alone AdamNV on those tensors;
  * the two subsets are complementary (rank <= 1 versus rank >= 2);
  * gamma = 0 makes every subset equal to AdamNV with gamma = 0;
  * invalid arguments are rejected.
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


def _load(name):
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    spec = importlib.util.spec_from_file_location(name + "_exp14_under_test", SRC / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tensors(dtype=torch.float64):
    gen = torch.Generator().manual_seed(2)
    return [torch.randn(8, generator=gen).to(dtype),          # 1-D
            torch.randn(4, 6, generator=gen).to(dtype),        # 2-D
            torch.randn(3, 2, 2, 2, generator=gen).to(dtype),  # 4-D
            torch.randn(5, generator=gen).to(dtype)]           # 1-D


def _grads(steps, tensors, seed=0):
    gen = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(steps):
        step = []
        for t in tensors:
            scale = torch.ones(t.shape)
            scale.view(-1)[::2] = 4.0
            step.append((torch.randn(t.shape, generator=gen) * scale).to(t.dtype))
        out.append(step)
    return out


def _run(make_opt, tensors, grads):
    params = [torch.nn.Parameter(t.clone()) for t in tensors]
    opt = make_opt(params)
    for step in grads:
        for p, g in zip(params, step):
            p.grad = g.clone()
        opt.step()
    return [p.detach().clone() for p in params]


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class AdamNVSubsetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.AdamNV = _load("adam_nv").AdamNV
        cls.Subset = _load("adam_nv_subset").AdamNVSubset

    def _check_subset(self, subset, adaptive_idx, plain_idx):
        tensors = _tensors()
        grads = _grads(300, tensors)
        mixed = _run(lambda ps: self.Subset(ps, lr=1e-3, subset=subset), tensors, grads)
        # plain tensors == torch Adam on those tensors
        t_plain = [tensors[i] for i in plain_idx]
        g_plain = [[step[i] for i in plain_idx] for step in grads]
        ref = _run(lambda ps: torch.optim.Adam(ps, lr=1e-3, betas=(0.9, 0.999), eps=1e-8), t_plain, g_plain)
        for k, i in enumerate(plain_idx):
            self.assertTrue(torch.allclose(mixed[i], ref[k], atol=1e-6, rtol=1e-6), f"plain tensor {i}")
        # adaptive tensors == AdamNV on those tensors alone
        t_ad = [tensors[i] for i in adaptive_idx]
        g_ad = [[step[i] for i in adaptive_idx] for step in grads]
        ref_ad = _run(lambda ps: self.AdamNV(ps, lr=1e-3), t_ad, g_ad)
        for k, i in enumerate(adaptive_idx):
            self.assertTrue(torch.equal(mixed[i], ref_ad[k]), f"adaptive tensor {i}")
        # and the adaptive tensors really differ from plain Adam (the controller is active)
        ref_plain_on_ad = _run(lambda ps: torch.optim.Adam(ps, lr=1e-3, betas=(0.9, 0.999), eps=1e-8), t_ad, g_ad)
        self.assertTrue(any(not torch.allclose(mixed[i], ref_plain_on_ad[k], atol=1e-6)
                            for k, i in enumerate(adaptive_idx)))

    def test_onedim_subset(self):
        self._check_subset("onedim", adaptive_idx=[0, 3], plain_idx=[1, 2])

    def test_rest_subset(self):
        self._check_subset("rest", adaptive_idx=[1, 2], plain_idx=[0, 3])

    def test_subsets_are_complementary(self):
        tensors = _tensors()
        ps = [torch.nn.Parameter(t.clone()) for t in tensors]
        a = self.Subset(ps, subset="onedim")
        b = self.Subset([torch.nn.Parameter(t.clone()) for t in tensors], subset="rest")
        self.assertEqual((a.n_adaptive_tensors, a.n_plain_tensors), (2, 2))
        self.assertEqual((b.n_adaptive_tensors, b.n_plain_tensors), (2, 2))

    def test_gamma_zero_makes_every_subset_adam_nv_with_gamma_zero(self):
        tensors = _tensors()
        grads = _grads(100, tensors)
        base = _run(lambda ps: self.AdamNV(ps, lr=1e-3, gamma=0.0), tensors, grads)
        for subset in ("onedim", "rest"):
            sub = _run(lambda ps: self.Subset(ps, lr=1e-3, subset=subset, gamma=0.0), tensors, grads)
            for x, y in zip(base, sub):
                self.assertTrue(torch.equal(x, y), subset)

    def test_invalid_arguments_rejected(self):
        p = torch.nn.Parameter(torch.zeros(3, 3))
        with self.assertRaises(ValueError):
            self.Subset([p], subset="bogus")
        with self.assertRaises(ValueError):
            self.Subset([{"params": [p]}], subset="onedim")


if __name__ == "__main__":
    unittest.main()
