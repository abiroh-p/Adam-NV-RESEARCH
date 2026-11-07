"""Exp. 20 tests: AdamConstSubset (constant beta_1 on a subset of tensors, plain Adam elsewhere).

  * selected tensors follow torch Adam with betas (c, 0.999) (float64, 300 steps, tol 1e-6);
  * the other tensors follow torch Adam with betas (0.9, 0.999);
  * the two subsets are complementary (rank <= 1 versus rank >= 2) and the parameter groups differ
    only in beta_base, beta_min (gamma is 0 in both);
  * beta_1 is exactly the constant: get_beta1_stats() min/max equal c and 0.9;
  * c = 0.9 makes every tensor follow plain torch Adam;
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
    spec = importlib.util.spec_from_file_location(name + "_exp20_under_test", SRC / (name + ".py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def _tensors(dtype=torch.float64):
    g = torch.Generator().manual_seed(2)
    return [torch.randn(8, generator=g).to(dtype), torch.randn(4, 6, generator=g).to(dtype),
            torch.randn(3, 2, 2, 2, generator=g).to(dtype), torch.randn(5, generator=g).to(dtype)]


def _grads(steps, tensors, seed=0):
    g = torch.Generator().manual_seed(seed); out = []
    for _ in range(steps):
        step = []
        for t in tensors:
            sc = torch.ones(t.shape); sc.view(-1)[::2] = 4.0
            step.append((torch.randn(t.shape, generator=g) * sc).to(t.dtype))
        out.append(step)
    return out


def _run(make, tensors, grads):
    ps = [torch.nn.Parameter(t.clone()) for t in tensors]; o = make(ps)
    for st in grads:
        for p, gg in zip(ps, st): p.grad = gg.clone()
        o.step()
    return [p.detach() for p in ps], o


@unittest.skipIf(torch is None, "PyTorch not installed")
class ConstSubsetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load("adam_const_subset")
        cls.T = _tensors(); cls.G = _grads(300, cls.T)

    def _check(self, subset, c):
        res, o = _run(lambda ps: self.mod.AdamConstSubset(ps, lr=1e-3, subset=subset, const_beta1=c), self.T, self.G)
        sel = [i for i, t in enumerate(self.T) if (t.dim() <= 1) == (subset == "onedim")]
        oth = [i for i in range(len(self.T)) if i not in sel]
        for idx, b1 in ((sel, c), (oth, 0.9)):
            ref, _ = _run(lambda ps: torch.optim.Adam(ps, lr=1e-3, betas=(b1, 0.999), eps=1e-8),
                          [self.T[i] for i in idx], [[st[i] for i in idx] for st in self.G])
            for k, i in enumerate(idx):
                self.assertLess((res[i] - ref[k]).abs().max().item(), 1e-6, f"{subset} c={c} tensor {i} beta1={b1}")
        return o

    def test_selected_and_plain_groups_follow_torch_adam(self):
        for subset in ("onedim", "rest"):
            for c in (0.7, 0.79, 0.95):
                self._check(subset, c)

    def test_groups_are_complementary_and_differ_only_in_beta(self):
        for subset in ("onedim", "rest"):
            o = self.mod.AdamConstSubset([t.clone().requires_grad_() for t in _tensors(torch.float32)], lr=1e-3, subset=subset, const_beta1=0.79)
            g0, g1 = o.param_groups
            self.assertEqual({k for k in g0 if k != "params" and g0[k] != g1[k]}, {"beta_base", "beta_min"})
            self.assertEqual((g0["gamma"], g1["gamma"]), (0.0, 0.0))
            d0 = {p.dim() <= 1 for p in g0["params"]}; d1 = {p.dim() <= 1 for p in g1["params"]}
            self.assertEqual(d0, {subset == "onedim"}); self.assertEqual(d1, {subset != "onedim"})

    def test_beta1_is_exactly_the_constant(self):
        o = self._check("onedim", 0.79)
        mean, mn, mx = o.get_beta1_stats()
        self.assertAlmostEqual(mn, 0.79, places=6); self.assertAlmostEqual(mx, 0.9, places=6)

    def test_constant_equal_to_plain_gives_plain_adam_everywhere(self):
        res, _ = _run(lambda ps: self.mod.AdamConstSubset(ps, lr=1e-3, subset="onedim", const_beta1=0.9), self.T, self.G)
        ref, _ = _run(lambda ps: torch.optim.Adam(ps, lr=1e-3, betas=(0.9, 0.999), eps=1e-8), self.T, self.G)
        for a, b in zip(res, ref): self.assertLess((a - b).abs().max().item(), 1e-6)

    def test_invalid_arguments_rejected(self):
        ps = [torch.nn.Parameter(t.clone()) for t in self.T]
        for kw in (dict(subset="bad", const_beta1=0.8), dict(subset="onedim", const_beta1=None),
                   dict(subset="onedim", const_beta1=0.0), dict(subset="onedim", const_beta1=1.0)):
            with self.assertRaises(ValueError): self.mod.AdamConstSubset(ps, lr=1e-3, **kw)
        with self.assertRaises(ValueError):
            self.mod.AdamConstSubset([{"params": ps}], lr=1e-3, subset="onedim", const_beta1=0.8)


if __name__ == "__main__":
    unittest.main()
