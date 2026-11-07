"""Exp. 13 tests: AdamNVTensorMean (tensor-mean beta_1 control).

Checks (CPU, small tensors):
  * mode="none" is bit-identical to AdamNV;
  * with gamma = 0 (uniform beta_1) tensor_mean matches AdamNV to float64 rounding;
  * in tensor_mean mode the applied beta_1 is constant within every tensor and equals the mean of
    the controller's own per-element beta_1;
  * tensor_mean differs from AdamNV when the controller is active (heteroscedastic gradients);
  * the per-tensor log lists every tensor with its shape and mean/min/max at the right cadence;
  * invalid modes are rejected.
"""

import importlib.util
import json
import sys
import tempfile
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
    spec = importlib.util.spec_from_file_location(name + "_exp13_under_test", SRC / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_params(dtype=torch.float32):
    gen = torch.Generator().manual_seed(1)
    return [torch.nn.Parameter(torch.randn(6, 8, generator=gen).to(dtype)),
            torch.nn.Parameter(torch.randn(10, generator=gen).to(dtype)),
            torch.nn.Parameter(torch.randn(3, 3, 2, 2, generator=gen).to(dtype))]


def _grads(steps, params, seed=0):
    gen = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(steps):
        step = []
        for p in params:
            scale = torch.ones(p.shape)
            scale.view(-1)[::2] = 4.0
            step.append((torch.randn(p.shape, generator=gen) * scale).to(p.dtype))
        out.append(step)
    return out


def _run(cls, steps=60, dtype=torch.float32, **kw):
    params = _make_params(dtype)
    opt = cls(params, lr=1e-3, **kw)
    for step in _grads(steps, params):
        for p, g in zip(params, step):
            p.grad = g.clone()
        opt.step()
    return [p.detach().clone() for p in params], opt


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class AdamNVTensorMeanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.AdamNV = _load("adam_nv").AdamNV
        cls.TM = _load("adam_nv_tensormean").AdamNVTensorMean

    def test_none_is_bit_identical_to_adam_nv(self):
        a, _ = _run(self.AdamNV)
        b, _ = _run(self.TM, mode="none")
        for x, y in zip(a, b):
            self.assertTrue(torch.equal(x, y))

    def test_gamma_zero_matches_adam_nv_to_rounding(self):
        a, _ = _run(self.AdamNV, dtype=torch.float64, gamma=0.0)
        b, _ = _run(self.TM, dtype=torch.float64, mode="tensor_mean", gamma=0.0)
        for x, y in zip(a, b):
            self.assertTrue(torch.allclose(x, y, atol=1e-12, rtol=1e-12))

    def test_applied_beta1_is_the_tensor_mean(self):
        records = []

        class Recording(self.TM):
            def _apply_shuffle(self, p, state, beta_1):
                out = super()._apply_shuffle(p, state, beta_1)
                records.append((beta_1.detach().clone(), out.detach().clone()))
                return out

        _run(Recording, mode="tensor_mean")
        spread_before = 0.0
        for before, after in records:
            self.assertTrue(torch.allclose(after, before.mean().expand_as(before), atol=1e-7))
            self.assertLess(float(after.max() - after.min()), 1e-7)
            spread_before = max(spread_before, float(before.max() - before.min()))
        self.assertGreater(spread_before, 1e-3, "the controller never produced within-tensor spread")

    def test_tensor_mean_differs_when_the_controller_is_active(self):
        base, _ = _run(self.AdamNV)
        tm, _ = _run(self.TM, mode="tensor_mean")
        self.assertTrue(any(not torch.equal(x, y) for x, y in zip(base, tm)))

    def test_tensor_log_lists_every_tensor_at_the_cadence(self):
        with tempfile.TemporaryDirectory() as d:
            tpath, dpath = Path(d) / "tensors.jsonl", Path(d) / "dist.jsonl"
            _run(self.TM, mode="none", steps=12, tensor_log_path=str(tpath), dist_log_path=str(dpath),
                 dist_log_every=4)
            lines = [json.loads(x) for x in tpath.read_text().splitlines()]
            self.assertEqual([r["step"] for r in lines], [4, 8, 12])
            shapes = [[6, 8], [10], [3, 3, 2, 2]]
            for r in lines:
                self.assertEqual([t["shape"] for t in r["tensors"]], shapes)
                for t in r["tensors"]:
                    self.assertLessEqual(t["min"], t["mean"] + 1e-9)
                    self.assertLessEqual(t["mean"], t["max"] + 1e-9)

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            self.TM([torch.nn.Parameter(torch.zeros(3, 3))], mode="bogus")


if __name__ == "__main__":
    unittest.main()
