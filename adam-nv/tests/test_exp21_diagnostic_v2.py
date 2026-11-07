"""Exp. 21 tests: noise_proxy_diagnostic_v2.

  * estimators: unclamped noise fraction, group structure (K/4, K/4, K/2), split halves, stream accumulator;
  * partial Spearman and Spearman-Brown;
  * s0 (lambda = 0) formula;
  * collect_diagnostics_v2 leaves model weights, BatchNorm buffers and optimizer state unchanged;
  * with legacy_batches the first-K/4 statistics equal Exp. 16's function on the same batches;
  * pipeline controls: a proxy that tracks the true noise fraction gives a high reliability-corrected correlation, a random one gives about zero.
"""
import importlib.util
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

try:
    import numpy as np
    import torch
    import torch.nn as nn
    from scipy import stats
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "e2_noisy_gradient" / "src"


def _load(name):
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    spec = importlib.util.spec_from_file_location(name + "_exp21_under_test", SRC / (name + ".py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1); self.bn = nn.BatchNorm2d(8); self.fc = nn.Linear(8, 4)

    def forward(self, x):
        x = torch.relu(self.bn(self.conv(x))); return self.fc(x.mean((2, 3)))


def _setup(seed=0, n=512):
    torch.manual_seed(seed)
    model = TinyNet(); AdamNV = _load("adam_nv").AdamNV
    opt = AdamNV(model.parameters(), lr=1e-3)
    ds = torch.utils.data.TensorDataset(torch.randn(n, 3, 8, 8), torch.randint(0, 4, (n,)))
    loss_fn = nn.CrossEntropyLoss(); g = torch.Generator().manual_seed(1)
    for xb, yb in torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, generator=g):
        opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
    return model, opt, ds


@unittest.skipIf(torch is None, "PyTorch/scipy not installed")
class DiagV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v1 = _load("noise_proxy_diagnostic"); cls.v2 = _load("noise_proxy_diagnostic_v2")

    def test_unclamped_fraction_matches_numpy_and_exceeds_one_for_noise(self):
        g = torch.randn(64, 20000, dtype=torch.float64); sg, sg2 = g.sum(0), (g * g).sum(0)
        qu, ms = self.v2.unclamped_noise_fraction(sg, sg2, 64)
        ref = g.var(0, unbiased=True) / (g * g).mean(0)
        self.assertLess((qu - ref).abs().max().item(), 1e-9)
        self.assertGreater(float((qu > 1).double().mean()), 0.3)       # pure noise: about a third of the elements exceed 1 unclamped
        q, _ = self.v1.direct_noise_fraction(sg, sg2, 64); self.assertLessEqual(float(q.max()), 1.0)

    def test_groups_and_halves(self):
        p = [torch.zeros(5)]; acc = self.v2.Accumulators(p, 16); gs = []
        for b in range(16):
            p[0].grad = torch.randn(5); gs.append(p[0].grad.clone()); acc.add(b, p)
        self.assertEqual(acc.n, [4, 4, 8])
        g1, _, n1 = acc.combined(0, [0]); gA, _, nA = acc.combined(0, [0, 1]); gB, _, nB = acc.combined(0, [2]); gT, _, nT = acc.combined(0, [0, 1, 2])
        self.assertLess((g1 - torch.stack(gs[:4]).sum(0)).abs().max().item(), 1e-6); self.assertEqual((n1, nA, nB, nT), (4, 8, 8, 16))
        self.assertLess((gA - torch.stack(gs[:8]).sum(0)).abs().max().item(), 1e-6); self.assertLess((gB - torch.stack(gs[8:]).sum(0)).abs().max().item(), 1e-6)

    def test_stream_accumulator(self):
        p = [torch.zeros(4), torch.zeros(2, 3)]; sa = self.v2.StreamAccumulator(p); tot = [torch.zeros_like(x) for x in p]
        for _ in range(5):
            for x in p: x.grad = torch.randn_like(x)
            sa.add(p)
            for i, x in enumerate(p): tot[i] += x.grad
        self.assertEqual(sa.n, 5)
        for i in range(2): self.assertLess((sa.sg[i] - tot[i]).abs().max().item(), 1e-6)

    def test_partial_spearman_and_brown(self):
        rng = np.random.default_rng(0); c = rng.standard_normal(5000); a = c + 0.5 * rng.standard_normal(5000); b = c + 0.5 * rng.standard_normal(5000)
        ra, rb, rab = stats.spearmanr(a, c)[0], stats.spearmanr(b, c)[0], stats.spearmanr(a, b)[0]
        ref = (rab - ra * rb) / np.sqrt((1 - ra ** 2) * (1 - rb ** 2))
        self.assertAlmostEqual(self.v2.partial_spearman(torch.tensor(a), torch.tensor(b), torch.tensor(c)), ref, places=9)
        self.assertLess(abs(ref), 0.05)                                   # given the common cause, a and b are unrelated
        self.assertAlmostEqual(self.v2.spearman_brown(0.5), 2 * 0.5 / 1.5, places=12)

    def test_s0_is_the_scale_free_score(self):
        m, v = torch.randn(1000) * 1e-4, torch.rand(1000) * 1e-7 + 1e-9
        s0 = self.v1.proxy_score(m, v, 0.0, 1e-30); ref = torch.clamp(v - m * m, min=0) / (v + 1e-30)
        self.assertLess((s0 - ref).abs().max().item(), 1e-6)
        k = 1000.0; self.assertLess((self.v1.proxy_score(m * k, v * k * k, 0.0, 1e-30) - s0).abs().max().item(), 1e-4)   # scale-invariant
        s = self.v1.proxy_score(m * k, v * k * k, 0.1, 1e-8); self.assertGreater((s - s0).abs().max().item(), 0.01)       # lambda=0.1 version is not

    def test_diagnostics_do_not_change_model_or_optimizer(self):
        model, opt, ds = _setup(); tmp = tempfile.mkdtemp()
        sd = {k: v.clone() for k, v in model.state_dict().items()}; st = {id(p): {k: (v.clone() if torch.is_tensor(v) else v) for k, v in opt.state[p].items()} for g in opt.param_groups for p in g["params"]}
        args = SimpleNamespace(batch_size=16, k_batches=16, sample=100000, seed=3, legacy_batches=False, log_dir=tmp, label_noise=0.0)
        stream = self.v2.StreamAccumulator([p for g in opt.param_groups for p in g["params"]])
        rec = self.v2.collect_diagnostics_v2(model, opt, ds, args, torch.device("cpu"), epoch=5, stream=stream)
        for k, v in model.state_dict().items(): self.assertTrue(torch.equal(v, sd[k]), k)
        for g in opt.param_groups:
            for p in g["params"]:
                for k, v in st[id(p)].items():
                    if torch.is_tensor(v): self.assertTrue(torch.equal(opt.state[p][k], v), k)
                self.assertIsNone(p.grad)
        self.assertEqual(len(rec["tensors"]), len(list(model.parameters()))); self.assertTrue((Path(tmp) / "proxy2_sample_ln0.0_seed3_ep5.npz").is_file())
        z = np.load(Path(tmp) / "proxy2_sample_ln0.0_seed3_ep5.npz"); self.assertEqual(set(z.files), {"s", "s0", "q16", "qu", "rms", "qs", "tid", "rank"})
        for t in rec["tensors"]:
            if t["reliability"] == t["reliability"]: self.assertLessEqual(t["reliability"], 1.0 + 1e-9)

    def test_legacy_first_group_equals_experiment_16(self):
        model, opt, ds = _setup(); tmp = tempfile.mkdtemp(); dev = torch.device("cpu")
        a1 = SimpleNamespace(batch_size=16, k_batches=4, sample=100000, seed=0)
        r1 = self.v1.collect_diagnostics(model, opt, ds, a1, dev, epoch=7)
        a2 = SimpleNamespace(batch_size=16, k_batches=16, sample=100000, seed=0, legacy_batches=True, log_dir=tmp, label_noise=0.0)
        r2 = self.v2.collect_diagnostics_v2(model, opt, ds, a2, dev, epoch=7)
        self.assertEqual(r2["k_legacy"], 4)
        self.assertAlmostEqual(r2["spearman_s_q16"], r1["spearman_s_q"], places=9); self.assertAlmostEqual(r2["mean_q16"], r1["mean_q"], places=9)
        self.assertAlmostEqual(r2["frac_s_zero"], r1["frac_s_zero"], places=9)

    def _population(self, seed, tracking, n=20000, K=64):
        rng = np.random.default_rng(seed); f = rng.beta(1, 9, n); qtrue = 1 - f
        g = np.sqrt(f) + np.sqrt(1 - f) * rng.standard_normal((K, n))
        def qu(x): return x.var(0, ddof=1) / (x * x).mean(0)
        qA, qB, qT = qu(g[:K // 2]), qu(g[K // 2:]), qu(g)
        s = (-qtrue + 0.01 * rng.standard_normal(n)) if tracking else rng.standard_normal(n)     # monotone in true q (negated: s is a noise-LIKE score; rank sign irrelevant below)
        r = abs(self.v2._sp(s, qT)); rel = self.v2.spearman_brown(self.v2._sp(qA, qB))
        return r, rel, r / np.sqrt(rel)

    def test_pipeline_controls(self):
        r, rel, rc = self._population(0, True); self.assertGreater(rel, 0.3); self.assertGreater(rc, 0.7); self.assertLess(rc, 1.15)    # positive control
        r, rel, rc = self._population(1, False); self.assertLess(abs(rc), 0.15)                                                         # negative control


if __name__ == "__main__":
    unittest.main()
