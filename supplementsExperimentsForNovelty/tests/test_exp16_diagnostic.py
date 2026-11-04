"""Exp. 16 tests: noise_proxy_diagnostic helpers (CPU, tiny model and data).

  * spearman agrees with scipy on data with ties;
  * the direct noise fraction q recovers the known variance fraction of synthetic gradients
    (g_k = mu + noise: q is close to var / (var + mu^2));
  * q is 0 for identical gradients and clipped to at most 1;
  * proxy_score equals the formula used by AdamNV;
  * collect_diagnostics leaves the model weights, the BatchNorm running statistics and the optimizer
    state unchanged and returns correlations in [-1, 1].
"""

import importlib.util
import sys
from pathlib import Path
import types
import unittest

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "e2_noisy_gradient" / "src"


def _load(name):
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    spec = importlib.util.spec_from_file_location(name + "_exp16_under_test", SRC / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class DiagnosticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.diag = _load("noise_proxy_diagnostic")
        cls.AdamNV = _load("adam_nv").AdamNV

    def test_spearman_matches_scipy_with_ties(self):
        from scipy import stats
        gen = torch.Generator().manual_seed(0)
        a = torch.randint(0, 5, (500,), generator=gen).float()
        b = a + torch.randn(500, generator=gen)
        self.assertAlmostEqual(self.diag.spearman(a, b), float(stats.spearmanr(a.numpy(), b.numpy())[0]), places=9)
        self.assertTrue(torch.isnan(torch.tensor(self.diag.spearman(torch.ones(10), b[:10]))))

    def test_direct_noise_fraction_recovers_known_variance_fraction(self):
        gen = torch.Generator().manual_seed(1)
        k, n = 400, 2000
        for mu, sigma in [(0.0, 1.0), (1.0, 1.0), (3.0, 1.0)]:
            g = mu + sigma * torch.randn(k, n, generator=gen)
            q, _ = self.diag.direct_noise_fraction(g.sum(0), (g * g).sum(0), k)
            expected = sigma ** 2 / (sigma ** 2 + mu ** 2)
            self.assertAlmostEqual(float(q.mean()), expected, delta=0.03)

    def test_q_is_zero_for_identical_gradients_and_bounded(self):
        g = torch.full((8, 50), 0.7)
        q, _ = self.diag.direct_noise_fraction(g.sum(0), (g * g).sum(0), 8)
        self.assertLess(float(q.max()), 1e-5)
        h = torch.randn(3, 100)
        q2, _ = self.diag.direct_noise_fraction(h.sum(0), (h * h).sum(0), 3)
        self.assertLessEqual(float(q2.max()), 1.0)
        self.assertGreaterEqual(float(q2.min()), 0.0)

    def test_proxy_score_matches_the_formula(self):
        m, v = torch.randn(100), torch.rand(100) + 0.1
        s = self.diag.proxy_score(m, v, 0.1, 1e-8)
        expected = torch.clamp(v - m * m, min=0.0) / (v + 0.1 * m.abs() + 1e-8)
        self.assertTrue(torch.equal(s, expected))
        self.assertTrue(bool((s >= 0).all()) and bool((s <= 1).all()))

    def test_collect_diagnostics_does_not_change_model_or_optimizer(self):
        torch.manual_seed(3)
        model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.ReLU(), nn.Flatten(), nn.Linear(4 * 30 * 30, 10))
        opt = self.AdamNV(model.parameters(), lr=1e-3)
        data = torch.utils.data.TensorDataset(torch.randn(64, 3, 32, 32), torch.randint(0, 10, (64,)))
        loader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True)
        loss_fn = nn.CrossEntropyLoss()
        for x, y in loader:
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        before_model = {k: v.clone() for k, v in model.state_dict().items()}
        before_opt = [(k, v.clone()) for p in opt.state for k, v in opt.state[p].items() if torch.is_tensor(v)]
        args = types.SimpleNamespace(k_batches=3, batch_size=16, sample=500, seed=0)
        rec = self.diag.collect_diagnostics(model, opt, data, args, torch.device("cpu"), epoch=5)
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(v, before_model[k]), k)
        after_opt = [(k, v) for p in opt.state for k, v in opt.state[p].items() if torch.is_tensor(v)]
        for (k1, a), (k2, b) in zip(before_opt, after_opt):
            self.assertTrue(torch.equal(a, b), k1)
        for key in ("spearman_s_q", "spearman_s_scale", "spearman_rho_q", "spearman_q_scale"):
            if rec[key] == rec[key]:  # not NaN
                self.assertTrue(-1.0 <= rec[key] <= 1.0)
        self.assertEqual(rec["k_batches"], 3)
        self.assertEqual(rec["n_elements"], sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
