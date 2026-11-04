"""Adam-NV optimizer checks added by the repository repair.

Tested file: the byte-exact archived adam_nv.py@bcc7824b (the version pinned by the
EXP07/EXP08 runbooks and recorded in the exp03/05/07/08 run metadata), and the older
adam_nv.py@2d6cd7df for the old-vs-new equivalence check. Nothing here modifies the algorithm.

Known, documented deviation (NOT hidden by tolerances): running product P is clamped at 1e-7,
so with gamma=0 the optimizer equals torch.optim.Adam to ~1e-15 only while beta^t > 1e-7
(about 150 steps at beta1=0.9). Afterwards the difference is ~1e-9..1e-7 in parameters.
The module docstring's "reduces exactly to Adam" is therefore only true up to that horizon.
"""
import importlib.machinery
import importlib.util
import unittest
import warnings
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

ROOT = Path(__file__).resolve().parents[1]
ARCH = ROOT / "provenance" / "source_archive"


def _load(name):
    modname = "adam_nv_" + name.replace("@", "_").replace(".", "_")
    loader = importlib.machinery.SourceFileLoader(modname, str(ARCH / name))  # '@hash' suffix: explicit loader
    spec = importlib.util.spec_from_loader(modname, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod.AdamNV


def _grads(steps, n, seed):
    rng = np.random.default_rng(seed)
    return [rng.normal(0.1, 0.5, n) for _ in range(steps)]


def run_torch(cls, steps, n=20, seed=0, lr=1e-2, **kw):
    p = torch.nn.Parameter(torch.tensor(np.linspace(-1, 1, n), dtype=torch.float64))
    opt = cls([p], lr=lr, **kw)
    for g in _grads(steps, n, seed):
        opt.zero_grad()
        p.grad = torch.tensor(g, dtype=torch.float64)
        opt.step()
    return p.detach().numpy().copy(), opt


def reference_adam_nv(steps, n=20, seed=0, lr=1e-2, beta_base=0.9, beta_min=0.1, beta_2=0.999, gamma=1.0,
                      alpha=0.95, k=100, lam=0.1, eps=1e-8, mode="distrust", beta_max=0.999, floor=1e-7):
    """Independent float64 NumPy transcription of the documented Adam-NV formulation."""
    p = np.linspace(-1, 1, n)
    m = np.zeros(n); v = np.zeros(n); rho = np.zeros(n); P = np.ones(n)
    for t, g in enumerate(_grads(steps, n, seed), start=1):
        sigma2 = np.maximum(v - m**2, 0.0) / (v + lam * np.abs(m) + eps)      # stale moments
        rho = alpha * rho + (1 - alpha) * (t / (t + k)) * sigma2
        if mode == "distrust":
            b1 = beta_min + (beta_base - beta_min) * np.maximum(1 - gamma * rho, 0.0)
        else:
            b1 = beta_base + (beta_max - beta_base) * np.clip(gamma * rho, 0.0, 1.0)
        m = b1 * m + (1 - b1) * g
        v = beta_2 * v + (1 - beta_2) * g * g
        P = np.maximum(P * b1, floor)
        m_hat = m / np.maximum(1 - P, eps)
        v_hat = v / (1 - beta_2**t)
        p = p - lr * m_hat / (np.sqrt(v_hat) + eps)
    return p


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class AdamNVRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore", RuntimeWarning)
        cls.New = _load("adam_nv.py@bcc7824b")
        cls.Old = _load("adam_nv.py@2d6cd7df")

    # --- agreement with an independent reference -------------------------------------------
    def test_matches_independent_reference_distrust_many_steps(self):
        for steps in (1, 2, 5, 60, 400):
            got, _ = run_torch(self.New, steps)
            ref = reference_adam_nv(steps)
            self.assertLess(np.abs(got - ref).max(), 1e-11, f"steps={steps}")

    def test_matches_independent_reference_smooth_mode(self):
        got, _ = run_torch(self.New, 300, response_mode="smooth")
        self.assertLess(np.abs(got - reference_adam_nv(300, mode="smooth")).max(), 1e-11)

    def test_matches_reference_with_weightless_gamma_and_alpha_variants(self):
        got, _ = run_torch(self.New, 200, gamma=0.5, alpha=0.9, k=20, lam=0.3)
        ref = reference_adam_nv(200, gamma=0.5, alpha=0.9, k=20, lam=0.3)
        self.assertLess(np.abs(got - ref).max(), 1e-11)

    # --- gamma = 0 / fixed beta1 relationship to torch Adam ---------------------------------
    def test_gamma_zero_equals_torch_adam_within_clamp_horizon(self):
        for steps in (1, 2, 3, 10, 150):   # 0.9**150 = 1.4e-7 > 1e-7 floor
            ours, _ = run_torch(self.New, steps, gamma=0.0)
            ref, _ = run_torch(torch.optim.Adam, steps, betas=(0.9, 0.999), eps=1e-8)
            self.assertLess(np.abs(ours - ref).max(), 1e-12, f"steps={steps}")

    def test_gamma_zero_long_horizon_deviation_is_small_but_documented(self):
        ours, _ = run_torch(self.New, 3000, gamma=0.0)
        ref, _ = run_torch(torch.optim.Adam, 3000, betas=(0.9, 0.999), eps=1e-8)
        d = np.abs(ours - ref).max()
        self.assertLess(d, 1e-6)  # observed ~3e-8; the clamp makes it NOT exact

    def test_fixed_beta1_matches_torch_adam(self):
        for b1, steps, tol in ((0.81, 50, 1e-12), (0.81, 500, 1e-6), (0.5, 20, 1e-12)):
            ours, _ = run_torch(self.New, steps, gamma=0.0, beta_base=b1, beta_min=b1)
            ref, _ = run_torch(torch.optim.Adam, steps, betas=(b1, 0.999), eps=1e-8)
            self.assertLess(np.abs(ours - ref).max(), tol, f"b1={b1} steps={steps}")

    # --- bias correction --------------------------------------------------------------------
    def test_first_step_bias_correction_is_exact(self):
        p = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64))
        opt = self.New([p], lr=0.1)
        g = torch.tensor([0.3, -0.7, 2.0], dtype=torch.float64)
        p.grad = g.clone(); before = p.detach().clone(); opt.step()
        # m_hat = g, v_hat = g^2  ->  step = lr * g / (|g| + eps)
        expected = before - 0.1 * g / (g.abs() + 1e-8)
        self.assertTrue(torch.allclose(p.detach(), expected, atol=1e-12))

    def test_running_product_equals_beta_power_when_gamma_zero(self):
        _, opt = run_torch(self.New, 7, gamma=0.0)
        P = next(iter(opt.state.values()))["P"].numpy()
        np.testing.assert_allclose(P, 0.9**7, rtol=1e-12)

    def test_base_bias_correction_is_a_distinct_ablation(self):
        a, _ = run_torch(self.New, 80)
        b, _ = run_torch(self.New, 80, bias_correction="base")
        self.assertGreater(np.abs(a - b).max(), 1e-6)

    # --- numerical stability ----------------------------------------------------------------
    def test_zero_gradients_never_produce_nan(self):
        p = torch.nn.Parameter(torch.ones(5, dtype=torch.float64)); opt = self.New([p])
        for _ in range(300):
            p.grad = torch.zeros_like(p); opt.step()
        self.assertTrue(torch.isfinite(p).all())

    def test_huge_and_tiny_gradients_stay_finite(self):
        for scale in (1e30, 1e-30):
            p = torch.nn.Parameter(torch.ones(4, dtype=torch.float64)); opt = self.New([p], lr=1e-3)
            rng = np.random.default_rng(1)
            for _ in range(200):
                p.grad = torch.tensor(rng.normal(size=4) * scale, dtype=torch.float64); opt.step()
            self.assertTrue(torch.isfinite(p).all(), scale)

    def test_float32_path_is_finite_over_many_steps(self):
        p = torch.nn.Parameter(torch.randn(16)); opt = self.New([p], lr=1e-3, beta_min=0.1)
        g = torch.Generator().manual_seed(0)
        for _ in range(1500):
            p.grad = torch.randn(16, generator=g); opt.step()
        self.assertTrue(torch.isfinite(p).all())

    def test_beta1_stays_within_bounds_and_state_is_per_element(self):
        _, opt = run_torch(self.New, 120)
        st = next(iter(opt.state.values()))
        self.assertEqual(st["t"], 120)
        self.assertEqual(st["rho"].shape, st["m"].shape)
        self.assertTrue(((st["rho"] >= 0) & (st["rho"] <= 1)).all())

    # --- parameter groups / missing gradients / weight decay --------------------------------
    def test_param_groups_use_their_own_hyperparameters(self):
        a = torch.nn.Parameter(torch.ones(3, dtype=torch.float64)); b = torch.nn.Parameter(torch.ones(3, dtype=torch.float64))
        opt = self.New([{"params": [a], "lr": 0.1}, {"params": [b], "lr": 0.001}])
        for _ in range(3):
            a.grad = torch.full_like(a, 0.5); b.grad = torch.full_like(b, 0.5); opt.step()
        self.assertGreater((1 - a).abs().mean().item(), 10 * (1 - b).abs().mean().item())

    def test_missing_gradient_is_skipped_and_creates_no_state(self):
        a = torch.nn.Parameter(torch.ones(2, dtype=torch.float64)); b = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
        opt = self.New([a, b]); a.grad = torch.ones_like(a); opt.step()
        self.assertTrue(torch.equal(b.detach(), torch.ones(2, dtype=torch.float64)))
        self.assertIn(a, opt.state); self.assertNotIn(b, opt.state)

    def test_weight_decay_is_coupled_l2_like_torch_adam(self):
        ours, _ = run_torch(self.New, 40, gamma=0.0, weight_decay=0.01)
        ref, _ = run_torch(torch.optim.Adam, 40, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        self.assertLess(np.abs(ours - ref).max(), 1e-12)

    def test_gradient_tensor_is_not_modified_by_weight_decay(self):
        p = torch.nn.Parameter(torch.ones(3, dtype=torch.float64)); opt = self.New([p], weight_decay=0.1)
        p.grad = torch.full_like(p, 0.25); opt.step()
        self.assertTrue(torch.equal(p.grad, torch.full_like(p, 0.25)))

    # --- historical versions ----------------------------------------------------------------
    def test_old_and_new_versions_are_bit_identical_at_default_settings(self):
        for kw in ({}, {"response_mode": "smooth"}, {"beta_2_noise": 0.95}, {"gamma": 0.0}):
            o, _ = run_torch(self.Old, 250, **kw); n, _ = run_torch(self.New, 250, **kw)
            self.assertTrue(np.array_equal(o, n), kw)

    # --- deterministic regression (characterization, NOT a correctness proof) -----------------
    def test_golden_regression_values(self):
        got, _ = run_torch(self.New, 100, seed=7)
        golden = np.load(Path(__file__).with_name("golden_adam_nv_seed7_100steps.npy"))
        np.testing.assert_allclose(got, golden, rtol=0, atol=1e-13)


if __name__ == "__main__":
    unittest.main()
