"""Exp. 21 analysis tests: the pre-registered rule returns TRACKS, DOES NOT TRACK, INCONCLUSIVE and UNRESOLVED on synthetic records
whose within-tensor statistics are set to known values (built from the pipeline controls of test_exp21_diagnostic_v2)."""
import importlib.util, io, json, tempfile, unittest, contextlib, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]


def _load():
    sys.path.insert(0, str(ROOT / "analysis"))
    spec = importlib.util.spec_from_file_location("an21", ROOT / "analysis" / "analyze_exp21.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def _record(epoch, r_s_qu, rel, r_rms, rng):
    ts = []
    for i in range(62):
        rank = 4 if i < 20 else 1
        ts.append({"tensor": i, "rank": rank, "numel": 100, "n_used": 100, "mean_s": 0.01 + 0.001 * i, "mean_s0": 0.5, "mean_q": 0.9, "mean_qu": 0.9 + 0.001 * i, "mean_rms": 1e-4 * (i + 1), "median_rms": 1e-4,
                   "frac_q_eq1": 0.5, "r_s_q": 0, "r_s_qu": r_s_qu + 0.01 * rng.standard_normal(), "r_s0_qu": r_s_qu + 0.02, "r_s_rms": r_rms + 0.01 * rng.standard_normal(), "r_s0_rms": 0.0,
                   "r_qu_rms": 0.0, "pr_s_qu_given_rms": r_s_qu, "pr_s0_qu_given_rms": r_s_qu, "r_half": rel / (2 - rel), "reliability": rel + 0.01 * rng.standard_normal(),
                   "r_s_qstream": 0.1, "r_s_q16": 0.0})
    return {"epoch": epoch, "k_batches": 64, "k_legacy": 16, "mean_qu": 0.9, "frac_q16_eq1": 0.5, "spearman_s_qu": r_s_qu, "spearman_s_q16": 0.0, "tensors": ts}


def _write(d, ln, seeds, r_s_qu, rel, r_rms, seed_sd=0.02):
    rng = np.random.default_rng(0)
    for s in seeds:
        shift = seed_sd * rng.standard_normal()
        with open(Path(d) / f"proxy2_diag_ln{ln}_seed{s}.jsonl", "w") as fh:
            for e in (1, 5, 10, 20, 30, 45, 60): fh.write(json.dumps(_record(e, r_s_qu + shift, rel, r_rms, rng)) + "\n")


def _run(an, d, ln):
    import sys as _s
    old = _s.argv; _s.argv = ["x", "--log-dirs", str(d), "--conditions", ln]; buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf): an.main()
    finally: _s.argv = old
    return buf.getvalue()


class AnalysisRuleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.an = _load()

    def _rule_outcome(self, r_s_qu, rel, r_rms=0.0, seed_sd=0.02):
        d = tempfile.mkdtemp(); _write(d, "0.0", range(5), r_s_qu, rel, r_rms, seed_sd); out = _run(self.an, d, "0.0")
        return [l for l in out.splitlines() if "PRE-REGISTERED OUTCOME" in l][0], out

    def test_tracks(self): self.assertIn("TRACKS", self._rule_outcome(0.76, 0.585)[0]); self.assertNotIn("DOES NOT", self._rule_outcome(0.76, 0.585)[0])           # R* = 0.99
    def test_does_not_track(self): self.assertIn("DOES NOT TRACK", self._rule_outcome(0.01, 0.58)[0])
    def test_unresolved_when_the_estimate_cannot_rank(self): self.assertIn("UNRESOLVED", self._rule_outcome(0.30, 0.10)[0])
    def test_inconclusive_in_between(self): self.assertIn("INCONCLUSIVE", self._rule_outcome(0.12, 0.5, seed_sd=0.05)[0])
    def test_scale_dominated_flag(self):
        _, out = self._rule_outcome(0.05, 0.5, r_rms=0.4); self.assertIn("? True", [l for l in out.splitlines() if "scale-dominated" in l][0])
        _, out = self._rule_outcome(0.76, 0.585, r_rms=0.05); self.assertIn("? False", [l for l in out.splitlines() if "scale-dominated" in l][0])


if __name__ == "__main__":
    unittest.main()
