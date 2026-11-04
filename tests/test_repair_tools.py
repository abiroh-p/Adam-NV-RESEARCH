"""Tests for tools/provenance.py, tools/check_runs.py, analysis/run_io.py and statistics integrity."""
import hashlib, importlib.util, io, json, subprocess, sys, tempfile, unittest, zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis")); sys.path.insert(0, str(ROOT / "tools"))
import run_io, provenance  # noqa: E402


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m

E13 = _mod("e13", ROOT / "analysis" / "analyze_e13_e15.py")
CHECK = _mod("check_runs", ROOT / "tools" / "check_runs.py")


def write_run(path, n=60, acc=None, dup=None, drop=None, nan_at=None):
    acc = acc if acc is not None else [0.5 + 0.005 * i for i in range(1, n + 1)]
    epochs = list(range(1, n + 1))
    df = pd.DataFrame({"epoch": epochs, "val_acc": acc[:n], "train_loss": 1.0})
    if drop is not None: df = df[df.epoch != drop]
    if dup is not None: df = pd.concat([df, df[df.epoch == dup]])
    if nan_at is not None: df.loc[df.epoch == nan_at, "val_acc"] = float("nan")
    df.to_csv(path, index=False)


class ProvenanceToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.arch = self.tmp / "provenance" / "source_archive"; self.arch.mkdir(parents=True)
        self.src = b"line1\r\nline2\nline3\r\n"          # mixed EOL like the archived noise.py
        h = hashlib.sha256(self.src).hexdigest()
        (self.arch / f"noise.py@{h[:8]}").write_bytes(self.src)
        provenance.REPO = self.tmp                       # registry paths are repo-relative
        self.reg = provenance.build_registry(self.arch)
        self.h = h

    def test_exact_match(self):
        self.assertEqual(provenance.classify("noise.py", self.h, self.reg)[0], "VERIFIED_EXACT")

    def test_renamed_file_is_flagged_not_hidden(self):
        self.assertEqual(provenance.classify("train_resnet.py", self.h, self.reg)[0], "VERIFIED_EXACT_RENAMED")

    def test_line_ending_variant_is_eol_normalized_not_exact(self):
        lf = self.src.replace(b"\r\n", b"\n")
        v, _ = provenance.classify("noise.py", hashlib.sha256(lf).hexdigest(), self.reg)
        self.assertEqual(v, "VERIFIED_EOL_NORMALIZED")

    def test_one_changed_byte_is_detected(self):
        bad = hashlib.sha256(self.src + b" ").hexdigest()
        self.assertEqual(provenance.classify("noise.py", bad, self.reg)[0], "UNVERIFIED_NOT_ARCHIVED")

    def test_wrong_label_in_archive_is_rejected(self):
        (self.arch / "x.py@deadbeef").write_bytes(b"abc")
        with self.assertRaises(SystemExit):
            provenance.build_registry(self.arch)

    def test_verify_walks_directories_and_nested_zips_and_reports_missing_hash(self):
        meta_ok = {"code_sha256": {"noise.py": self.h}}
        meta_bad = {"code_sha256": {"noise.py": "0" * 64}}
        (self.tmp / "logs").mkdir()
        (self.tmp / "logs" / "a_metadata.json").write_text(json.dumps(meta_ok))
        (self.tmp / "logs" / "b_metadata.json").write_text(json.dumps({"arguments": {}}))
        inner = io.BytesIO()
        with zipfile.ZipFile(inner, "w") as z: z.writestr("deep/c_metadata.json", json.dumps(meta_bad))
        with zipfile.ZipFile(self.tmp / "logs" / "outer.zip", "w") as z: z.writestr("inner.zip", inner.getvalue())
        rows = provenance.verify(self.tmp, self.reg)
        got = {(r["run"], r["verdict"]) for r in rows}
        self.assertEqual(got, {("a", "VERIFIED_EXACT"), ("b", "NO_HASH_RECORDED"), ("c", "UNVERIFIED_NOT_ARCHIVED")})


    def test_folder_level_hash_file_covers_hashless_metadata_only_when_all_claims_verify(self):
        good = self.tmp / "g"; bad = self.tmp / "b"; good.mkdir(); bad.mkdir()
        for d, h in ((good, self.h), (bad, "0" * 64)):
            (d / "run_metadata.json").write_text(json.dumps({"arguments": {}}))
            (d / "_code_hashes_1.txt").write_text(f"{h}  noise.py\n")
        rows = provenance.apply_folder_coverage(provenance.verify(self.tmp, self.reg))
        by = {(r["member"]): r["verdict"] for r in rows if r["member"].endswith("run_metadata.json")}
        self.assertEqual(by["g/run_metadata.json"], "COVERED_BY_FOLDER_HASHES")
        self.assertEqual(by["b/run_metadata.json"], "NO_HASH_RECORDED")


class RepoProvenanceTests(unittest.TestCase):
    def test_every_archived_file_matches_registry(self):
        reg = provenance.load_registry(ROOT / "provenance" / "source_hash_registry.json")
        self.assertGreaterEqual(len(reg["entries"]), 9)
        for e in reg["entries"]:
            self.assertEqual(hashlib.sha256((ROOT / e["archived_as"]).read_bytes()).hexdigest(), e["sha256"])

    def test_known_unverified_hashes_are_still_reported_unverified(self):
        reg = provenance.load_registry(ROOT / "provenance" / "source_hash_registry.json")
        for name, pre in (("adam_nv.py", "969712867050"), ("gono.py", "82665f3c0bed"), ("train_resnet.py", "7b5134fd2472"),
                          ("train_resnet.py", "5ac384a15bc8")):
            full = pre + "0" * (64 - len(pre))   # prefix cannot match any archived 64-hex digest
            self.assertFalse(any(e["sha256"].startswith(pre) for e in reg["entries"]), pre)


class RunIntegrityTests(unittest.TestCase):
    def setUp(self): self.d = Path(tempfile.mkdtemp())

    def test_complete_run_loads(self):
        write_run(self.d / "ok.csv"); self.assertEqual(len(run_io.load_run(self.d / "ok.csv").frame), 60)

    def test_missing_last_epoch_detected(self):
        write_run(self.d / "a.csv", n=59)
        with self.assertRaisesRegex(run_io.RunIntegrityError, "missing epochs \\[60\\]"): run_io.load_run(self.d / "a.csv")

    def test_gap_duplicate_nan_detected(self):
        write_run(self.d / "g.csv", drop=17)
        with self.assertRaisesRegex(run_io.RunIntegrityError, "17"): run_io.load_run(self.d / "g.csv")
        write_run(self.d / "d.csv", dup=5)
        with self.assertRaisesRegex(run_io.RunIntegrityError, "duplicated"): run_io.load_run(self.d / "d.csv")
        write_run(self.d / "n.csv", nan_at=30)
        with self.assertRaisesRegex(run_io.RunIntegrityError, "non-finite"): run_io.load_run(self.d / "n.csv")

    def test_final_epoch_and_last5_are_different_quantities(self):
        acc = [0.5] * 55 + [0.6, 0.6, 0.6, 0.6, 0.9]      # last5 mean = 0.66, final = 0.9
        write_run(self.d / "m.csv", acc=acc); r = run_io.load_run(self.d / "m.csv")
        self.assertAlmostEqual(run_io.final_epoch_metric(r), 0.9)
        self.assertAlmostEqual(run_io.last_k_epoch_mean_metric(r), 0.66)

    def test_supplement_is_opt_in_fills_only_missing_epoch_and_never_overwrites(self):
        csv = self.d / "s.csv"; write_run(csv, n=59)
        sup_dir = self.d / "sup"; sup_dir.mkdir()
        doc = {"run_csv_sha256": run_io.sha256_file(csv), "recovered": {"60": {"val_acc": 0.9}, "10": {"val_acc": 0.123}}}
        (sup_dir / "s.json").write_text(json.dumps(doc))
        with self.assertRaises(run_io.RunIntegrityError): run_io.load_run(csv, supplements_dir=sup_dir)      # default: off
        r = run_io.load_run(csv, use_supplement=True, supplements_dir=sup_dir)
        self.assertEqual(set(r.supplement_used), {60})                       # epoch 10 exists -> untouched
        self.assertAlmostEqual(float(r.frame.loc[r.frame.epoch == 10, "val_acc"].iloc[0]), 0.5 + 0.05)
        self.assertTrue(r.used_supplement)

    def test_supplement_not_applied_to_a_different_file(self):
        csv = self.d / "t.csv"; write_run(csv, n=59); other = self.d / "u.csv"; write_run(other, n=59, acc=[0.4] * 59)
        sup_dir = self.d / "sup"; sup_dir.mkdir()
        (sup_dir / "s.json").write_text(json.dumps({"run_csv_sha256": run_io.sha256_file(csv), "recovered": {"60": {"val_acc": 0.9}}}))
        with self.assertRaises(run_io.RunIntegrityError):
            run_io.load_run(other, use_supplement=True, supplements_dir=sup_dir)

    def test_check_runs_cli_flags_problems_and_strict_exit(self):
        root = self.d / "repo"; (root / "logs").mkdir(parents=True)
        write_run(root / "logs" / "ok_seed0.csv"); write_run(root / "logs" / "short_seed1.csv", n=59)
        write_run(root / "logs" / "dup_seed2.csv", dup=3)
        out = root / "out.csv"
        rc = CHECK.main(["--root", str(root), "--csv", str(out), "--strict"])
        self.assertEqual(rc, 1)
        st = dict(zip(pd.read_csv(out).path, pd.read_csv(out).status))
        self.assertEqual(st["logs/ok_seed0.csv"], "OK"); self.assertEqual(st["logs/short_seed1.csv"], "INCOMPLETE")
        self.assertEqual(st["logs/dup_seed2.csv"], "DUPLICATE_EPOCHS")


class StatisticsIntegrityTests(unittest.TestCase):
    def test_pairing_uses_matching_seeds_and_reports_unpaired(self):
        left = {0: 0.93, 1: 0.94, 2: 0.95, 3: 0.96}; right = {0: 0.92, 1: 0.93, 3: 0.95}
        r = E13.paired_result("T", "x", "l", "r", left, right, [0, 1, 2, 3])
        self.assertEqual(r["paired_seeds"], "0 1 3"); self.assertFalse(r["complete"]); self.assertEqual(r["n_pairs"], 3)

    def test_single_pair_is_rejected(self):
        with self.assertRaises(ValueError): E13.paired_result("T", "x", "l", "r", {0: .9}, {0: .8}, [0])

    def test_percentage_point_conversion(self):
        r = E13.paired_result("T", "x", "l", "r", {0: .9330, 1: .9340}, {0: .9320, 1: .9330}, [0, 1])
        self.assertAlmostEqual(r["mean_difference_percentage_points"], 100 * r["mean_difference"])
        self.assertAlmostEqual(r["mean_difference_percentage_points"], 0.1, places=9)

    def test_exact_sign_flip_known_values(self):
        self.assertAlmostEqual(E13.exact_sign_flip_pvalue(np.array([1., 2., 3.])), 0.25)          # 2/8
        self.assertAlmostEqual(E13.exact_sign_flip_pvalue(np.array([1., -1., 1., -1.])), 1.0)

    def test_holm_known_values_and_monotonicity(self):
        adj = E13.holm_adjust([0.01, 0.04, 0.03])
        np.testing.assert_allclose(adj, [0.03, 0.06, 0.06])
        self.assertTrue(all(0 <= a <= 1 for a in E13.holm_adjust([0.9, 0.8])))

    def test_metric_switch_changes_the_computed_quantity(self):
        d = Path(tempfile.mkdtemp()); write_run(d / "a.csv", acc=[0.5] * 55 + [0.6, 0.6, 0.6, 0.6, 0.9])
        try:
            E13.METRIC_MODE = "last5"; a = E13.trailing_accuracy(d / "a.csv")
            E13.METRIC_MODE = "final"; b = E13.trailing_accuracy(d / "a.csv")
        finally:
            E13.METRIC_MODE = "last5"
        self.assertAlmostEqual(a, 0.66); self.assertAlmostEqual(b, 0.9)

    def test_incomplete_run_is_refused_by_analysis_loader(self):
        d = Path(tempfile.mkdtemp()); write_run(d / "a.csv", n=59)
        with self.assertRaises(ValueError): E13.trailing_accuracy(d / "a.csv")

    def test_committed_recomputed_tables_match_a_fresh_recomputation(self):
        """Table-value consistency: results_recomputed/*.csv must equal what the code gives on the logs."""
        for metric in ("last5", "final"):
            out = Path(tempfile.mkdtemp()) / "t.csv"
            subprocess.run([sys.executable, str(ROOT / "analysis" / "analyze_e13_e15.py"), "--metric", metric, "--output", str(out)],
                           check=True, capture_output=True, env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"})
            fresh = pd.read_csv(out); kept = pd.read_csv(ROOT / "analysis" / "results_recomputed" / f"e13_e15_{metric}_statistics.csv")
            pd.testing.assert_frame_equal(fresh, kept, check_exact=False, rtol=1e-12)

    def test_recomputed_table_marks_all_rows_complete_with_expected_pair_counts(self):
        t = pd.read_csv(ROOT / "analysis" / "results_recomputed" / "e13_e15_last5_statistics.csv")
        self.assertTrue(t["complete"].all()); self.assertEqual(len(t), 22)
        self.assertEqual(int(t[t.comparison.str.startswith("intermittent: Adam-NV")].n_pairs.iloc[0]), 8)
        row = t[(t.experiment == "E14") & (t.comparison == "intermittent: coupled beta2=0.95 minus coupled beta2=0.999")].iloc[0]
        self.assertEqual(row.n_pairs, 4)


if __name__ == "__main__":
    unittest.main()


class DocumentationAndCheckLogsTests(unittest.TestCase):
    TOP = ("analysis/", "tools/", "adam-nv/", "e2_noisy_gradient/", "e1_convergence/", "provenance/",
           "supplements/", "reports/", "docs/", "tests/", "environment/")
    EXT = (".py", ".md", ".json", ".csv", ".sh", ".txt")

    def test_file_paths_named_in_docs_exist(self):
        import re
        missing = []
        for doc in ("README.md", "docs/REPO_LAYOUT.md", "docs/REPRODUCE.md"):
            text = (ROOT / doc).read_text(encoding="utf-8")
            for tok in re.findall(r"`([^`\s]+)`", text):
                if tok.startswith(self.TOP) and tok.endswith(self.EXT) and not any(c in tok for c in "*<>{}…"):
                    if not (ROOT / tok).exists():
                        missing.append(f"{doc}: {tok}")
        self.assertEqual(missing, [])

    def test_check_logs_strict_fails_on_unarchived_hash_and_warn_passes(self):
        src = ROOT / "adam-nv" / "e1_convergence" / "src"
        logs = ROOT / "adam-nv" / "e1_convergence" / "logs"
        base = [sys.executable, "check_logs.py", "--log-dir", str(logs), "--optimizers", "gono", "--seeds", "0",
                "--expect-torch", "", "--expect-torchvision", ""]
        strict = subprocess.run(base + ["--hash-mode", "strict"], cwd=src, capture_output=True, text=True)
        warn = subprocess.run(base + ["--hash-mode", "warn"], cwd=src, capture_output=True, text=True)
        self.assertEqual(strict.returncode, 1); self.assertIn("UNVERIFIED_NOT_ARCHIVED", strict.stdout)
        self.assertEqual(warn.returncode, 0)

    def test_check_logs_strict_passes_a_run_whose_hashes_are_archived(self):
        # exp05 runs recorded hashes that ARE archived (train_resnet_ex.py recorded under the key train_resnet.py)
        d = Path(tempfile.mkdtemp()); z = ROOT / "adam-nv" / "e1_convergence" / "exp05_logs_adam.zip"
        with zipfile.ZipFile(z) as zf: zf.extractall(d)
        sub = next(p for p in d.rglob("adam_lr3e-3_seed0.csv")).parent
        r = subprocess.run([sys.executable, "check_logs.py", "--log-dir", str(sub), "--optimizers", "adam", "--seeds", "0",
                            "--run-tag", "_lr3e-3", "--lr", "0.003", "--expect-torch", "", "--expect-torchvision", ""],
                           cwd=ROOT / "adam-nv" / "e1_convergence" / "src", capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout[-400:])


class Exp07SeedBalanceTest(unittest.TestCase):
    def test_unbalanced_arm_is_flagged_and_strict_mode_fails(self):
        d = Path(tempfile.mkdtemp()); logs = ROOT / "adam-nv/e2_noisy_gradient/logs"
        for n in ("logs_exp07_A", "logs_exp07_B"):
            with zipfile.ZipFile(logs / f"{n}.zip") as z: z.extractall(d / n)
        dirs = [str(next(p for p in (d / n).rglob("*") if p.is_dir() and p.name == n)) for n in ("logs_exp07_A", "logs_exp07_B")]
        cmd = [sys.executable, "adam-nv/analysis/analyze_exp07.py", "--orig-dir", str(ROOT / "e2_noisy_gradient/logs/Label_Diagnostic"), "--new-dirs", *dirs]
        loose = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        strict = subprocess.run(cmd + ["--strict-seeds"], cwd=ROOT, capture_output=True, text=True)
        self.assertIn("unbalanced arms", loose.stdout); self.assertIn("'adam_nv': [4, 5, 6, 7]", loose.stdout)
        self.assertEqual(strict.returncode, 2)


class ZipScanAndRerunPolicyTests(unittest.TestCase):
    def setUp(self): self.d = Path(tempfile.mkdtemp())

    def _csv_bytes(self, n=60, acc=None):
        p = self.d / "tmp.csv"; write_run(p, n=n, acc=acc); return p.read_bytes()

    def test_check_runs_looks_inside_nested_zips(self):
        root = self.d / "repo"; root.mkdir()
        inner = io.BytesIO()
        with zipfile.ZipFile(inner, "w") as z:
            z.writestr("logs/ok_seed0.csv", self._csv_bytes()); z.writestr("logs/ok_seed0_metadata.json", "{}")
            z.writestr("logs/short_seed1.csv", self._csv_bytes(n=58))
        with zipfile.ZipFile(root / "outer.zip", "w") as z: z.writestr("deep/inner.zip", inner.getvalue())
        out = root / "o.csv"; rc = CHECK.main(["--root", str(root), "--csv", str(out), "--strict"])
        st = {Path(p.split(">")[-1]).name: (s, m) for p, s, m in zip(*[pd.read_csv(out)[c] for c in ("path", "status", "metadata")])}
        self.assertEqual(rc, 1)
        self.assertEqual(st["ok_seed0.csv"], ("OK", "yes")); self.assertEqual(st["short_seed1.csv"][0], "INCOMPLETE")
        self.assertEqual(CHECK.main(["--root", str(root), "--no-zips", "--strict"]), 0)  # --no-zips really skips them

    def _e1_tree(self, rerun_acc_shift=0.0, original_complete=False):
        a = _mod("analyze_e1", ROOT / "analysis" / "analyze_e1.py")
        top, gono, rer = self.d / "top", self.d / "gono", self.d / "rerun"
        for p in (top, gono, rer): p.mkdir(exist_ok=True)
        for opt in ("adam", "adam_nv"):
            for s in range(4): write_run(top / f"{opt}_seed{s}.csv", acc=[0.5 + 0.004 * i + 0.001 * s for i in range(1, 61)])
        base = lambda s: [0.45 + 0.004 * i + 0.001 * s for i in range(1, 61)]
        for s in range(4):
            write_run(gono / f"gono_seed{s}.csv", n=59 if (s == 1 and not original_complete) else 60, acc=base(s))
        write_run(rer / "gono_seed1.csv", acc=[v + rerun_acc_shift * (i < 59) for i, v in enumerate(base(1))])
        return a, top, gono, rer

    def _run(self, a, top, gono, rer, extra=()):
        out = self.d / "out"
        rc = a.main(["--top-logs", str(top), "--additional-logs", str(self.d / "none"), "--gono-logs", str(gono),
                     "--rerun-logs", str(rer), "--supplements", str(self.d / "nosup"), "--out-dir", str(out), *extra])
        return rc, out

    def test_complete_rerun_replaces_incomplete_original_after_consistency_check(self):
        rc, out = self._run(*self._e1_tree())
        self.assertEqual(rc, 0)
        ps = pd.read_csv(out / "e1_per_seed.csv"); g = ps[(ps.optimizer == "gono") & (ps.seed == 1)].iloc[0]
        self.assertIn("rerun replacing incomplete original", g.source_folder)
        self.assertEqual(int((ps.optimizer == "gono").sum()), 4)
        cons = pd.read_csv(out / "e1_rerun_consistency.csv"); self.assertTrue(bool(cons.consistent.iloc[0])); self.assertEqual(int(cons.overlapping_epochs.iloc[0]), 59)

    def test_inconsistent_rerun_is_refused_not_merged(self):
        rc, _out = self._run(*self._e1_tree(rerun_acc_shift=0.01))
        self.assertEqual(rc, 2)

    def test_complete_original_always_wins_over_a_rerun(self):
        rc, out = self._run(*self._e1_tree(original_complete=True, rerun_acc_shift=0.01))
        self.assertEqual(rc, 0)
        ps = pd.read_csv(out / "e1_per_seed.csv"); g = ps[(ps.optimizer == "gono") & (ps.seed == 1)].iloc[0]
        self.assertNotIn("rerun", g.source_folder)
        self.assertFalse((out / "e1_rerun_consistency.csv").exists())

    def test_without_rerun_the_incomplete_run_is_excluded_and_listed(self):
        a, top, gono, rer = self._e1_tree(); (rer / "gono_seed1.csv").unlink()
        rc, out = self._run(a, top, gono, rer)
        self.assertEqual(rc, 0); self.assertEqual(int((pd.read_csv(out / "e1_per_seed.csv").optimizer == "gono").sum()), 3)
        self.assertEqual(int(pd.read_csv(out / "e1_excluded_runs_last5.csv").seed.iloc[0]), 1)

    def test_known_superseded_original_and_complete_rerun_in_real_repo(self):
        st = pd.read_csv(ROOT / "reports" / "run_completeness.csv"); st = dict(zip(st.path, st.status))
        self.assertEqual(st["adam-nv/e1_convergence/logs/gono_seed1.csv"], "KNOWN_PARTIAL_SUPERSEDED")
        self.assertEqual(st["adam-nv/e1_convergence/logs/gono_seed_1/gono_seed1.csv"], "OK")


class SeedGuardAcrossExperimentScriptsTests(unittest.TestCase):
    def test_check_seed_balance_function(self):
        import io as _io, contextlib
        e7 = _mod("analyze_exp07_guard", ROOT / "adam-nv" / "analysis" / "analyze_exp07.py")
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(e7.check_seed_balance({"a": {0: 1, 1: 1}, "b": {0: 1}}, [0, 1]), {"b": [1]})
            self.assertEqual(e7.check_seed_balance({"a": {0: 1, 1: 1}}, [0, 1]), {})
            with self.assertRaises(SystemExit) as cm: e7.check_seed_balance({"b": {0: 1}}, [0, 1], strict=True)
        self.assertEqual(cm.exception.code, 2); self.assertIn("unbalanced arms", buf.getvalue())

    def test_exp12_and_exp09_warn_and_fail_strict_on_missing_seeds(self):
        d = Path(tempfile.mkdtemp()); L = ROOT / "adam-nv/e2_noisy_gradient/logs"
        for n in ("logs_exp07_A", "logs_exp07_B"):
            with zipfile.ZipFile(L / f"{n}.zip") as z: z.extractall(d / n)
        with zipfile.ZipFile(L / "exp07_log_C.zip") as z: z.extractall(d / "c")
        with zipfile.ZipFile(L / "exp09_logs_A.zip") as z: z.extractall(d / "e9")
        with zipfile.ZipFile(L / "exp12_logs_3.zip") as z: z.extractall(d / "e12")
        nm = lambda root, n: str(next(p for p in root.rglob("*") if p.is_dir() and p.name == n))
        base = ["--orig-dir", str(ROOT / "e2_noisy_gradient/logs/Label_Diagnostic"), "--exp07-dirs", nm(d / "logs_exp07_A", "logs_exp07_A"), nm(d / "logs_exp07_B", "logs_exp07_B"), nm(d / "c", "log_exp07_C")]
        smooth = ["--smooth-dirs", nm(d / "e9", "logs_exp09_A")]   # only 4 of 8 smooth seeds
        r9 = subprocess.run([sys.executable, "adam-nv/analysis/analyze_exp09.py", *base, *smooth], cwd=ROOT, capture_output=True, text=True)
        r9s = subprocess.run([sys.executable, "adam-nv/analysis/analyze_exp09.py", *base, *smooth, "--strict-seeds"], cwd=ROOT, capture_output=True, text=True)
        self.assertIn("'smooth': [4, 5, 6, 7]", r9.stdout); self.assertEqual(r9s.returncode, 2)
        r12 = subprocess.run([sys.executable, "adam-nv/analysis/analyze_exp12.py", *base, *smooth, "--c095-dirs", nm(d / "e12", "logs_exp12_b095_1"), "--strict-seeds"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r12.returncode, 2); self.assertIn("unbalanced arms", r12.stdout)


class PaperDraftCheckStateTest(unittest.TestCase):
    def test_only_the_documented_claims_are_not_matches(self):
        d = pd.read_csv(ROOT / "reports" / "paper_draft_claim_check.csv")
        bad = d[d.verdict != "MATCH"]
        self.assertEqual(sorted(bad.verdict.value_counts().to_dict().items()), [("DIFFERS", 1), ("NOT VERIFIABLE", 6)])
        self.assertEqual(set(bad.draft_location), {"Table I", "Table II", "V.A", "Table III / Abstract", "Table III"})
        self.assertTrue((d.verdict == "MATCH").sum() >= 90)
