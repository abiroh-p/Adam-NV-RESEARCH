#!/usr/bin/env python3
"""Run every validation step and record command, exit code, duration and output.

Writes reports/validation_logs/<step>.txt and reports/validation_summary.json.
A step passes only if its process exits 0; nothing is marked passed otherwise. Steps that need GPU
training are NOT included (they are listed as not run in reports/VALIDATION_REPORT.md).
"""
import json, os, shutil, subprocess, sys, tempfile, time, zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOGS = REPO / "reports" / "validation_logs"; LOGS.mkdir(parents=True, exist_ok=True)
PY = sys.executable
env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
tmp = Path(tempfile.mkdtemp())
def unzip(rel, dest):
    with zipfile.ZipFile(REPO / rel) as z: z.extractall(tmp / dest)
    return tmp / dest
E2L = "adam-nv/e2_noisy_gradient/logs"
a = unzip(f"{E2L}/logs_exp07_A.zip", "e7a"); b = unzip(f"{E2L}/logs_exp07_B.zip", "e7b"); c = unzip(f"{E2L}/exp07_log_C.zip", "e7c")
b1 = unzip(f"{E2L}/exp08_logs_A.zip", "e8a"); b2 = unzip(f"{E2L}/exp08_logs_B.zip", "e8b"); b3 = unzip(f"{E2L}/logs_exp08_b070_C.zip", "e8c")
e9 = [unzip(f"{E2L}/exp09_logs_{n}.zip", f"e9{n}") for n in ("A", "B")]
e10 = [unzip(f"adam-nv/e1_convergence/exp10_logs_{n}.zip", f"e10{n}") for n in ("A", "B")]
e10c = unzip(f"{E2L}/exp10_logs_C.zip", "e10C")
e11 = {n: unzip(f"{E2L}/exp11_logs_{n}.zip", f"e11{n}") for n in "ABC"}
e12 = {i: unzip(f"{E2L}/exp12_logs_{i}.zip", f"e12{i}") for i in (1, 2, 3, 4)}
x05 = [unzip(f"adam-nv/e1_convergence/exp05_logs_{n}.zip", f"e5{n}") for n in ("adam", "adamnv")]
def first(p, pat): return next(p.rglob(pat)).parent
def named(root, name): return next(d for d in root.rglob("*") if d.is_dir() and d.name == name)
exp07 = [named(a, "logs_exp07_A"), named(b, "logs_exp07_B"), named(c, "log_exp07_C")]  # explicit by name: C also holds a canary folder
canary_dir = next(p for p in c.rglob("*") if p.is_dir() and p.name == "logs_exp07_canary")
lab = REPO / "e2_noisy_gradient/logs/Label_Diagnostic"
b078 = [named(b1, "logs_exp08_b078_A"), named(b2, "logs_exp08_b078_B")]
b070 = [named(b1, "logs_exp08_b070_A"), named(b2, "logs_exp08_b070_B"), named(b3, "logs_exp08_b070_C")]
steps = [
 ("01_imports", [PY, "-c", "import sys; sys.path[:0]=['analysis','tools']; import run_io, provenance; import importlib.util as u; [u.spec_from_file_location('m',p).loader.exec_module(u.module_from_spec(u.spec_from_file_location('m',p))) for p in ('analysis/analyze_e1.py','analysis/analyze_e13_e15.py','tools/check_runs.py')]; print('imports ok')"], REPO),
 ("02_unit_tests_all_roots", [PY, "tools/run_all_tests.py", "-v"], REPO),
 ("03_provenance_verify", [PY, "tools/provenance.py", "verify", "--csv", "reports/provenance_verification.csv"], REPO),
 ("04_provenance_verify_strict_EXPECTED_TO_FAIL", [PY, "tools/provenance.py", "verify", "--strict"], REPO),
 ("05_run_completeness", [PY, "tools/check_runs.py", "--csv", "reports/run_completeness.csv"], REPO),
 ("06_run_completeness_strict", [PY, "tools/check_runs.py", "--strict"], REPO),   # passes only because every incomplete log is registered as superseded by a complete rerun
 ("07_e1_last5", [PY, "analysis/analyze_e1.py", "--metric", "last5"], REPO),
 ("08_e1_final", [PY, "analysis/analyze_e1.py", "--metric", "final"], REPO),
 ("09_e1_last5_with_supplement", [PY, "analysis/analyze_e1.py", "--metric", "last5", "--use-supplement"], REPO),
 ("10_e13_e15_last5", [PY, "analysis/analyze_e13_e15.py", "--metric", "last5"], REPO),
 ("11_e13_e15_final", [PY, "analysis/analyze_e13_e15.py", "--metric", "final"], REPO),
 ("12_exp06_sanity_exp05_logs", [PY, "adam-nv/analysis/analyze_exp06.py", "--log-dirs", str(x05[0]), str(x05[1]), "--tag", "_lr3e-3", "--opts", "adam", "adam_nv", "--ref", "adam"], REPO),
 ("13_exp07", [PY, "adam-nv/analysis/analyze_exp07.py", "--orig-dir", str(lab), "--new-dirs", *map(str, exp07), "--canary-dir", str(canary_dir), "--strict-seeds"], REPO),
 ("13b_exp08_dose_response", [PY, "analyze_exp08.py", "--orig-dir", str(lab), "--exp07-dirs", *map(str, exp07), "--c078-dirs", *map(str, b078), "--c070-dirs", *map(str, b070), "--strict-seeds"], REPO / "adam-nv/analysis"),
 ("13c_exp09_smooth_vs_distrust", [PY, "analyze_exp09.py", "--orig-dir", str(lab), "--exp07-dirs", *map(str, exp07), "--smooth-dirs", str(named(e9[0], "logs_exp09_A")), str(named(e9[1], "logs_exp09_B")), "--strict-seeds"], REPO / "adam-nv/analysis"),
 ("13d_exp10_ten_seeds_controls_seeds4to9_only", [PY, "analyze_exp10.py", "--log-dirs", str(named(x05[0], "logs_exp05")), str(named(x05[1], "logs_exp05")), str(named(e10[0], "logs_exp10_A")), str(named(e10[1], "logs_exp10_B")), str(named(e10c, "logs_exp10_C"))], REPO / "adam-nv/analysis"),
 ("13e_exp11_beta1_shuffle", [PY, "analyze_exp11.py", "--orig-dir", str(lab), "--exp07-dirs", *map(str, exp07), "--step-dirs", str(named(e11["A"], "logs_exp11_step_A")), str(named(e11["C"], "logs_exp11_step_C")), "--fixed-dirs", str(named(e11["B"], "logs_exp11_fixed_B")), str(named(e11["C"], "logs_exp11_fixed_C")), "--none-dirs", str(named(e11["C"], "logs_exp11_none_C")), "--strict-seeds"], REPO / "adam-nv/analysis"),
 ("13f_exp12_constant_beta1_above_0.9", [PY, "analyze_exp12.py", "--orig-dir", str(lab), "--exp07-dirs", *map(str, exp07), "--smooth-dirs", str(named(e9[0], "logs_exp09_A")), str(named(e9[1], "logs_exp09_B")), "--c092-dirs", str(named(e12[1], "logs_exp12_b092_1")), str(named(e12[2], "logs_exp12_b092_2")), "--c095-dirs", str(named(e12[3], "logs_exp12_b095_1")), str(named(e12[4], "logs_exp12_b095_2")), "--low-dirs-081", *map(str, exp07[:2]), "--low-dirs-078", *map(str, b078), "--low-dirs-070", *map(str, b070), "--strict-seeds"], REPO / "adam-nv/analysis"),
 ("13g_exp12_unbalanced_must_fail_EXPECTED_TO_FAIL", [PY, "analyze_exp12.py", "--orig-dir", str(lab), "--exp07-dirs", *map(str, exp07), "--smooth-dirs", str(named(e9[0], "logs_exp09_A")), str(named(e9[1], "logs_exp09_B")), "--c095-dirs", str(named(e12[3], "logs_exp12_b095_1")), "--strict-seeds"], REPO / "adam-nv/analysis"),
 ("13h_paper_draft_claim_check", [PY, "reports/_build_paper_check.py"], REPO),
 ("14_supplement_rebuild_is_idempotent", [PY, "supplements/e1/build_gono_seed1_supplement.py"], REPO),
 ("15_label_noise_suite_dry_run", [PY, "e2_noisy_gradient/src/run_label_noise_direction_suite.py", "--dry-run", "--seeds", "0", "1"], REPO),
 ("16_check_logs_hash_mode_warn_gono0", [PY, "check_logs.py", "--log-dir", "../logs", "--optimizers", "gono", "--seeds", "0", "--expect-torch", "", "--expect-torchvision", "", "--hash-mode", "warn"], REPO / "adam-nv/e1_convergence/src"),
 ("17_check_logs_strict_gono0_EXPECTED_TO_FAIL", [PY, "check_logs.py", "--log-dir", "../logs", "--optimizers", "gono", "--seeds", "0", "--expect-torch", "", "--expect-torchvision", ""], REPO / "adam-nv/e1_convergence/src"),
 ("17b_check_logs_gono_rerun_strict_hashes", [PY, "check_logs.py", "--log-dir", "../logs/gono_seed_1", "--optimizers", "gono", "--seeds", "1", "--expect-torch", "", "--expect-torchvision", ""], REPO / "adam-nv/e1_convergence/src"),
 ("18_check_logs_gono_seed1_EXPECTED_TO_FAIL", [PY, "check_logs.py", "--log-dir", "../logs", "--optimizers", "gono", "--seeds", "1", "--expect-torch", "", "--expect-torchvision", "", "--hash-mode", "off"], REPO / "adam-nv/e1_convergence/src"),
]
summary = []
for name, cmd, cwd in steps:
    t0 = time.time(); r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    out = (r.stdout + ("\n--- stderr ---\n" + r.stderr if r.stderr.strip() else ""))
    (LOGS / f"{name}.txt").write_text("$ " + " ".join(cmd) + f"\n(cwd={cwd.relative_to(REPO) if cwd != REPO else '.'})\n\n" + out, encoding="utf-8")
    expected_fail = name.endswith("EXPECTED_TO_FAIL")
    ok = (r.returncode != 0) if expected_fail else (r.returncode == 0)
    summary.append(dict(step=name, exit_code=r.returncode, expected_nonzero=expected_fail, as_expected=ok, seconds=round(time.time() - t0, 1)))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  exit={r.returncode}  {round(time.time()-t0,1)}s", flush=True)
(REPO / "reports" / "validation_summary.json").write_text(json.dumps(
    dict(python=sys.version.split()[0], steps=summary), indent=2), encoding="utf-8")
shutil.rmtree(tmp, ignore_errors=True)
sys.exit(0 if all(s["as_expected"] for s in summary) else 1)
