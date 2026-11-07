#!/usr/bin/env python3
"""Recompute every number typed in the paper's Experiment 14-21 sections from the raw logs stored in this repository.

What it does: unpacks the log archives of Experiments 7 and 14-21 (and the Label_Diagnostic baseline) into a scratch directory, extracts the paper's
verification scripts from ResearchPaperSupplements/adamnv_paper.zip, and runs verify_e14 ... verify_e21. Each verifier recomputes the numbers of its
experiment from the raw CSV/JSONL logs and compares them with the values typed in main.tex at the printed precision.

Usage:   python tools/reproduce_paper_numbers.py [--work DIR] [--keep]
Needs:   python 3.10+, numpy, pandas, scipy (no GPU, no PyTorch). Exit code 0 only if every verifier reports 0 mismatches.
"""
import argparse, os, re, shutil, subprocess, sys, tempfile, zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOGS = REPO / "adam-nv" / "e2_noisy_gradient" / "logs"
BASELINE = REPO / "e2_noisy_gradient" / "logs" / "Label_Diagnostic"
PAPER_ZIP = REPO / "ResearchPaperSupplements" / "adamnv_paper.zip"
ZIP_PATTERNS = [r"logs_exp07_[AB]\.zip", r"exp07_log_C\.zip", r"exp14_logs_C\.zip", r"logs_exp14_(onedim_A|rest_B)\.zip", r"exp15_logs_[1-4]\.zip", r"exp16\.zip",
                r"exp17_logs_[1-4]\.zip", r"exp18_logs_[1-4]\.zip", r"exp19_logs_[12]\.zip", r"exp20_logs_[12]\.zip", r"exp21_logs_[AB]\.zip"]
VERIFIERS = [("E14", "verify_e14_numbers.py"), ("E15", "verify_e15_numbers.py"), ("E16", "verify_e16_numbers.py"), ("E17", "verify_e17_numbers.py"),
             ("E18", "verify_e18_numbers.py"), ("E19", "verify_e19_numbers.py"), ("E20", "verify_e20_numbers.py"), ("E21", "verify_e21_numbers.py")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, default=None, help="scratch directory (default: a temporary directory)")
    ap.add_argument("--keep", action="store_true", help="keep the unpacked logs")
    a = ap.parse_args()
    work = a.work or Path(tempfile.mkdtemp(prefix="adamnv_reproduce_")); work.mkdir(parents=True, exist_ok=True)
    problems = []
    zips = sorted(p for p in LOGS.glob("*.zip") if any(re.fullmatch(pat, p.name) for pat in ZIP_PATTERNS))
    print(f"unpacking {len(zips)} log archives into {work} ...")
    for z in zips:
        with zipfile.ZipFile(z) as zf: zf.extractall(work)
    if BASELINE.is_dir(): shutil.copytree(BASELINE, work / "Label_Diagnostic", dirs_exist_ok=True)
    elif (BASELINE.parent / "Label_Diagnostic.zip").is_file():
        with zipfile.ZipFile(BASELINE.parent / "Label_Diagnostic.zip") as zf: zf.extractall(work)
    else: problems.append("Label_Diagnostic baseline logs not found")
    tools = work / "_verify_tools"; tools.mkdir(exist_ok=True)
    with zipfile.ZipFile(PAPER_ZIP) as zf:
        for m in zf.namelist():
            if "/tools/" in m and m.endswith(".py"): (tools / Path(m).name).write_bytes(zf.read(m))
    env = dict(os.environ, **{f"E{n}_ROOT": str(work) for n in range(14, 21)}, E21_ROOT=str(work / "e2_noisy_gradient"))
    results = []
    for name, script in VERIFIERS:
        if not (tools / script).is_file(): results.append((name, "script missing")); continue
        r = subprocess.run([sys.executable, str(tools / script)], env=env, capture_output=True, text=True)
        last = [l for l in r.stdout.splitlines() if "MISMATCHES" in l]
        results.append((name, (last[-1].strip() if last else f"FAILED TO RUN: {(r.stderr or r.stdout).strip().splitlines()[-1][:160] if (r.stderr or r.stdout).strip() else 'no output'}")))
    print()
    ok = not problems
    for name, res in results:
        good = res.endswith("MISMATCHES: 0"); ok &= good; print(f"  {name}: {res}  {'OK' if good else '<-- PROBLEM'}")
    for p in problems: print("  problem:", p)
    print("\nALL VERIFIERS PASS" if ok else "\nSOME VERIFIERS DID NOT PASS")
    if not a.keep and a.work is None: shutil.rmtree(work, ignore_errors=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
