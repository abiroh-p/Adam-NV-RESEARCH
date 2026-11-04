#!/usr/bin/env python3
"""Run every test suite in the repository (the repo has two test roots).

  tests/            top-level suite (E2 optimizer, statistics, repair tooling)
  adam-nv/tests/    nested suite (GONO, EXP06/07 controls); must be run with adam-nv/ as cwd

Exit code is non-zero if either suite fails. Usage:  python tools/run_all_tests.py [-v]
"""
import subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
verbosity = ["-v"] if "-v" in sys.argv[1:] else []
suites = [("top-level tests/", REPO, ["-s", "tests"]), ("nested adam-nv/tests/", REPO / "adam-nv", ["-s", "tests"])]
rc = 0
for name, cwd, extra in suites:
    print(f"\n=== {name} (cwd={cwd.relative_to(REPO) if cwd != REPO else '.'}) ===", flush=True)
    r = subprocess.run([sys.executable, "-m", "unittest", "discover", *extra, *verbosity], cwd=cwd)
    rc |= r.returncode
sys.exit(1 if rc else 0)
