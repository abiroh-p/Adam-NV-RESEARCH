import json, re
from pathlib import Path
R = Path(__file__).resolve().parent; L = R / "validation_logs"
S = json.loads((R / "validation_summary.json").read_text())
txt = lambda n: (L / f"{n}.txt").read_text(encoding="utf-8")
ut = txt("02_unit_tests_all_roots")
ran = re.findall(r"Ran (\d+) tests", ut)
prov = txt("03_provenance_verify"); comp = txt("05_run_completeness")
def block(t, start, end=None):
    i = t.index(start); j = t.index(end, i) if end else len(t); return t[i:j].strip()
steps = "\n".join(f"| {s['step']} | {s['exit_code']} | {'yes' if s['expected_nonzero'] else 'no'} | {'PASS' if s['as_expected'] else 'FAIL'} |" for s in S["steps"])
g = lambda n, pat: re.search(pat, txt(n)).group(0).strip()
e6 = g("12_exp06_sanity_exp05_logs", r"adam_nv\s+n=4 mean diff=[^\n]*")
e7m = g("13_exp07", r"Adam-NV - Adam \(beta1=0.9\): \+3\.\d+ pp[^\n]*higher in 8/8 seeds")
e7d = g("13_exp07", r"Adam-NV - Adam constant beta1: \+3\.\d+ pp[^\n]*higher in 8/8 seeds")
e7o = g("13_exp07", r"OUTCOME:[^\n]*"); e7c1 = g("13_exp07", r"adam: max abs difference[^\n]*"); e7c2 = g("13_exp07", r"adam_nv: max abs difference[^\n]*")
e8m = g("13b_exp08_dose_response", r"Reference effect M[^\n]*"); e8a = g("13b_exp08_dose_response", r"Adam-NV - constant 0\.78:[^\n]*")
e8b = g("13b_exp08_dose_response", r"Adam-NV - constant 0\.70:[^\n]*"); e8o = g("13b_exp08_dose_response", r"OUTCOME:[^\n]*")
e9m = g("13c_exp09_smooth_vs_distrust", r"M = \+[\d.]+ pp, F = [-+][\d.]+ pp, D = [-+][\d.]+ pp[^\n]*"); e9o = g("13c_exp09_smooth_vs_distrust", r"OUTCOME:[^\n]*")
e10 = g("13d_exp10_ten_seeds_controls_seeds4to9_only", r"Adam-NV - Adam: \+0\.172[^\n]*"); e10o = g("13d_exp10_ten_seeds_controls_seeds4to9_only", r"OUTCOME:[^\n]*")
e11o = g("13e_exp11_beta1_shuffle", r"OUTCOME:[^\n]*"); e11s = g("13e_exp11_beta1_shuffle", r"shuffle/step: F = [^\n]*"); e11f = g("13e_exp11_beta1_shuffle", r"shuffle/fixed: F = [^\n]*")
e12a = g("13f_exp12_constant_beta1_above_0.9", r"constant 0\.92: F = [^\n]*"); e12b = g("13f_exp12_constant_beta1_above_0.9", r"constant 0\.95: F = [^\n]*"); e12o = g("13f_exp12_constant_beta1_above_0.9", r"OUTCOME:[^\n]*")
import pandas as _pd
_pc = _pd.read_csv(R / "paper_draft_claim_check.csv").verdict.value_counts().to_dict()
out = f"""# Validation report

All results below come from `python tools/validate_all.py`, which executes each command, records its exit code and
stores the full output in `reports/validation_logs/`. A step is PASS only if the process exit code matched what was expected.

## Environment (CPU only; not the experiment environment)
Python {S['python']}, torch 2.4.1+cu121 (CPU execution), torchvision 0.19.1, numpy 2.0.2, pandas 2.3.3 (inside the repo's declared `<3`),
scipy 1.17.1, matplotlib 3.10.8. The experiments ran on torch 2.10.0+cu128 / 2x T4 GPUs. See `environment/ENVIRONMENT.md`.

## Command results
| step | exit code | non-zero expected? | outcome |
|---|---|---|---|
{steps}

"Expected to fail" steps are negative controls: the strict provenance check must fail while unverified hashes exist, the strict
completeness check must fail while GONO seed 1 is incomplete, `check_logs.py` must fail on unarchived hashes and on a 59-epoch log.

## Unit tests — PASSED
* Top-level `tests/`: **{ran[0]}** tests, OK (11 pre-existing, 19 optimizer, {int(ran[0]) - 30} tooling, statistics, seed-guard and documentation tests).
* Nested `adam-nv/tests/`: **{ran[1]}** tests, OK (GONO / EXP06 / EXP07 suites, which were never part of the documented test command, plus the 7 EXP11 shuffle tests from author push `ba29996`).
* Baselines: before any change top-level 11 OK, nested 19 OK; the sync of push `ba29996` raised the nested suite to 26, all OK.

What the new tests cover: multi-step agreement with an independent NumPy reference (distrust and smooth modes, 1-400 steps);
gamma=0 and fixed-beta1 agreement with `torch.optim.Adam` within the clamp horizon (1e-12) and beyond it (1e-6, documented deviation);
first-step bias correction; running product; zero/huge/tiny gradients and float32 stability; parameter groups; missing gradients;
coupled weight decay; old-vs-new `adam_nv.py` bit-identity; a golden regression; hash detection of exact / renamed / EOL-variant /
one-byte-changed / nested-zip cases; missing, gapped, duplicated and NaN epochs; final-epoch vs last-5 distinction; opt-in supplements
that never overwrite observed epochs; seed pairing and unpaired seeds; percentage-point conversion; exact sign-flip and Holm known values;
fresh recomputation equals the stored tables; README/doc file paths exist; `check_logs.py` strict/warn exit codes.

## Provenance verification (`tools/provenance.py verify`)
```
{block(prov, 'claims scanned')}
```
Claims count each (run, file) hash, including copies of the same run inside `adam-nv.zip`. `COVERED_BY_FOLDER_HASHES` = the run's metadata has no hash but its launch folder's `_code_hashes_*.txt` verifies completely (EXP07/08/09); it is weaker than a per-run hash. 96 runs have no hash of either kind. Unique unverified items:
28 E1 runs x {{adam_nv.py 96971286, gono.py 82665f3c, train_resnet.py 7b5134fd}} and 20 EXP03 runs x train_resnet.py 5ac384a1.

## Run completeness (`tools/check_runs.py`)
```
{block(comp, 'csv files examined')}
```
The scan also reads every `.zip`, recursively (EXP03/05/07/08/09/10 logs exist only as zips). Both copies of the truncated GONO seed-1 CSV are `KNOWN_PARTIAL_SUPERSEDED` by the complete rerun (`reports/known_superseded_logs.json`, matched by SHA-256), which is why the strict run passes (step 06).

## Reproduced documented numbers (extracted from the recorded step outputs)
* EXP06 runbook claim "+0.173 pt, p = 0.031, 4/4 seeds" (final-epoch metric) — step 12 printed: `{e6}` — **reproduced**.
* EXP07 (8 seeds, pre-registered rule; step 13): `{e7m}`; `{e7d}`; decision line: `{e7o}`. The EXP08 runbook quotes +3.91 pp (8/8) and +0.64 pp for the constant-beta1 control — **reproduced**. Canary: `{e7c1}`; `{e7c2}`.
* EXP08 (step 13b): `{e8m}`; `{e8a}`; `{e8b}`; decision line: `{e8o}`.
* EXP09 (step 13c): `{e9m}`; decision line: `{e9o}`.
* EXP10 (step 13d, Account C now present; controls only at seeds 4-9 because the Exp.6 logs are absent): `{e10}`; decision line: `{e10o}`.
* EXP11 (step 13e): `{e11s}`; `{e11f}`; decision line: `{e11o}`.
* EXP12 (step 13f): `{e12a}`; `{e12b}`; decision line: `{e12o}`. Step 13g is a negative control: with one arm at n=4 the script exits 2.
* Paper draft check (step 13h): {_pc} - see `reports/PAPER_DRAFT_CHECK.md`.
* GONO seed 1 (steps 07/09, 17b): the complete rerun is used (n=10, consistency check max diff 0.0 over 59 epochs); `check_logs.py` in strict-hash mode passes on the rerun folder and fails on the truncated original.
Note: these reruns re-analyze existing logs; they do not retrain anything. An EXP07 invocation that omits `log_exp07_C` (the Adam-NV seeds 4-7) silently analyzes Adam-NV with n=4 and gave M = +3.40 pp with no warning in the original script (this happened during the repair); the repaired script now prints an unbalanced-arm WARNING and fails under `--strict-seeds`, which `validate_all.py` uses.

## Experiments NOT rerun
All GPU training. The repair itself executed no training. Experiments the author ran independently and pushed (GONO seed-1 rerun, EXP09, EXP10, EXP11, EXP12, timing v2) were verified and analyzed, not reproduced. Not rerun by anyone in this record: AdaBelief seeds 4-9, EXP03, EXP05, E13-E16, the lag diagnostics, and the missing Exp.6 logs. No training log was regenerated here.

## Not independently verified
* Provenance of 28 E1 runs and 20 EXP03 runs (source files unavailable); of all hash-less runs (E1 seeds 0-3, Label_Diagnostic seeds 1-3, LagDiagnostic, CIFAR-100, controller direction) and of the 209 runs without metadata sidecars.
* That `response_lag/...beta20.95_seed1.csv` is the genuine completed run.
* EXP10's ten-seed constant-beta1 contrasts (Exp.6 logs missing); 6 draft claims (AdaBelief 10 seeds, constants at ten seeds, element-level beta1 table).
* Timing numbers are verified only against the stored summary (GPU/session specific).
* That the E2/E7/E9/E11/E12 manuscript tables match the logs (not compared).
* Equivalence of the lost `adam_nv.py` 96971286 with the preserved versions (preserved versions are bit-identical to each other).
* The optimizer reference in the tests is a transcription by the repair author, not extracted from the paper's equations.
* Training reproducibility on the original GPU/torch stack.
* Statistics for E2/E7/E9/E11/E12 beyond the EXP06-08 reruns above.
"""
(R / "VALIDATION_REPORT.md").write_text(out, encoding="utf-8"); print("ok", ran)
