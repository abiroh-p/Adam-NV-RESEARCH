# Validation report

All results below come from `python tools/validate_all.py`, which executes each command, records its exit code and
stores the full output in `reports/validation_logs/`. A step is PASS only if the process exit code matched what was expected.

## Environment (CPU only; not the experiment environment)
Python 3.12.3, torch 2.4.1+cu121 (CPU execution), torchvision 0.19.1, numpy 2.0.2, pandas 2.3.3 (inside the repo's declared `<3`),
scipy 1.17.1, matplotlib 3.10.8. The experiments ran on torch 2.10.0+cu128 / 2x T4 GPUs. See `environment/ENVIRONMENT.md`.

## Command results
| step | exit code | non-zero expected? | outcome |
|---|---|---|---|
| 01_imports | 0 | no | PASS |
| 02_unit_tests_all_roots | 0 | no | PASS |
| 03_provenance_verify | 0 | no | PASS |
| 04_provenance_verify_strict_EXPECTED_TO_FAIL | 1 | yes | PASS |
| 05_run_completeness | 0 | no | PASS |
| 06_run_completeness_strict | 0 | no | PASS |
| 07_e1_last5 | 0 | no | PASS |
| 08_e1_final | 0 | no | PASS |
| 09_e1_last5_with_supplement | 0 | no | PASS |
| 10_e13_e15_last5 | 0 | no | PASS |
| 11_e13_e15_final | 0 | no | PASS |
| 12_exp06_sanity_exp05_logs | 0 | no | PASS |
| 13_exp07 | 0 | no | PASS |
| 13b_exp08_dose_response | 0 | no | PASS |
| 13c_exp09_smooth_vs_distrust | 0 | no | PASS |
| 13d_exp10_ten_seeds_controls_seeds4to9_only | 0 | no | PASS |
| 13e_exp11_beta1_shuffle | 0 | no | PASS |
| 13f_exp12_constant_beta1_above_0.9 | 0 | no | PASS |
| 13g_exp12_unbalanced_must_fail_EXPECTED_TO_FAIL | 2 | yes | PASS |
| 13h_paper_draft_claim_check | 0 | no | PASS |
| 14_supplement_rebuild_is_idempotent | 0 | no | PASS |
| 15_label_noise_suite_dry_run | 0 | no | PASS |
| 16_check_logs_hash_mode_warn_gono0 | 0 | no | PASS |
| 17_check_logs_strict_gono0_EXPECTED_TO_FAIL | 1 | yes | PASS |
| 17b_check_logs_gono_rerun_strict_hashes | 0 | no | PASS |
| 18_check_logs_gono_seed1_EXPECTED_TO_FAIL | 1 | yes | PASS |

"Expected to fail" steps are negative controls: the strict provenance check must fail while unverified hashes exist, the strict
completeness check must fail while GONO seed 1 is incomplete, `check_logs.py` must fail on unarchived hashes and on a 59-epoch log.

## Unit tests — PASSED
* Top-level `tests/`: **68** tests, OK (11 pre-existing, 19 optimizer, 38 tooling, statistics, seed-guard and documentation tests).
* Nested `adam-nv/tests/`: **26** tests, OK (GONO / EXP06 / EXP07 suites, which were never part of the documented test command, plus the 7 EXP11 shuffle tests from author push `ba29996`).
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
claims scanned: 1260
  COVERED_BY_FOLDER_HASHES     148
  NO_HASH_RECORDED             104
  UNVERIFIED_NOT_ARCHIVED      208
  VERIFIED_EXACT               752
  VERIFIED_EXACT_RENAMED       48
```
Claims count each (run, file) hash, including copies of the same run inside `adam-nv.zip`. `COVERED_BY_FOLDER_HASHES` = the run's metadata has no hash but its launch folder's `_code_hashes_*.txt` verifies completely (EXP07/08/09); it is weaker than a per-run hash. 96 runs have no hash of either kind. Unique unverified items:
28 E1 runs x {adam_nv.py 96971286, gono.py 82665f3c, train_resnet.py 7b5134fd} and 20 EXP03 runs x train_resnet.py 5ac384a1.

## Run completeness (`tools/check_runs.py`)
```
csv files examined: 735
  EXCLUDED_AUXILIARY                 80
  KNOWN_PARTIAL_SUPERSEDED           3
  OK                                 652
  per-epoch runs without a metadata sidecar: 209
  [KNOWN_PARTIAL_SUPERSEDED] zip:adam-nv.zip>adam-nv/e1_convergence/logs/gono_seed1.csv  adam-nv/e1_convergence/logs/gono_seed_1/gono_seed1.csv
  [KNOWN_PARTIAL_SUPERSEDED] e2_noisy_gradient/logs/dynamic_noise/adam_nv_scheduleintermittent_sigma2.0_beta20.95_seed3.csv  e2_noisy_gradient/logs/response_lag/adam_nv_scheduleintermittent_sigma2.0_beta20.95_seed3.csv
  [KNOWN_PARTIAL_SUPERSEDED] adam-nv/e1_convergence/logs/gono_seed1.csv  adam-nv/e1_convergence/logs/gono_seed_1/gono_seed1.csv
```
The scan also reads every `.zip`, recursively (EXP03/05/07/08/09/10 logs exist only as zips). Both copies of the truncated GONO seed-1 CSV are `KNOWN_PARTIAL_SUPERSEDED` by the complete rerun (`reports/known_superseded_logs.json`, matched by SHA-256), which is why the strict run passes (step 06).

## Reproduced documented numbers (extracted from the recorded step outputs)
* EXP06 runbook claim "+0.173 pt, p = 0.031, 4/4 seeds" (final-epoch metric) — step 12 printed: `adam_nv      n=4 mean diff=+0.173  95% CI [+0.030, +0.315]  p=0.031  seeds higher: 4/4` — **reproduced**.
* EXP07 (8 seeds, pre-registered rule; step 13): `Adam-NV - Adam (beta1=0.9): +3.909 pp  95% CI [+3.111, +4.706]  p=0.0000  p_Holm=0.0000  sign-flip p=0.0078  higher in 8/8 seeds`; `Adam-NV - Adam constant beta1: +3.274 pp  95% CI [+2.392, +4.155]  p=0.0001  p_Holm=0.0001  sign-flip p=0.0078  higher in 8/8 seeds`; decision line: `OUTCOME: ADAPTATION BEYOND BETA1 (D = +3.27 CI above 0, F < 0.25*M)`. The EXP08 runbook quotes +3.91 pp (8/8) and +0.64 pp for the constant-beta1 control — **reproduced**. Canary: `adam: max abs difference over 10 columns = 0.000e+00 (IDENTICAL)`; `adam_nv: max abs difference over 15 columns = 0.000e+00 (IDENTICAL)`.
* EXP08 (step 13b): `Reference effect M = Adam-NV - Adam: +3.91 pp [+3.11, +4.71] (8/8 seeds)`; `Adam-NV - constant 0.78: +3.143 pp  95% CI [+2.155, +4.131]  p=0.0001  p_Holm=0.0005  sign-flip p=0.0078  higher in 8/8`; `Adam-NV - constant 0.70: +3.080 pp  95% CI [+1.924, +4.235]  p=0.0004  p_Holm=0.0012  sign-flip p=0.0078  higher in 8/8`; decision line: `OUTCOME: ADAPTATION BEYOND ANY CONSTANT BETA1 (both controls < 0.25*M, D CIs above 0)`.
* EXP09 (step 13c): `M = +3.91 pp, F = -1.47 pp, D = +5.38 pp (CI [+4.55, +6.21])`; decision line: `OUTCOME: DISTRUST DIRECTION DRIVES THE EFFECT (F < 0.25*M, D CI above 0)`.
* EXP10 (step 13d, Account C now present; controls only at seeds 4-9 because the Exp.6 logs are absent): `Adam-NV - Adam: +0.172  95% CI [+0.049, +0.294]  p=0.0114  p_Holm=0.0572  sign-flip p=0.0176  higher in 9/10`; decision line: `OUTCOME: INCONCLUSIVE (see the half-widths above; more seeds or a tighter design needed)`.
* EXP11 (step 13e): `shuffle/step: F = +2.40 pp, D = +1.51 pp [+0.93, +2.08]  -> PARTIAL`; `shuffle/fixed: F = +2.81 pp, D = +1.10 pp [+0.39, +1.81]  -> PARTIAL`; decision line: `OUTCOME: INCONCLUSIVE; report all contrasts and extend seeds.`.
* EXP12 (step 13f): `constant 0.92: F = +0.44 pp, D = -1.91 pp [-3.52, -0.30] -> ADAPTATION ADDS`; `constant 0.95: F = -1.03 pp, D = -0.45 pp [-1.32, +0.43] -> REPRODUCES`; decision line: `OUTCOME: PARTIAL / INCONCLUSIVE; read the dose-response and extend seeds.`. Step 13g is a negative control: with one arm at n=4 the script exits 2.
* Paper draft check (step 13h): {'MATCH': 95, 'NOT VERIFIABLE': 6, 'DIFFERS': 1} - see `reports/PAPER_DRAFT_CHECK.md`.
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
