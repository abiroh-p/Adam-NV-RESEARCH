# Repair report

Date of audit: 2026-10-25. Source: `https://github.com/abiroh-p/Adam-NV.git` (7 commits, 655 files at clone time).

## Principles applied
* Nothing was deleted. Only 9 original files were modified (listed below); every original log, zip,
  metadata file, historical source and the stale `analysis/results/` table is byte-for-byte unchanged.
* No result, epoch, seed, metric or hash was invented. Where the repository cannot answer, the issue is
  left open in `DISCREPANCY_REGISTER.md`.
* No historical run was re-associated with a newer source file. Hash-pinned sources (`adam_nv.py`,
  `part_b_resnet_v8.py`, `noise.py`, ...) were not edited.
* No training was rerun by the repair (no GPU, no CIFAR data). The author's later reruns/experiments (commit `ba29996`: GONO seed 1, EXP09, EXP10 partial, EXP11 code) were integrated and verified, not reproduced.

## Original files modified (9)
| File | Change |
|---|---|
| `README.md` | Repair notice; corrected source paths, test commands, status of E1/E14 items; metric wording; commands for new tools |
| `RESEARCH_STATUS.md` | Original text kept; dated addendum appended |
| `adam-nv/e1_convergence/src/check_logs.py` | Now compares recorded hashes with the registry (`--hash-mode strict|warn|off`); before, it only checked that the field existed |
| `adam-nv/analysis/analyze_exp07.py` | Added the shared `check_seed_balance()` (unbalanced-arm WARNING, `--strict-seeds` exits 2). Statistics untouched; output on balanced data is identical |
| `adam-nv/analysis/analyze_exp08.py`, `analyze_exp09.py`, `analyze_exp11.py`, `analyze_exp12.py` | Each calls `check_seed_balance()` and accepts `--strict-seeds`; nothing else changed |
| `analysis/analyze_e13_e15.py` | Strict loader; `--metric last5|final`; metric label in output; default output moved to `analysis/results_recomputed/`. Numeric output on the same logs is identical to the unmodified script (tested) |

## Added
* `provenance/` — 9 byte-exact historical source versions (`name@sha8`) and `source_hash_registry.json`
* `tools/` — `provenance.py` (registry builder / verifier, recurses into nested zips), `check_runs.py`
  (epoch completeness, duplicates, NaN, sidecars, collisions), `run_all_tests.py`, `validate_all.py`
* `analysis/run_io.py` (strict loader; `final_epoch_metric` and `last_k_epoch_mean_metric` are separate functions),
  `analysis/analyze_e1.py`, `analysis/results_recomputed/`
* `supplements/e1/` — recovered epoch-60 record for GONO seed 1 (+ builder), never merged into original CSV
* `e2_noisy_gradient/src/run_label_noise_direction_suite.py` — the missing ablation arm (not run)
* `tests/test_repair_optimizer.py` (19), `tests/test_repair_tools.py` (incl. documentation and `check_logs.py` tests), golden `.npy`
* `tools/provenance.py` also recognises folder-level `_code_hashes_*.txt` (`COVERED_BY_FOLDER_HASHES`); `tools/check_runs.py` also scans zips
* `analysis/analyze_e1.py` uses a verified complete rerun only to replace an incomplete original (refuses to merge on any mismatch)
* `.gitattributes`, `environment/ENVIRONMENT.md`, `docs/REPO_LAYOUT.md`, `docs/REPRODUCE.md`
* `reports/` — register (MD+CSV), manuscript discrepancies, this report, validation report and raw outputs

## Integrated after the first audit (author push `ba29996`, 2026-10-25)
* New files synced: EXP09/10/11 runbooks, analysis scripts, launchers, `adam_nv_shuffle.py`, `part_b_resnet_v8_shuffle.py`, EXP09/EXP10 log zips, `gono_seed_1/` rerun, updated `adam-nv.zip`, `test_exp11_shuffle.py`. No upstream file I had modified was touched.
* Upstream deleted `adam-nv/e1_convergence/src/train_resnet_ex.py`; the repaired tree keeps it (needed to verify EXP05 hashes).
* Archived `train_resnet.py@2ab3aa02`, now recorded by the GONO rerun and EXP10.
* Effect: LOG-01 and LOG-06 resolved by author-supplied data; DAT-01 (EXP10 incomplete) and DAT-03 (EXP11 no logs) opened.

## Integrated after the second author push (commits `8485e57` exp12, `c534451` 11_12)
* New data (all complete, hashes verified): `exp10_logs_C` (24 runs), EXP11 (36), EXP12 (16), `timing_v2_results.zip`, plus the paper draft PDF, EXP12 runbook/analysis, `analyze_exp11/12` scripts.
* Archived three newly-referenced sources (`adam_nv_shuffle.py@8e4df34c`, `part_b_resnet_v8_shuffle.py@b8612dff`, `benchmark_optimizer_overhead_v2.py@fb71196a`); registry now holds 13 versions. The four unrecoverable hashes are unchanged.
* The analysis scripts for EXP08/09/11/12 gave a pre-registered "OUTCOME" even when an arm lacked seeds; a shared `check_seed_balance()` plus `--strict-seeds` now covers EXP07-12.
* Paper drafts recomputed claim by claim (newer repo draft plus an older version you uploaded, stored beside it): 95 match, 6 not verifiable, 1 inconsistent (`reports/PAPER_DRAFT_CHECK.md`, `reports/UPLOADED_DRAFT_COMPARISON.md`). Neither draft is edited.
* Register: DAT-01 narrowed (Exp.6 logs still missing), DAT-03 resolved, DAT-04/05 and DOC-03 added.

## What changed in results
* E13--E15: 16 values in 4 comparisons (stale table used 3 of 4 seeds for intermittent beta2 = 0.95).
  Details: `reports/e13_e15_original_vs_corrected.csv`.
* E1: no committed E1 table reproduces the manuscript; recomputed tables are new files.
* No raw log was changed.
* E1 GONO is now 10 seeds (rerun for seed 1): last-5 mean 0.92445, unchanged from the supplement-only route.

## Decisions deliberately left to the authors
1. Whether last-5-epoch or final-epoch accuracy governs conclusions (E14 ramp-up 0.95 and E15 ramp-up 0.95 flip).
2. The E1 multiplicity family and any equivalence margin (not defined in any runbook).
3. Whether to rerun / trust `response_lag/...beta20.95_seed1.csv` (complete, but no sidecar).
4. Manuscript edits (docx untouched).
5. Whether the E1 analysis should use the GONO seed-1 rerun (default here; the supplement route gives identical numbers).
6. Supplying the Exp.6 logs (beta1 0.83/0.80, seeds 0-3) and the AdaBelief seed 4-9 logs, which the paper draft relies on.
7. How to resolve the draft's Table I vs Section VII mismatch (10 repeats vs 20 rounds).

## Not done
GPU reruns (AdaBelief seeds 4-9; the Exp.6 logs; any reproduction of historical training),
recovery of the four missing source versions, EXP03/EXP05 analysis scripts and `PROTOCOL_V2.md`,
comparison of E2/E7/E9/E11/E12 manuscript tables with logs.
