# Repository layout

The repository grew as two overlapping trees. Nothing was moved in the repair (moving files would
break the relative paths recorded in run metadata and runbooks). Use this map.

| Path | What it is |
|---|---|
| `e1_convergence/logs/` | E1 logs, seeds 0--3 for adam, adamw, adabelief, adam_nv (no source hashes recorded) |
| `e1_convergence/e1_results_table.csv` | Original E1 table (final-epoch metric, **population SD**); generator script is not in the repo |
| `adam-nv/e1_convergence/src/` | E1/EXP01--EXP06 training and launch code, GONO, `check_logs.py` |
| `adam-nv/e1_convergence/logs/` | GONO seeds 0--9 (seed 1 CSV ends at epoch 59), stdout captures |
| `adam-nv/e1_convergence/AdditionalsLogs/` | adam / adamw / adam_nv seeds 4--9 |
| `adam-nv/e1_convergence/logs/gono_seed_1/` | Author's complete rerun of GONO seed 1 (original CSV ends at epoch 59) |
| `adam-nv/e1_convergence/exp10_logs_{A,B}.zip` | EXP10 controls (adam_b083 / adam_b080, seeds 4--9) |
| `adam-nv/e2_noisy_gradient/logs/exp10_logs_C.zip` | EXP10 Adam / Adam-NV lr 3e-3, seeds 4--9 |
| `adam-nv/e2_noisy_gradient/logs/exp11_logs_*.zip`, `exp12_logs_*.zip` | EXP11 (shuffle) and EXP12 (constant beta1 > 0.9) logs |
| `adam-nv/benchmarks/` | Optimizer-overhead benchmarks and `timing_v2_results.zip` |
| `adam-nv/paperdraft/` | Paper draft PDF (numbers checked in `reports/PAPER_DRAFT_CHECK.md`) |
| `adam-nv/e1_convergence/exp03_*.zip`, `exp05_*.zip` | EXP03 (bias-correction ablation) and EXP05 (LR sensitivity) logs, **zipped only** |
| `adam-nv/e2_noisy_gradient/src/` | Newer E2 sources, hash-pinned by the EXP07/08 runbooks (`adam_nv.py` bcc7824b) |
| `adam-nv/e2_noisy_gradient/logs/*.zip` | EXP07 / EXP08 / EXP09 (and EXP10-C, 11, 12) logs, **zipped only** |
| `adam-nv/analysis/` | `analyze_exp06/07/08.py` |
| `adam-nv/tests/` | GONO, EXP06/07 tests (second test root) |
| `adam-nv/EXP0*_RUNBOOK.md` | Decision rules and run instructions per experiment |
| `adam-nv.zip` | Snapshot of `adam-nv/` (subset; 36 files differ from the directory by line endings only; since push `c534451` it also holds nested copies of the experiment zips, so scans see those runs twice) |
| `e2_noisy_gradient/src/` | v1--v8 training scripts, older `adam_nv.py` (2d6cd7df), suites |
| `e2_noisy_gradient/logs/` | E2/E13--E16 logs by experiment folder |
| `analysis/analyze_e13_e15.py` | E13--E15 pipeline |
| `analysis/analyze_e1.py`, `analysis/run_io.py` | **Added**: E1 pipeline, strict loader |
| `analysis/results/` | ORIGINAL derived table (stale, preserved) |
| `analysis/results_recomputed/` | **Added**: corrected tables |
| `provenance/` | **Added**: byte-exact archived sources + hash registry |
| `supplements/` | **Added**: recovered-record supplements (never edits originals) |
| `tools/` | **Added**: provenance verifier, completeness checker, test runner |
| `reports/` | **Added**: register, repair and validation reports, machine-readable results |
| `tests/` | Top-level tests, plus `test_repair_*.py` |

## Known layout hazards
* Same file name in different log folders: `Intermittent/` duplicates `dynamic_noise/` byte-for-byte;
  `controller_direction/` reuses names with different content. Analyses address files by full path.
* `part_b_resnet_v8.py`, `noise.py`, `noise_schedule.py`, `experiment_metadata.py`, `adam_nv.py` are
  hash-pinned; **do not edit them in place**. Add a new version and register it
  (`provenance/source_archive/<name>@<sha8>` then `python tools/provenance.py build`).
