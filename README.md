# Adam-NV

Adam-NV is an experimental Adam variant whose per-parameter-element first
moment coefficient (`beta_1`) responds to a bounded gradient-noise heuristic.
The repository evaluates the mechanism on CIFAR models under constant and
non-stationary synthetic gradient noise.

> **Repair notice (2026-10-25).** This copy has been audited and repaired. Read
> `reports/DISCREPANCY_REGISTER.md`, `reports/REPAIR_REPORT.md` and
> `reports/VALIDATION_REPORT.md` first. Original evidence (logs, the stale
> `analysis/results/` table, historical sources) is preserved unmodified; corrected
> derived results are in `analysis/results_recomputed/`. Several issues remain
> unresolved (unrecoverable source hashes, AdaBelief seeds, the Exp. 6 control logs).
> Author pushes `ba29996` (Experiments 9--11) and `c534451` (Experiments 10C, 11, 12, timing v2,
> paper draft) were integrated after the first audit: the GONO seed-1 rerun, EXP09, EXP10-C,
> EXP11 and EXP12 logs are complete and hash-verified. The paper draft's numbers were
> recomputed from the logs: `reports/PAPER_DRAFT_CHECK.md` (an older draft you supplied is kept next to
> the repo's and compared in `reports/UPLOADED_DRAFT_COMPARISON.md`).

## Current evidence

The completed experiments show that Adam-NV's controller responds to changing
gradient statistics. They do **not** establish a consistent accuracy,
convergence, or stability advantage over Adam. The appropriate current framing
is a mechanism-validation and diagnostic-redesign study.

Legacy step-level response claims are invalid: the original diagnostic stepped
the cosine scheduler once per minibatch instead of once per epoch. Corrected
diagnostics use protocol v2 and are not yet rerun on the required seeds.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA, install the PyTorch build appropriate for the machine before the
remaining requirements if the default wheel is not suitable.

## Authoritative code

The repository holds two parallel trees (see `docs/REPO_LAYOUT.md`). Paths below are exact.

- E1 training/launch code: `adam-nv/e1_convergence/src/` (`train_resnet.py`, `run_e1_suite.py`,
  `gono.py`, `check_logs.py`). There is **no** `e1_convergence/src/` at the top level.
- E2/E13--E16 training: `e2_noisy_gradient/src/part_b_resnet_v8.py`
  (hash-pinned copy used for EXP07/08: `adam-nv/e2_noisy_gradient/src/part_b_resnet_v8.py`)
- Optimizer: `adam-nv/e2_noisy_gradient/src/adam_nv.py` (`bcc7824b…`, has the
  `bias_correction` ablation). The older `e2_noisy_gradient/src/adam_nv.py` (`2d6cd7df…`) is
  kept as a historical version; the two are bit-identical at default settings (tested).
- Statistics: `analysis/analyze_e1.py` (E1, new), `analysis/analyze_e13_e15.py` (E13--E15,
  `--metric last5|final`), `adam-nv/analysis/analyze_exp06.py`, `analyze_exp07.py`, `analyze_exp08.py`
- Strict log loader and metric definitions: `analysis/run_io.py`
- Provenance and integrity tools: `tools/provenance.py`, `tools/check_runs.py`
- Lag diagnostic: `e2_noisy_gradient/src/diagnose_response_lag.py`

Files named `part_b_resnet_v1.py` through `part_b_resnet_v7.py` are historical
snapshots. See `e2_noisy_gradient/src/LEGACY_SCRIPTS.md`.
`e2_noisy_gradient/src/analyze_e2.py` is an empty (0-byte) file in the original repository.

## Tests

```bash
python tools/run_all_tests.py          # runs BOTH test roots (tests/ and adam-nv/tests/)
python tools/provenance.py verify      # compare every recorded code hash with archived sources
python tools/check_runs.py             # epoch completeness / duplicates / NaN for every CSV log
```

Running only `python -m unittest discover -s tests` skips the nested GONO/EXP06/EXP07
suite. PyTorch-dependent numerical tests are skipped in analysis-only environments
without PyTorch.

## Regenerate tables

```bash
python analysis/analyze_e1.py --metric last5            # E1 (add --metric final, --use-supplement)
python analysis/analyze_e13_e15.py --metric last5       # E13-E15 (primary)
python analysis/analyze_e13_e15.py --metric final       # E13-E15 (sensitivity, separate file)
```
Outputs go to `analysis/results_recomputed/`. `analysis/results/` is the original (stale) table.

## Required completion runs

Run commands from the indicated `src` directory.

### 1. E1 baseline seeds (status updated)

Seeds 0--3 exist for Adam, AdamW, AdaBelief and Adam-NV; seeds 4--9 exist for Adam, AdamW and
Adam-NV; GONO seeds 0--9 exist (the original seed-1 CSV lacks epoch 60; the complete rerun in
`adam-nv/e1_convergence/logs/gono_seed_1/` is used by `analysis/analyze_e1.py` after a consistency
check, see `supplements/e1/` and the register). **AdaBelief still has only seeds 0--3**, although the
EXP10 runbook mentions an AdaBelief extension and the paper draft reports 10 AdaBelief seeds; those
logs are not in the repository. To balance it (GPU required, not
done in the repair):

```bash
cd adam-nv/e1_convergence/src
python run_e1_suite.py --epochs 60 --optimizers adabelief --seeds 4 5 6 7 8 9 \
  --skip-existing --deterministic
python check_logs.py --log-dir ../logs --optimizers adabelief --seeds 4 5 6 7 8 9
cd ../../..
python tools/provenance.py build   # after archiving any NEW source version in provenance/source_archive
python analysis/analyze_e1.py
```

### 2. E14 intermittent beta_2=0.95, seed 1 (status updated: nothing to rerun)

The original text said this CSV stops at epoch 32. The file in
`e2_noisy_gradient/logs/response_lag/` currently has 60 contiguous epochs, but it has **no
metadata sidecar**, so whether it is the original completed run cannot be verified from the
repository. `tools/check_runs.py` reports it as complete. Rerunning it would replace evidence
with a different (equally valid) draw only if the original is trusted; that is an author
decision, not a repair.

### 3. Rerun corrected multi-seed lag diagnostics

```bash
cd e2_noisy_gradient/src
python run_response_lag_v2_suite.py --epochs 60 --skip-existing --deterministic
```

Only `_lagdiag_v2` files containing `protocol_version=2` are valid. All older
files under `logs/diagnose` are explicitly invalidated.

### 4. Test controller direction

The original rule lowers `beta_1` when the heuristic increases. Because higher
momentum is the usual smoothing response to noise, the repository now includes
a matched reverse-direction ablation:

```bash
cd e2_noisy_gradient/src
python run_controller_direction_suite.py --epochs 60 --skip-existing --deterministic
```

### 5. Run CIFAR-100 generality testing

```bash
cd e2_noisy_gradient/src
python run_cifar100_generality_suite.py --epochs 60 --skip-existing --deterministic
```

This suite compares Adam and Adam-NV on clean CIFAR-100 and under intermittent
noise using four matched seeds.

### 6. Smooth-vs-distrust label-noise ablation (now present: Experiment 9)

The original repository had no smooth-response arm under label noise. Experiment 9
(`adam-nv/EXP09_RUNBOOK.md`, logs in `adam-nv/e2_noisy_gradient/logs/exp09_logs_{A,B}.zip`, 8 seeds)
supplies it; its hashes verify and the pre-registered analysis runs
(`adam-nv/analysis/analyze_exp09.py`). The repair's own
`run_label_noise_direction_suite.py` is superseded and kept only as a record.

### 6b. Regenerate label-noise evidence

`part_b_resnet_v8.py` now logs training accuracy on uncorrupted examples and,
for corrupted examples, accuracy against both clean and noisy labels. These
curves directly test the claimed memorization transition.

```bash
cd e2_noisy_gradient/src
python part_b_resnet_v8.py --optimizer adam_nv --schedule constant --sigma 0 \
  --label-noise 0.2 --label-diagnostic-interval 1 --seed 0 --epochs 60 \
  --log-dir ../logs/label_noise_v2 --deterministic
```

Repeat for Adam and seeds 1--3 before making comparative claims.

### 7. Benchmark overhead formally

```bash
cd ../..
python benchmarks/benchmark_optimizer_overhead.py --device cuda \
  --warmup-steps 50 --measured-steps 200 --repeats 10
```

The benchmark synchronizes CUDA, records hardware/software metadata, separates
optimizer-step and end-to-end time, and reports means, standard deviations, and
95% confidence intervals.

## Statistical policy

- Primary accuracy metric: mean validation accuracy over the LAST FIVE epochs (56--60).
  The final-epoch (epoch 60) accuracy is a different quantity, computed only with
  `--metric final` and reported separately. Conclusions can depend on the choice
  (see `reports/VALIDATION_REPORT.md`, E14/E15 rows).
- Match conditions by seed and use paired tests.
- Report mean paired differences and 95% confidence intervals.
- Treat the exact sign-flip result as a robustness check for small samples.
- Apply Holm correction within each declared experiment family.
- Do not infer that two conditions differ merely because one is significant and
  the other is not; use a direct contrast or interaction test.
- Phrase null findings as “no detectable advantage under the tested
  conditions,” not proof of equivalence or absence.

## Reproducibility notes

New training runs write a JSON metadata sidecar with arguments, command,
Python/PyTorch/torchvision versions, CUDA/GPU details, hostname, and RNG-stream
seeds. Gradient-noise runs with the same seed reuse matched Gaussian directions,
but the final perturbation tensors are not byte-identical: each optimizer's
direction is scaled by its own gradient norm.

See `RESEARCH_STATUS.md` (with the 2026-10-25 addendum) and `reports/DISCREPANCY_REGISTER.md`.
