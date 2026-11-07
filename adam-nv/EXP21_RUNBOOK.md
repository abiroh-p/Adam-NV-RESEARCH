# Experiment 21: noise-proxy diagnostic v2 (run book; written before any of these runs)

Why. Exp. 16 could not decide whether Adam-NV's score tracks gradient noise: the direct estimate (K = 16) was saturated under label noise
(mean 0.96-0.97, the value of pure noise), the registered thresholds ignored the estimate's ceiling, only pooled correlations (dominated by differences between
tensors) were registered, the absolute gradient scale was not logged, per-tensor values were not stored, and there were two seeds. This experiment fixes those.

| Exp. 16 weakness | Exp. 21 |
|---|---|
| ceiling of the direct estimate unknown (simulated only) | split-half reliability of the direct estimate in every tensor (Spearman-Brown), and reliability-corrected correlations |
| clamped estimate saturates and ties | unclamped noise fraction `qu` for ranking; clamp shares logged; K = 64 |
| absolute scale not logged | per-element rms, per-tensor means, `r(s, rms)` and the partial correlation given rms |
| pooled over elements hides a within-tensor null | within-tensor correlation in EVERY tensor; primary statistic = median over the 20 convolution tensors; between-tensor correlation reported separately |
| lambda*|m| mixes units | the same score with lambda = 0 (`s0`), computed from the same optimizer state (diagnostic only) |
| time scales differ (EMA vs fixed weights) | extra estimate from the last 64 training steps (read-only), exploratory |
| aggregates only | a 100,000-element sample per checkpoint saved (`.npz`) |
| same minibatch index sets in every run | per-run batch generator (`--legacy-batches` reproduces Exp. 16 for regression tests only) |
| 2 seeds | 5 seeds; the seed is the unit of replication |

Training is unchanged (Adam-NV defaults, lr 1e-3, 60 epochs, batch 128, `--deterministic`), so the 20%-noise runs must reproduce the validation accuracy of the earlier Adam-NV runs of the same seed at every epoch; the checker verifies this (seeds 0-7 exist from Exp. 7).

Runs: label noise 0.0 and 0.2, seeds 0-4 = 10 runs, about 65 min each (diagnostics add a few minutes). Two calls: A (6 runs, about 3.3 h on 2 x T4), B (4 runs, about 2.2 h). Disk: about 2 MB per checkpoint sample file, about 150 MB in total.

## 0. Setup

Unzip into the repository root `/kaggle/working/Adam-NV` (adds `noise_proxy_diagnostic_v2.py`, `run_exp21_proxy_v2.sh`, `check_exp21_logs.py`, `analysis/analyze_exp21.py`, `tests/`). Then:

```bash
cd /kaggle/working/Adam-NV
python3 -m unittest discover -s tests -v          # all tests must print OK; send me the output
```

Do not change any other file. Environment: python 3.12.13, torch 2.10.0+cu128, torchvision 0.25.0+cu128, 2 x Tesla T4 (the launcher checks).

## 1. Commands (every call needs its OWN LOG_DIR)

```bash
cd /kaggle/working/Adam-NV/e2_noisy_gradient/src

# Call A: seeds 0-2, both conditions
JOBS_GPU0="0.0:0 0.2:0 0.0:1" JOBS_GPU1="0.2:1 0.0:2 0.2:2" \
DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp21_A \
bash run_exp21_proxy_v2.sh

# Call B: seeds 3-4, both conditions
JOBS_GPU0="0.0:3 0.2:3" JOBS_GPU1="0.0:4 0.2:4" \
DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp21_B \
bash run_exp21_proxy_v2.sh
```

Add `DRY_RUN=1` in front to print the commands only. If your earlier Adam-NV logs are not in `/kaggle/working/Adam-NV/{Label_Diagnostic,logs_exp07_A,...}`, set `REF_DIRS="dir1 dir2 ..."`; the checker prints a note and skips the training-equivalence check if it finds no reference.
Expected last lines of each call: `N/N jobs verified`.

## 2. What to send back

Each `LOG_DIR` zipped completely (proxy2_train/diag/config files, the `.npz` sample files, stdout, `_status_*`, `_code_hashes_*`), the last lines of the call's output (checker table), and once the `unittest` output.

## 3. Analysis (I run this when the logs arrive)

```bash
cd /kaggle/working/Adam-NV/analysis
python3 analyze_exp21.py --log-dirs ../e2_noisy_gradient/logs_exp21_A ../e2_noisy_gradient/logs_exp21_B
```

## 4. Pre-registered decision rule (per label-noise condition; seeds are the unit; t-based 95% CI)

Per run and checkpoint, over the 20 convolution tensors, the median of: `R` = within-tensor Spearman(s, qu); `rel` = split-half reliability of qu (Spearman-Brown);
`R* = R / sqrt(rel)` if rel >= 0.2 (else undefined); `R0*` the same for the lambda = 0 score; `Rrms` = Spearman(s, rms). Per seed: mean over the primary checkpoints (epochs 30, 45, 60).
- UNRESOLVED: mean reliability < 0.2. No verdict; the direct estimate cannot rank elements within tensors.
- TRACKS: lower 95% limit of mean R* >= 0.3.
- DOES NOT TRACK: upper 95% limit of mean R* < 0.1.
- INCONCLUSIVE: otherwise.
- SCALE-DOMINATED: mean |Rrms| > mean R*. Lambda effect: paired difference R0* - R* over seeds with its CI (above 0: the lambda*|m| term lowers tracking).
Descriptive only: all other checkpoints, between-tensor correlations (62 tensors), the pooled statistic of Exp. 16, the stream estimate, the other tensor ranks. Exploratory: anything not listed here.

Pipeline controls (unit tests, synthetic): a score that is a monotone function of the true noise fraction gives raw r = 0.76, reliability 0.585, corrected R* = 0.99 (TRACKS); a random score gives 0.011, 0.581, 0.015 (DOES NOT TRACK). The rule is also tested to return UNRESOLVED when reliability is 0.10.

## 5. How each outcome changes the paper (decided now)

- TRACKS (clean and/or noisy): the paper's "no within-tensor relation" sentence is replaced by the measured corrected correlation with its interval; if the lambda effect is above 0 the units point (lambda*|m|) is reported as the cause of weak tracking.
- DOES NOT TRACK: the Exp. 16 conclusion stands with a measurable ceiling now ("the score does not track measured gradient noise within tensors, with a resolvable direct estimate, in five seeds").
- INCONCLUSIVE or UNRESOLVED: the paper keeps "the proxy remains unvalidated", states the reliability that was reached, and Exp. 16 remains descriptive.
In every case Section V-G is rewritten around the Exp. 21 statistics and Exp. 16 is reduced to one sentence.

## Deviations log

(none yet)
