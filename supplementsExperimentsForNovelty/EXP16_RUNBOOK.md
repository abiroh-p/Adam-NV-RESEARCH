# Experiment 16: noise-proxy diagnostic (CIFAR-10, ResNet-18)

**Question:** Adam-NV's controller uses a moment-based proxy, s = max(v − m², 0) / (v + λ|m| + ε), as its noise score. The paper calls it a proxy, notes that it also rises when gradient magnitudes shrink (v has a much longer memory than m), and has not validated it against a direct noise measurement. Does s track the actual minibatch gradient noise of each element, or mostly the gradient's scale?

## What is measured
`noise_proxy_diagnostic.py` trains a ResNet-18 with Adam-NV (default settings, lr 1e-3, 60 epochs, cosine schedule, batch 128, the same data pipeline as the label-noise experiments, imported from `part_b_resnet_v8.py`). After epochs 1, 5, 10, 20, 30, 45 and 60 it measures, at the current (fixed) weights and without any optimizer step:
- K = 16 minibatch gradients (BatchNorm in training mode; the running statistics are restored afterwards);
- per element, the **direct noise fraction** q = unbiased variance / mean square of those gradients (the direct counterpart of s: the fraction of the gradient's second moment that is variance across minibatches);
- per element, the proxy s computed from the optimizer's own m and v, the controller state ρ, and the **scale ratio** sqrt(mean square) / sqrt(v);
- Spearman rank correlations over a random sample of 500,000 elements (and by tensor rank, and the median over convolution tensors): s–q, ρ–q, s–scale and q–scale.

## Design (4 runs)

| Condition | Seeds |
|---|---|
| Clean labels | 0, 1 |
| 20% symmetric label noise | 0, 1 |

Two seeds are descriptive only: the correlations are over millions of elements within a run, which are not independent observations, so no between-run inference is made.

## Pre-registered rule (before any Exp. 16 run)
For each condition let r = the mean over seeds and over checkpoints from epoch 5 on of the pooled Spearman correlation between s and q.
- r ≥ 0.5: the proxy **tracks** direct gradient noise.
- 0.2 ≤ r < 0.5: it tracks it **weakly**.
- r < 0.2: it **does not** track it.
Confound check: let c be the same mean of the Spearman correlation between s and the scale ratio. If |c| > r, the proxy is more related to the gradient-scale ratio than to direct noise (**scale-confounded**).
Descriptive additions, with no threshold: mean s, mean q and the correlations by tensor rank.

## Verified, and not verified
Verified (CPU PyTorch 2.14.1): 5 unit tests pass (Spearman against scipy including ties; the direct noise fraction recovers the known variance fraction of synthetic gradients, within 0.03; it is 0 for identical gradients and bounded by 1; the proxy equals the formula in the optimizer; `collect_diagnostics` leaves the model weights, the BatchNorm statistics and the optimizer state unchanged). The script ran end to end on a real ResNet-18 in synthetic smoke mode and wrote the JSON lines, the training CSV and the configuration file; the launcher was dry-run; the analysis was tested on synthetic diagnostic files (its printed numbers mean nothing).
NOT verified: any real-data run, the actual correlations, the running time (my estimate: about 55 to 65 minutes per run), and that 16 minibatches give a stable variance estimate per element (the unbiased estimate from K = 16 is noisy for each element, but the pooled rank correlation over 500,000 elements averages this out; a K = 32 rerun would check it).

## Before launching: hashes
Each launch writes `_code_hashes_*.txt`. Expect (first 8 characters) `part_b_resnet_v8.py` 242e6a77, `adam_nv.py` bcc7824b, `noise_schedule.py` 90e11e2c, `noise.py` eee61201, `experiment_metadata.py` ab020aaf. The new file is `noise_proxy_diagnostic.py`.

## Account plan (one account, about 2.2 h)

| GPU 0 | GPU 1 |
|---|---|
| `0.0:0` then `0.2:0` | `0.0:1` then `0.2:1` |

## Steps (same setup as Exp. 7–15)

```bash
!unzip -o exp16_proxy_diagnostic.zip -d /kaggle/working/Adam-NV
%cd /kaggle/working/Adam-NV/e2_noisy_gradient/src
!cd /kaggle/working/Adam-NV && python3 -m unittest discover -s tests -v 2>&1 | tail -8     # must end with OK

!JOBS_GPU0="0.0:0 0.2:0" JOBS_GPU1="0.0:1 0.2:1" DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp16 bash run_exp16_proxy_diagnostic.sh
!cat /kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp16/_code_hashes_*.txt
!cd /kaggle/working/Adam-NV/e2_noisy_gradient && zip -r exp16_logs.zip logs_exp16
```
If a session dies, run the same cell again; complete jobs are skipped. The launcher prints a verification table at the end (training epochs and the 7 diagnostic records per run).

## Analysis

```bash
python3 analysis/analyze_exp16.py --dirs logs_exp16
```

## Send back
`exp16_logs.zip`, the unit-test output, the verification table, the `_code_hashes` file, and any error verbatim. Do not rename or delete any log file.

## Do not
Change any argument or edit any file.
