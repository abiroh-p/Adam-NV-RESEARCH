# Exp 22 + 23: Frozen tensor-wise momentum and memorization trajectories

**Two experiments in one GPU batch (no duplicated evaluation training):**

- **Exp 22, causal control:** Independently train one Adam-NV pilot at 20% label noise, seed 100. Every 100 optimizer steps sample the *actual* beta1 tensor after the step; average each named tensor's per-element beta1 and then average across sampled steps. Freeze one coefficient per named tensor for **all training steps** of evaluation seed 0, 1 and 2; compare paired runs against ordinary Adam-NV at identical seed / CIFAR-10 noise mask / model / learning rate / cosine schedule. Adam is an additional comparator.
- **Exp 23, trajectory:** The *same* evaluation runs log validation accuracy each epoch and clean/corrupt/noisy-label training-subset accuracies at epochs 1, 5, 10, 20, 30, 45, 60. These are deterministic diagnostic-set evaluations with unaugmented inputs, not noisy training-batch accuracy. Show all trajectories, not just final numbers.

## Experimental interpretation

The frozen per-tensor coefficients are estimated **independently** of the three evaluation seeds, to avoid double-dipping. It is a **post-hoc time-averaged coefficient** control, not a matched coefficient at each time point. It tests whether a static per-tensor pattern is sufficient to reproduce much of the effect. It does **not** isolate causal temporal adaptation perfectly; any difference can also arise from time-dependent coefficient levels and their interaction with training. Compare final **seed-paired** differences, and do not treat epochs or individual examples as independent replications. Three seeds are exploratory, **not confirmatory**. In particular, the 95% CIs are unadjusted and should not be used for new claims of conclusive statistical significance.

## Prerequisites

Existing repository **working directory layout identical to Exp 16** at `/kaggle/working/Adam-NV` with:

- `e2_noisy_gradient/src/part_b_resnet_v8.py`
- `e2_noisy_gradient/src/adam_nv.py`, `noise.py`, `noise_schedule.py`, `experiment_metadata.py`
- Python, torch, torchvision, numpy, scipy already installed
- Two Kaggle GPUs recommended (can use one by `GPUS="0"`)

**This ZIP is an overlay** just like your Exp 16 example, not a standalone replacement of the existing repository. It doesn't alter any existing source file. To use a fresh GitHub clone, its `adam-nv/` subfolder corresponds to the `Adam-NV` working directory assumed by the commands below.

## Kaggle cells (copy/paste)

```python
!unzip -o /kaggle/working/exp22_23_frozen_beta_trajectories.zip -d /kaggle/working/Adam-NV
%cd /kaggle/working/Adam-NV
!python3 -m unittest discover -s tests -p 'test_exp22_23.py' -v
!cd e2_noisy_gradient/src && GPUS="0 1" EVAL_SEEDS="0 1 2" DRY_RUN=1 bash run_exp22_23.sh
!cd e2_noisy_gradient/src && GPUS="0 1" EVAL_SEEDS="0 1 2" DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp22_23 bash run_exp22_23.sh
!python3 analysis/analyze_exp22_23.py --dir e2_noisy_gradient/logs_exp22_23 --comparison frozen --reference adam_nv
!python3 analysis/analyze_exp22_23.py --dir e2_noisy_gradient/logs_exp22_23 --comparison adam --reference adam_nv
!cd e2_noisy_gradient && zip -r exp22_23_logs.zip logs_exp22_23
```

If uploading ZIP to Kaggle under `/kaggle/input/<dataset>/...`, adjust the first ZIP path accordingly.

The launcher runs a seed-100 pilot **first** (one GPU), then runs **nine paired evaluation jobs** (`adam_nv`, `frozen`, `adam` x seeds 0–2) distributed across the two GPUs. You can reduce to only the two core variants with `METHODS="adam_nv frozen"` to save compute; Adam's historical logs remain available in the repository. On relaunch, jobs with full CSV row counts are skipped, but **partial jobs restart** (no checkpoint resume). Use *the same* settings for all paired runs. Do not interrupt the pilot before its JSON map is saved. A full run may require considerable GPU time; no runtime guarantee is made.

To reproduce a fast CPU check **on a machine with the repository installed**, run individually with `--synthetic-data --smoke-test --epochs 2 --checkpoints 1 2 --sample-every 1`; synthetic mode does not produce meaningful memorization results.

## Expected outputs

- `pilot_ln0.2_seed100.json`: source of frozen per-tensor beta map, including sample count and range.
- `trajectory_<method>_ln0.2_seed<seed>.csv`: per-epoch train/test loss and accuracy, diagnostic memorization on checkpoints.
- `config_*.json`: run arguments, environment, pilot-map SHA-256 for each frozen run.
- `_code_hashes.txt`: source provenance, including original shared training sources.
- `stdout_*.txt`: individual run logs.
- `exp22_23_paired_trajectory.csv`: analysis-ready seed paired comparisons.

## What to send back

`exp22_23_logs.zip`, the unit-test output, and any errors verbatim. **Do not delete individual run logs or change seeds, hyperparameters, or the pilot map mid-run.** We can combine these results with your prior experiments and rewrite only scientifically justified claims.
