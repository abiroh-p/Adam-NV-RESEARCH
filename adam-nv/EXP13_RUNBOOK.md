# Experiment 13: tensor-mean β₁ control under 20% label noise (CIFAR-10, ResNet-18)

**Question:** Exp. 11 permuted Adam-NV's β₁ *within* each parameter tensor and kept 61–72% of the extra memorization (Adam-NV minus Adam: +3.91 pp). A within-tensor permutation preserves each tensor's own β₁ mean and distribution, so **tensor-level (layer-wise) adaptation** could still explain the part that survives. This experiment removes all within-tensor spread instead: at every step each element of a tensor receives that tensor's current mean β₁. If the effect survives, it is layer-level adaptation; if not, the within-tensor structure matters.

## The intervention
AdamNVTensorMean (a subclass of the Exp. 11 `AdamNVShuffle`, overriding one hook; `adam_nv.py`, `adam_nv_shuffle.py`, `part_b_resnet_v8.py` are not modified):
- `tensor_mean`: after the controller computes its per-element β₁ for a tensor, every element is set to the tensor's mean β₁. The running product P_t uses the applied values, so the bias correction stays exact.
- `none`: no change (bit-identical to Adam-NV). Used as a regression check and for **per-tensor β₁ statistics**: one record per epoch lists, for each of the 62 parameter tensors, its shape and the mean, minimum and maximum of the controller's own β₁.

## Prediction to check (stated before the runs)
ResNet-18 has 62 parameter tensors: 41 one-dimensional (normalization parameters and one bias, 9,610 elements in total), 1 two-dimensional (the classifier weight, 5,120 elements) and 20 four-dimensional convolution tensors (11,159,232 elements). The logged per-tensor mean (about 0.78) gives each tensor equal weight, whereas the element-weighted mean is about 0.88–0.89. I predict that the low values come from the 41 one-dimensional tensors, and that the convolution tensors stay close to 0.9. The `none` runs test this.

## Design (10 new runs)

| Arm | Variant | Seeds | Source |
|---|---|---|---|
| Adam | β₁ = 0.9 | 0–7 | Label_Diagnostic (0–3) + Exp. 7 (4–7) |
| Adam-NV | none | 0–7 | Label_Diagnostic (0–3) + Exp. 7 (4–7) |
| **Adam-NV, tensor-mean** | `tensor_mean` | 0–7 | **NEW** |
| Canary and per-tensor statistics | `none` | 0–1 | **NEW** (must equal the existing Adam-NV logs) |
| Shuffle arms | step, fixed | 0–7 | Exp. 11 (comparison only) |

All runs: 20% label noise, constant schedule, σ = 0, lr 1e-3 (default), 60 epochs, batch 128, `--deterministic`, `--optimizer adam_nv`.

## Pre-registered rule (before any Exp. 13 run; primary metric = last-5-epoch memorization, paired by seed)
M = Adam-NV − Adam; F = tensor-mean − Adam; D = Adam-NV − tensor-mean. Holm over {F, D, tensor-mean − shuffle/step, tensor-mean − shuffle/fixed}.
0. If the 95% CI of M is not above 0, the rule does not apply.
1. **Tensor-level adaptation suffices**: F ≥ 0.75·M and the 95% CI of D includes 0.
2. **Within-tensor spread matters**: F < 0.5·M and the 95% CI of D is above 0.
3. Otherwise **partial**: both levels contribute.
Read together with Exp. 11: rule 1 → the effect is layer-level; rule 2 → the within-tensor structure carries a substantial part; rule 3 → both contribute.

## Verified, and not verified
Verified (CPU PyTorch 2.14.1): 6 new unit tests pass (bit-identity of `none` with AdamNV, match with AdamNV when γ = 0 to float64 rounding, the applied β₁ equals each tensor's mean and is constant within a tensor, the per-tensor log, invalid mode rejected). The wrapper ran end to end on a real ResNet-18 for both variants in synthetic smoke mode and wrote the distribution, per-tensor and configuration files. The launcher was dry-run; the analysis was tested on synthetic arms with a known answer (those printed numbers mean nothing).
NOT verified: any GPU run, real label-noise runs, the outcome, and the time cost (small, since the tensor mean is cheap; plan for about 70 to 80 min per run).

## Before launching: hashes
Each launch writes `_code_hashes_*.txt`. Expect (first 8 characters): `part_b_resnet_v8.py` 242e6a77, `adam_nv.py` bcc7824b, `noise_schedule.py` 90e11e2c, `noise.py` eee61201, `experiment_metadata.py` ab020aaf; `adam_nv_shuffle.py` must equal the file used in Exp. 11 (this package contains an identical copy). The two new files are `adam_nv_tensormean.py` and `part_b_resnet_v8_tensormean.py`.

## Account plan (about 75 min per run)

| Account | Call (separate log folder) |
|---|---|
| A | `VARIANT=tensor_mean`, seeds 0 1 on GPU 0 and 2 3 on GPU 1 → `logs_exp13_tm_A` (2 rounds, about 2.5 h) |
| B | `VARIANT=tensor_mean`, seeds 4 5 / 6 7 → `logs_exp13_tm_B` (about 2.5 h) |
| C | `VARIANT=none`, seeds 0 / 1 → `logs_exp13_none_C` (1 round, about 1.3 h) |

## Steps (same setup as Exp. 7–12)

```bash
!unzip -o exp13_tensor_mean.zip -d /kaggle/working/Adam-NV
%cd /kaggle/working/Adam-NV/e2_noisy_gradient/src
!cd /kaggle/working/Adam-NV && python3 -m unittest discover -s tests -v 2>&1 | tail -8     # must end with OK

# ACCOUNT A
!VARIANT=tensor_mean SEEDS_GPU0="0 1" SEEDS_GPU1="2 3" DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp13_tm_A bash run_exp13_tensor_mean.sh
!cat /kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp13_tm_A/_code_hashes_*.txt
!cd /kaggle/working/Adam-NV/e2_noisy_gradient && zip -r exp13_logs_A.zip logs_exp13_tm_A

# ACCOUNT B
!VARIANT=tensor_mean SEEDS_GPU0="4 5" SEEDS_GPU1="6 7" DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp13_tm_B bash run_exp13_tensor_mean.sh
!cat /kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp13_tm_B/_code_hashes_*.txt
!cd /kaggle/working/Adam-NV/e2_noisy_gradient && zip -r exp13_logs_B.zip logs_exp13_tm_B

# ACCOUNT C
!VARIANT=none SEEDS_GPU0="0" SEEDS_GPU1="1" DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp13_none_C bash run_exp13_tensor_mean.sh
!cat /kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp13_none_C/_code_hashes_*.txt
!cd /kaggle/working/Adam-NV/e2_noisy_gradient && zip -r exp13_logs_C.zip logs_exp13_none_C
```
If a session dies, run the same cell again; complete runs are skipped. The checker table prints at the end of each call.

## Analysis (`analyze_exp07.py` and `analyze_exp13.py` in the same folder)

```bash
python3 analysis/analyze_exp13.py --orig-dir e2_noisy_gradient/logs/Label_Diagnostic \
    --exp07-dirs logs_exp07_A logs_exp07_B logs_exp07_C \
    --tm-dirs logs_exp13_tm_A logs_exp13_tm_B --none-dirs logs_exp13_none_C \
    --step-dirs logs_exp11_step_A logs_exp11_step_C --fixed-dirs logs_exp11_fixed_B logs_exp11_fixed_C
```

## Send back
All `exp13_logs_*.zip`, the unit-test output, the checker tables, the `_code_hashes` files, and any error verbatim. Do not rename or delete any log file.

## Do not
Change any argument, mix variants in one log folder, or edit any file.
