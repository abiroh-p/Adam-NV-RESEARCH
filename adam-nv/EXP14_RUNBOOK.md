# Experiment 14: Adam-NV controller on a subset of tensors, under 20% label noise (CIFAR-10, ResNet-18)

**Question:** Exp. 11 and 13 showed that about two thirds of Adam-NV's extra memorization (+3.91 pp over Adam) survives when β₁ is permuted within each tensor or replaced by the tensor's mean, so it comes from tensor-level adaptation. Exp. 13's per-tensor statistics show that the 41 one-dimensional tensors (normalization parameters and one bias; 9,610 elements) have the lowest β₁ on average (0.75 at epoch 60), but the 20 convolution tensors (99.9% of the elements) also fall below 0.9 (mean 0.85). Which tensors carry the effect?

## The intervention
`AdamNVSubset` (a subclass of `AdamNV`; `adam_nv.py` and `part_b_resnet_v8.py` are not modified) splits the parameters into two groups by tensor rank. In the adaptive group the controller runs as usual. In the other group γ = 0, so β₁ stays at 0.9 and those tensors follow plain Adam (exactly, up to the clamp on the running product).
- `onedim`: controller on tensors of rank ≤ 1 (41 tensors); plain Adam on the other 21.
- `rest`: controller on tensors of rank ≥ 2 (21 tensors, convolution and classifier weights); plain Adam on the 41 one-dimensional tensors.
The logged `beta1_*` columns average over all tensors including the plain-Adam ones (β₁ = 0.9), so they are diluted.

## Design (16 new runs)

| Arm | Subset | Seeds | Source |
|---|---|---|---|
| Adam | none | 0–7 | Label_Diagnostic (0–3) + Exp. 7 (4–7) |
| Adam-NV (all tensors) | all | 0–7 | Label_Diagnostic (0–3) + Exp. 7 (4–7) |
| **Controller on one-dimensional tensors only** | `onedim` | 0–7 | **NEW** |
| **Controller on rank ≥ 2 tensors only** | `rest` | 0–7 | **NEW** |

All runs: 20% label noise, constant schedule, σ = 0, lr 1e-3 (default), 60 epochs, batch 128, `--deterministic`, `--optimizer adam_nv`.

## Pre-registered rule (before any Exp. 14 run; primary metric = last-5-epoch memorization, paired by seed)
M = Adam-NV − Adam; F1 = onedim − Adam, F2 = rest − Adam; D1 = Adam-NV − onedim, D2 = Adam-NV − rest. Holm over {F1, F2, D1, D2}.
0. If the 95% CI of M is not above 0, the rule does not apply.
1. **One-dimensional tensors drive the effect**: F1 ≥ 0.75·M, the CI of D1 includes 0, and F2 < 0.5·M.
2. **Convolution and classifier tensors drive the effect**: F2 ≥ 0.75·M, the CI of D2 includes 0, and F1 < 0.5·M.
3. Otherwise **both subsets contribute** (or neither is sufficient). The analysis also reports F1 + F2 against M (additivity).

## Verified, and not verified
Verified (CPU PyTorch 2.14.1): 5 unit tests pass: the plain-Adam tensors match `torch.optim.Adam` (float64, 300 steps, tolerance 1e-6); the adaptive tensors are bit-identical to a stand-alone `AdamNV` on those tensors; the two subsets are complementary; γ = 0 makes every subset equal to Adam-NV with γ = 0; invalid arguments are rejected. The wrapper ran end to end on a real ResNet-18 for both subsets (synthetic smoke mode). The launcher was dry-run, and the analysis was tested on synthetic arms with a known answer (the printed numbers mean nothing).
NOT verified: any GPU run, real label-noise runs, and the outcome.

## Before launching: hashes
Each launch writes `_code_hashes_*.txt`. Expect (first 8 characters): `part_b_resnet_v8.py` 242e6a77, `adam_nv.py` bcc7824b, `noise_schedule.py` 90e11e2c, `noise.py` eee61201, `experiment_metadata.py` ab020aaf. The new files are `adam_nv_subset.py` and `part_b_resnet_v8_subset.py`.

## Account plan (about 70 min per run)

| Account | Calls (separate log folders) |
|---|---|
| A | `SUBSET=onedim`, seeds 0 1 2 on GPU 0 and 3 4 5 on GPU 1 → `logs_exp14_onedim_A` (about 3.5 h) |
| B | `SUBSET=rest`, seeds 0 1 2 and 3 4 5 → `logs_exp14_rest_B` (about 3.5 h) |
| C | `SUBSET=onedim`, seeds 6 / 7 → `logs_exp14_onedim_C`; then `SUBSET=rest`, seeds 6 / 7 → `logs_exp14_rest_C` (about 2.4 h) |

## Steps (same setup as Exp. 7–13)

```bash
!unzip -o exp14_subset.zip -d /kaggle/working/Adam-NV
%cd /kaggle/working/Adam-NV/e2_noisy_gradient/src
!cd /kaggle/working/Adam-NV && python3 -m unittest discover -s tests -v 2>&1 | tail -8     # must end with OK

# ACCOUNT A
!SUBSET=onedim SEEDS_GPU0="0 1 2" SEEDS_GPU1="3 4 5" DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp14_onedim_A bash run_exp14_subset.sh
!cat /kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp14_onedim_A/_code_hashes_*.txt
!cd /kaggle/working/Adam-NV/e2_noisy_gradient && zip -r exp14_logs_A.zip logs_exp14_onedim_A

# ACCOUNT B
!SUBSET=rest SEEDS_GPU0="0 1 2" SEEDS_GPU1="3 4 5" DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp14_rest_B bash run_exp14_subset.sh
!cat /kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp14_rest_B/_code_hashes_*.txt
!cd /kaggle/working/Adam-NV/e2_noisy_gradient && zip -r exp14_logs_B.zip logs_exp14_rest_B

# ACCOUNT C
!SUBSET=onedim SEEDS_GPU0="6" SEEDS_GPU1="7" DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp14_onedim_C bash run_exp14_subset.sh
!SUBSET=rest SEEDS_GPU0="6" SEEDS_GPU1="7" DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp14_rest_C bash run_exp14_subset.sh
!cd /kaggle/working/Adam-NV/e2_noisy_gradient && zip -r exp14_logs_C.zip logs_exp14_onedim_C logs_exp14_rest_C
```
If a session dies, run the same cell again; complete runs are skipped. The checker table prints at the end of each call.

## Analysis (`analyze_exp07.py` and `analyze_exp14.py` in the same folder)

```bash
python3 analysis/analyze_exp14.py --orig-dir e2_noisy_gradient/logs/Label_Diagnostic \
    --exp07-dirs logs_exp07_A logs_exp07_B logs_exp07_C \
    --onedim-dirs logs_exp14_onedim_A logs_exp14_onedim_C --rest-dirs logs_exp14_rest_B logs_exp14_rest_C
```

## Send back
All `exp14_logs_*.zip`, the unit-test output, the checker tables, the `_code_hashes` files, and any error verbatim. Do not rename or delete any log file.

## Do not
Change any argument, mix subsets in one log folder, or edit any file.
