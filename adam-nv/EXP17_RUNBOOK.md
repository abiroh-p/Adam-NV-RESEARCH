# Experiment 17: a second architecture (small Vision Transformer; VGG-11 as an option)

**Question:** every result so far uses a ResNet-18 on CIFAR-10 or CIFAR-100. Do the main findings survive on an architecture with a different inductive bias? (i) Does Adam-NV beat Adam on clean accuracy? (ii) Under 20% label noise, does Adam-NV memorize more corrupted labels than Adam, and does a constant β₁ reproduce it?

## Architectures (`models_alt.py`, no pretrained weights, no tuning)
- `vit_tiny`: Vision Transformer for 32×32 inputs, patch 4 (64 tokens plus a class token), width 192, depth 6, 3 heads, MLP width 384, pre-LayerNorm, learned position embeddings, no dropout; 1,806,538 parameters. It uses LayerNorm, not BatchNorm.
- `vgg11_bn`: VGG-11 with BatchNorm, no residual connections, linear classifier; 9,228,362 parameters.

## Design per architecture and setting (15 runs)
Arms: Adam, Adam-NV, and Adam with a constant β₁ of 0.85 (`adam_nv_fixed`; 0.85 is a mid-range value, chosen before the runs). Seeds 0–4. Settings: clean labels, and 20% symmetric label noise (memorization measured as before). Everything else as in the ResNet experiments (60 epochs, batch 128, constant schedule, σ = 0, `--deterministic`, cosine learning-rate schedule). Learning rate: **5e-4 for `vit_tiny`** (a common value for small ViTs; there is no warmup in the training script) and 1e-3 for `vgg11_bn`. The learning rate is not tuned for any optimizer.
`part_b_resnet_v8.py` and `adam_nv.py` are not modified; `part_b_resnet_v8_arch.py` replaces only `build_model`.

## Pilot first (go / no-go, about 15 minutes)
A small ViT trained from scratch with Adam and no warmup can diverge or stall. Before the main runs, run 5 epochs of each optimizer with seed 0 in a separate folder:
```bash
!ARCH=vit_tiny NOISE=0.0 EPOCHS=5 JOBS_GPU0="adam:0 adam_nv:0" JOBS_GPU1="adam_nv_fixed:0" DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/pilot_exp17_vit bash run_exp17_arch.sh
!python3 -c "import pandas as pd,glob; [print(f.split('/')[-1], pd.read_csv(f).val_acc.round(3).tolist()) for f in sorted(glob.glob('/kaggle/working/Adam-NV/e2_noisy_gradient/pilot_exp17_vit/*_seed0.csv'))]"
```
Go if, for all three optimizers, the validation accuracy rises over the five epochs and reaches at least 0.35 at epoch 5. If not, repeat the pilot with `LR=3e-4` and use that value for every main run (all arms and settings must share one learning rate). If it still fails, use `ARCH=vgg11_bn` instead. Report the pilot and any change in the paper. (The pilot calls the launcher with `EPOCHS=5`, so its checker table will show 5 epochs; ignore the unit-test and hash steps for the pilot.)

## Pre-registered rules (before any Exp. 17 run; M = Adam-NV − Adam, F = constant − Adam)
Clean accuracy (primary: last-5-epoch validation accuracy): **Adam-NV better** if the 95% CI of M is above 0; **worse** if below 0; otherwise **no detectable difference** (flagged equivalent only if the whole CI lies within ±0.15 pp).
Memorization under label noise (primary: last-5-epoch memorization): **effect reproduced** if the CI of M is above 0 and F < 0.5·M; **effect also in a constant β₁** if the CI of M is above 0 and F ≥ 0.5·M; **no effect** if the CI includes 0; **reversed** if it is below 0. Holm over the three contrasts of each setting.

## Verified, and not verified
Verified (CPU PyTorch 2.14.1): 5 unit tests pass (output shapes for 10 and 100 classes, parameter counts, seed reproducibility, every parameter receives a gradient, unknown architectures rejected). The wrapper ran end to end for both architectures with Adam-NV in synthetic smoke mode and wrote the logs and the sidecar. The launcher was dry-run for both settings and architectures. The checker accepted a fabricated complete folder and rejected a missing seed and a wrong architecture or learning rate. The analysis ran on stand-in logs (its printed numbers mean nothing).
NOT verified: that the ViT trains well at 5e-4 without warmup (hence the pilot), any real-data run, the running time (my guess is 20 to 40 minutes per ViT run on a T4; not measured), and the outcome.

## Before launching: hashes
Expect (first 8 characters) `part_b_resnet_v8.py` 242e6a77, `adam_nv.py` bcc7824b, `noise_schedule.py` 90e11e2c, `noise.py` eee61201, `experiment_metadata.py` ab020aaf. The new files are `models_alt.py` and `part_b_resnet_v8_arch.py`.

## Plan after a successful pilot: four calls for the ViT (about 3 h each)

| Call | Setting | GPU 0 jobs | GPU 1 jobs | Log folder |
|---|---|---|---|---|
| 1 | clean (`NOISE=0.0`) | adam:0 adam:1 adam:2 adam:3 adam:4 | adam_nv:0 adam_nv:1 adam_nv:2 adam_nv:3 adam_nv:4 | `logs_exp17_vit_clean_1` |
| 2 | clean | adam_nv_fixed:0 adam_nv_fixed:1 adam_nv_fixed:2 | adam_nv_fixed:3 adam_nv_fixed:4 | `logs_exp17_vit_clean_2` |
| 3 | noise (`NOISE=0.2`) | adam:0 … adam:4 | adam_nv:0 … adam_nv:4 | `logs_exp17_vit_noisy_1` |
| 4 | noise | adam_nv_fixed:0 adam_nv_fixed:1 adam_nv_fixed:2 | adam_nv_fixed:3 adam_nv_fixed:4 | `logs_exp17_vit_noisy_2` |

Optionally repeat the same four calls with `ARCH=vgg11_bn` (log folders `..._vgg_...`) as a second check.

## Steps (same setup as Exp. 7–16)

```bash
!unzip -o exp17_architecture.zip -d /kaggle/working/Adam-NV
%cd /kaggle/working/Adam-NV/e2_noisy_gradient/src
!cd /kaggle/working/Adam-NV && python3 -m unittest discover -s tests -v 2>&1 | tail -8     # must end with OK

# CALL 1 (change NOISE, JOBS and LOG_DIR for the other calls)
!ARCH=vit_tiny NOISE=0.0 JOBS_GPU0="adam:0 adam:1 adam:2 adam:3 adam:4" JOBS_GPU1="adam_nv:0 adam_nv:1 adam_nv:2 adam_nv:3 adam_nv:4" DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp17_vit_clean_1 bash run_exp17_arch.sh
!cat /kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp17_vit_clean_1/_code_hashes_*.txt
!cd /kaggle/working/Adam-NV/e2_noisy_gradient && zip -r exp17_logs_1.zip logs_exp17_vit_clean_1
```
If a session dies, run the same cell again; complete runs are skipped. If you used `LR=3e-4` after the pilot, add `LR=3e-4` to every call. The checker table prints at the end of each call.

## Analysis (`analyze_exp07.py` and `analyze_exp17.py` in the same folder)

```bash
python3 analysis/analyze_exp17.py --clean-dirs logs_exp17_vit_clean_1 logs_exp17_vit_clean_2 \
    --noisy-dirs logs_exp17_vit_noisy_1 logs_exp17_vit_noisy_2
```

## Send back
The pilot printout, all `exp17_logs_*.zip`, the unit-test output, the checker tables, the `_code_hashes` files, and any error verbatim. Do not rename or delete any log file.

## Do not
Change any argument, mix architectures or settings in one log folder, or edit any file.
