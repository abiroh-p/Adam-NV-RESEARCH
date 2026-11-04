# Experiments 18, 19 and 20 (run book; written before any of these runs)

Status: rules below are fixed BEFORE the runs. If anything is changed after seeing results, record it in the deviations log at the end.

| Exp. | Question | Setting | Arms | Seeds | Core runs | Est. GPU wall time (2x T4) |
|---|---|---|---|---|---|---|
| 18 | On the ViT, does a constant beta1 matched to Adam-NV's learned value reproduce the +3.85 pp memorization effect? | ViT-tiny, 20% noise, lr 5e-4 | constant beta1 = 0.80 (matched), 0.70, 0.95 (+ optional 0.92); references: Adam, Adam-NV, 0.85 from Exp. 17 | 0-4 | 15 | 3 calls x ~2.5 h |
| 19 | How far does Adam-NV's code path with gamma = 0 move away from torch Adam (trajectory noise floor)? | ResNet-18, 20% noise, lr 1e-3 (default) | `adam_nv_fixed` with beta1 = 0.9 (gamma = 0) vs the existing Adam logs | 0-7 | 8 | 2 calls x ~2.1 h |
| 20 | On the one-dimensional tensors only, is a constant lower beta1 enough to give Exp. 14's +1.19 pp, or is the adaptivity needed? | ResNet-18, 20% noise | constant beta1 = 0.79 on the 41 rank<=1 tensors, plain Adam elsewhere; references: Adam, Exp. 14 `onedim` arm | 0-7 | 8 (+8 optional with 0.70) | 2 calls x ~2.1 h |

Why these constants. "Matched" means the time average of Adam-NV's logged beta1, as in the paper's earlier controls. ViT: Adam-NV logs 0.79 (clean) and 0.80 (20% noise), tensor-mean over the 80 tensors, so the matched constant is 0.80 (the 0.85 of Exp. 17 was fixed beforehand and is NOT matched). ResNet one-dimensional tensors: the undiluted time-average of beta1 over the 41 tensors in the Exp. 14 `onedim` runs is 0.789 (8 seeds, seed SD 0.0004), so 0.79. 0.70 and 0.95 bracket the matched value as in the ResNet study.

Recommended order (most informative first): 18 (0.80) -> 19 (seeds 0-3, then 4-7) -> 20 (0.79, seeds 0-3, then 4-7) -> 18 (0.70, 0.95) -> optional 18 (0.92), 20 (0.70).

## 0. Setup (once per Kaggle session)

Unzip this package into the repository root (`/kaggle/working/Adam-NV`) so the folder layout matches Exp. 14/17. Then, from the repository root:

```bash
cd /kaggle/working/Adam-NV
python3 -m unittest discover -s tests -v          # must print OK for all tests; send me the output
ls e2_noisy_gradient/src/adam_const_subset.py e2_noisy_gradient/src/run_exp19_null.sh e2_noisy_gradient/src/run_exp20_constsubset.sh
```

Environment must be the one used so far: python 3.12.13, torch 2.10.0+cu128, torchvision 0.25.0+cu128, 2 x Tesla T4 (the launchers check this and refuse to start otherwise). Do not change any other file. Each launcher records the SHA-256 of the code in `_code_hashes_*.txt`.

Every call below needs its OWN `LOG_DIR` (file names do not contain the constant). Use exactly these folder names. Add `DRY_RUN=1` in front of a command to print the commands without running.

## 1. Experiment 18: ViT constant-beta1 sweep (20% noise, seeds 0-4)

Run from `e2_noisy_gradient/src`. Launcher: the existing `run_exp17_arch.sh`.

```bash
cd /kaggle/working/Adam-NV/e2_noisy_gradient/src

# 18a: matched constant 0.80  (do this one first)
ARCH=vit_tiny NOISE=0.2 FIXED_BETA1=0.80 \
JOBS_GPU0="adam_nv_fixed:0 adam_nv_fixed:1 adam_nv_fixed:2" JOBS_GPU1="adam_nv_fixed:3 adam_nv_fixed:4" \
DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp18_vit_b080 \
bash run_exp17_arch.sh

# 18b: 0.70
ARCH=vit_tiny NOISE=0.2 FIXED_BETA1=0.70 \
JOBS_GPU0="adam_nv_fixed:0 adam_nv_fixed:1 adam_nv_fixed:2" JOBS_GPU1="adam_nv_fixed:3 adam_nv_fixed:4" \
DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp18_vit_b070 \
bash run_exp17_arch.sh

# 18c: 0.95
ARCH=vit_tiny NOISE=0.2 FIXED_BETA1=0.95 \
JOBS_GPU0="adam_nv_fixed:0 adam_nv_fixed:1 adam_nv_fixed:2" JOBS_GPU1="adam_nv_fixed:3 adam_nv_fixed:4" \
DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp18_vit_b095 \
bash run_exp17_arch.sh

# 18d (optional): 0.92
ARCH=vit_tiny NOISE=0.2 FIXED_BETA1=0.92 \
JOBS_GPU0="adam_nv_fixed:0 adam_nv_fixed:1 adam_nv_fixed:2" JOBS_GPU1="adam_nv_fixed:3 adam_nv_fixed:4" \
DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp18_vit_b092 \
bash run_exp17_arch.sh
```

Do NOT pass `LR`: the default 5e-4 is the Exp. 17 value. The launcher's checker runs at the end and must print `5/5 jobs verified`.

## 2. Experiment 19: null run, Adam-NV code with gamma = 0 (ResNet-18, 20% noise, seeds 0-7)

```bash
cd /kaggle/working/Adam-NV/e2_noisy_gradient/src

# 19a: seeds 0-3
SEEDS_GPU0="0 1" SEEDS_GPU1="2 3" DATA_DIR=/kaggle/working/Adam-NV/data \
LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp19_null_A bash run_exp19_null.sh

# 19b: seeds 4-7
SEEDS_GPU0="4 5" SEEDS_GPU1="6 7" DATA_DIR=/kaggle/working/Adam-NV/data \
LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp19_null_B bash run_exp19_null.sh
```

Note: `adam_nv_fixed` with `--fixed-beta1 0.9` gives beta1 = 0.9 exactly (the Exp. 14 plain groups had 0.9 + 4e-8 from float32 rounding of 0.1 + 0.8); this difference is immaterial to the question (any tiny perturbation gives a different realization of training), but it is a difference.

## 3. Experiment 20: constant beta1 on the one-dimensional tensors only (ResNet-18, 20% noise, seeds 0-7)

```bash
cd /kaggle/working/Adam-NV/e2_noisy_gradient/src

# 20a: constant 0.79, seeds 0-3
SUBSET=onedim CONST_BETA1=0.79 SEEDS_GPU0="0 1" SEEDS_GPU1="2 3" DATA_DIR=/kaggle/working/Adam-NV/data \
LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp20_onedim_c079_A bash run_exp20_constsubset.sh

# 20b: constant 0.79, seeds 4-7
SUBSET=onedim CONST_BETA1=0.79 SEEDS_GPU0="4 5" SEEDS_GPU1="6 7" DATA_DIR=/kaggle/working/Adam-NV/data \
LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp20_onedim_c079_B bash run_exp20_constsubset.sh

# optional 20c/20d: constant 0.70 (same two calls with CONST_BETA1=0.70 and LOG_DIR ..._c070_A / ..._c070_B)
```

The logged beta1 columns of these runs are diluted (equal weight per tensor over all 62 tensors, 41 at the constant and 21 at 0.9): the expected `beta1_mean` is (41 x 0.79 + 21 x 0.9) / 62 = 0.8275, `beta1_min` 0.79, `beta1_max` 0.9. A different value means something is wrong; tell me.

## 4. What to send back

For every call: the whole `LOG_DIR` zipped (CSV, metadata, stdout, `_status_*`, `_code_hashes_*`, and the sidecar json), plus the last lines of the call's output (the checker table). Once: the `unittest` output. Also still missing from Exp. 17: the pilot printout; please confirm whether the pilot was run and that the learning rate was not changed.

## 5. Analysis (I run these when the logs arrive; commands for reproducibility)

```bash
cd /kaggle/working/Adam-NV/analysis
python3 analyze_exp18.py --adam-dir ../exp17_logs_3 --const 0.85=../logs_exp17_vit_noisy_2 0.70=../e2_noisy_gradient/logs_exp18_vit_b070 0.80=../e2_noisy_gradient/logs_exp18_vit_b080 0.95=../e2_noisy_gradient/logs_exp18_vit_b095
python3 analyze_exp19.py --adam-dirs ../Label_Diagnostic ../logs_exp07_A ../logs_exp07_B --null-dirs ../e2_noisy_gradient/logs_exp19_null_A ../e2_noisy_gradient/logs_exp19_null_B --nv-dirs ../Label_Diagnostic ../exp07_log_C/log_exp07_C
python3 analyze_exp20.py --adam-dirs ../Label_Diagnostic ../logs_exp07_A ../logs_exp07_B ../exp07_log_C/log_exp07_C --onedim-dirs ../logs_exp14_onedim_A ../logs_exp14_onedim_C --const-dirs ../e2_noisy_gradient/logs_exp20_onedim_c079_A ../e2_noisy_gradient/logs_exp20_onedim_c079_B
```

## 6. Pre-registered decision rules (primary metric: last-5-epoch memorization, pp; paired by seed; t-based 95% CI; Holm as stated)

Exp. 18 (per constant c; M = Adam-NV - Adam, F_c = constant - Adam, D_c = Adam-NV - constant). Requires the 95% CI of M above 0.
- EXPLAINED BY THE CONSTANT: F_c >= 0.5 M and the CI of D_c includes 0.
- MOSTLY EXPLAINED, SMALL RESIDUAL: F_c >= 0.5 M and the CI of D_c is above 0.
- ADAPTATION BEYOND THE CONSTANT: F_c < 0.25 M and the CI of D_c is above 0.
- otherwise INCONCLUSIVE.
Holm family: F_c and D_c for c in {0.70, 0.80, 0.95} (six contrasts). 0.85 (Exp. 17) and 0.92 are reported outside the family. The overall outcome is the outcome at the matched constant 0.80. The pooled slope across constants is descriptive only.

Exp. 19. N0 = null - Adam. Equivalence margin +/-0.5 pp (about 13% of the +3.91 pp effect).
- CODE PATH EQUIVALENT TO ADAM: the whole 95% CI of the mean of N0 is inside (-0.5, +0.5).
- CODE PATH DIFFERS: the CI excludes 0 and |mean| > 0.5. Then the Exp. 14 contrasts must be re-based on this null arm.
- otherwise INCONCLUSIVE. The paired SD of N0 is reported as the trajectory floor and compared with the paired SD of M.
No Holm family (one contrast per metric).

Exp. 20 (F1 = Exp. 14 `onedim` - Adam; C = constant 1-D - Adam; E = Exp. 14 `onedim` - constant 1-D). Requires the CI of F1 above 0 on these seeds.
- LOWERING beta1 ON THESE TENSORS SUFFICES: C >= 0.75 F1 and the CI of E includes 0.
- ADAPTIVITY NEEDED: C < 0.5 F1 and the CI of E is above 0.
- otherwise PARTLY / INCONCLUSIVE.
Holm family: {C, E}. F1 is the Exp. 14 reference.

Not claimed in any case: "no detectable difference" is never read as equivalence, except in Exp. 19 where equivalence is defined by the margin above. With 5 seeds the exact sign-flip test cannot go below 0.0625, and the t-based Holm is the registered inference, as in Exp. 17.

## 7. How the outcomes change the paper (decided now, so the text follows the data)

- Exp. 18 EXPLAINED at 0.80: the paper says the ViT effect is reproduced by a matched constant, and the ResNet-only result "no constant reproduces it" is reported as architecture-specific. Abstract and conclusion keep both statements. ADAPTATION BEYOND: the ViT supports the ResNet claim; the "second architecture" paragraph and the abstract sentence are rewritten accordingly.
- Exp. 19 EQUIVALENT: one sentence in Methods ("a gamma = 0 run through the same code differs from torch Adam by X pp, interval Y"). DIFFERS: Exp. 14 contrasts are recomputed against the null arm and the E14 text changes.
- Exp. 20 SUFFICES: the E14 statement "no constant on those tensors was run" becomes "a constant 0.79 on those tensors reproduces the effect, so adaptivity is not needed there". NEEDED: "adaptivity on the one-dimensional tensors is needed".

## 8. Experiment 21 (not in this package)

Proxy diagnostic v2 for Exp. 16: log absolute gradient rms and per-tensor means of s and q, K >= 64, a lambda = 0 variant of the score, >= 5 seeds, clamp share logged, and a within-tensor registered statistic. It needs new diagnostic code and tests and depends on what Exp. 18-20 show, so it follows them.

## Deviations log (fill in if anything differs from the above)

(none yet)
