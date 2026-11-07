# Experiment 15: Adam versus Adam-NV at other label-noise rates (10% and 40%), CIFAR-10, ResNet-18

**Question:** every label-noise result so far (extra memorization of +3.91 pp for Adam-NV, the controls, the smooth response, the shuffle and tensor-mean controls) uses one setting: symmetric 20% label noise. Does the Adam-NV memorization effect appear at 10% and at 40% noise?

## Design (24 new runs; no new code)

| Arm | Noise rate | Seeds | Runs |
|---|---|---|---|
| Adam | 10% | 0–5 | 6 |
| Adam-NV | 10% | 0–5 | 6 |
| Adam | 40% | 0–5 | 6 |
| Adam-NV | 40% | 0–5 | 6 |

Everything except the noise rate is as in Exp. 7: `part_b_resnet_v8.py` unchanged, symmetric noise (a random different class, corrupted subset determined by the run seed), constant schedule, σ = 0, lr 1e-3 (default), 60 epochs, batch 128, `--deterministic`. Six seeds per arm: the 20%-noise effect has a paired SD of about 1 pp, so six seeds resolve differences of roughly 1.2 pp.

## Pre-registered rule (before any Exp. 15 run; primary metric = last-5-epoch memorization, paired by seed)
M_r = Adam-NV − Adam at noise rate r. Holm over {M₀.₁, M₀.₄}; the 20%-noise reference is +3.91 pp.
At each rate: **effect present** if the 95% CI of M_r is above 0; **no effect** if it includes 0; **reversed** if it is below 0. The memorization effect **generalizes across noise rates** if it is present at both 10% and 40%; if present at only one rate, the dependence on the rate is reported; if at neither, it may be specific to 20%.
Secondary metrics (reported, not decisive): validation accuracy, clean-subset accuracy, peak validation accuracy.

## Verified, and not verified
Verified (CPU): shell syntax and a dry run of the launcher (it builds the exact `part_b_resnet_v8.py` command with `--label-noise 0.1` or `0.4`); the checker on a fabricated complete folder (it accepts a matching folder and rejects a missing seed and a wrong noise rate); the analysis on synthetic logs with a known answer (its printed numbers mean nothing).
NOT verified: any GPU run, and the outcome. I have not checked how Adam behaves at 40% noise on this setup; memorization may be much lower or higher than at 20%.

## Before launching: hashes
Each launch writes `_code_hashes_*.txt`. Expect (first 8 characters): `part_b_resnet_v8.py` 242e6a77, `adam_nv.py` bcc7824b, `noise_schedule.py` 90e11e2c, `noise.py` eee61201, `experiment_metadata.py` ab020aaf.

## Plan: four independent calls (about 65 min per run, 3 rounds per call, about 3.3 h each)
Run them on whichever accounts are free. File names contain the noise rate, but use a separate log folder per call anyway.

| Call | Noise | Seeds | GPU 0 jobs | GPU 1 jobs | Log folder |
|---|---|---|---|---|---|
| 1 | 0.1 | 0–2 | adam:0 adam:1 adam:2 | adam_nv:0 adam_nv:1 adam_nv:2 | `logs_exp15_ln01_1` |
| 2 | 0.1 | 3–5 | adam:3 adam:4 adam:5 | adam_nv:3 adam_nv:4 adam_nv:5 | `logs_exp15_ln01_2` |
| 3 | 0.4 | 0–2 | adam:0 adam:1 adam:2 | adam_nv:0 adam_nv:1 adam_nv:2 | `logs_exp15_ln04_1` |
| 4 | 0.4 | 3–5 | adam:3 adam:4 adam:5 | adam_nv:3 adam_nv:4 adam_nv:5 | `logs_exp15_ln04_2` |

## Steps (same setup as Exp. 7–14)

```bash
!unzip -o exp15_label_noise_rates.zip -d /kaggle/working/Adam-NV
%cd /kaggle/working/Adam-NV/e2_noisy_gradient/src
!cd /kaggle/working/Adam-NV && python3 -m unittest discover -s tests -v 2>&1 | tail -8     # must end with OK

# CALL 1 (change LN, JOBS and LOG_DIR for the other calls)
!LN=0.1 JOBS_GPU0="adam:0 adam:1 adam:2" JOBS_GPU1="adam_nv:0 adam_nv:1 adam_nv:2" DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp15_ln01_1 bash run_exp15_label_noise_rates.sh
!cat /kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp15_ln01_1/_code_hashes_*.txt
!cd /kaggle/working/Adam-NV/e2_noisy_gradient && zip -r exp15_logs_1.zip logs_exp15_ln01_1
```
For calls 2 to 4 use the noise rates, job lists and folders in the table (for example call 3: `LN=0.4`, `JOBS_GPU0="adam:0 adam:1 adam:2"`, `JOBS_GPU1="adam_nv:0 adam_nv:1 adam_nv:2"`, `LOG_DIR=.../logs_exp15_ln04_1`) and `zip -r exp15_logs_<call>.zip <folder>`. If a session dies, run the same cell again; complete runs are skipped. The checker table prints at the end of each call.

## Analysis (`analyze_exp07.py` and `analyze_exp15.py` in the same folder)

```bash
python3 analysis/analyze_exp15.py --ln01-dirs logs_exp15_ln01_1 logs_exp15_ln01_2 \
    --ln04-dirs logs_exp15_ln04_1 logs_exp15_ln04_2
```

## Send back
All `exp15_logs_*.zip`, the unit-test output, the checker tables, the `_code_hashes` files, and any error verbatim. Do not rename or delete any log file.

## Do not
Change any argument, put two noise rates in one log folder, or edit any file.
