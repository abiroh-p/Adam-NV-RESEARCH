# Experiment index

Numbers are the repository's run-book numbers; they are the numbers used in the paper for Experiments 14-21.
Paths are relative to `adam-nv/` unless they start with `e2_noisy_gradient/` at the repository root.

## Experiments 14-21 (the additional diagnostics of the paper)

| Exp. | Question | Run book | Source | Logs (`e2_noisy_gradient/logs/`) | Analysis | Paper |
|---|---|---|---|---|---|---|
| 14 | Where does the controller act? Controller only on the 41 one-dimensional tensors or only on the other 21 (ResNet-18, 20% noise, seeds 0-7) | `EXP14_RUNBOOK.md` | `adam_nv_subset.py`, `part_b_resnet_v8_subset.py`, `run_exp14_subset.sh` | `logs_exp14_onedim_A`, `logs_exp14_rest_B`, `exp14_logs_C` (+ Adam and Adam-NV baselines: `Label_Diagnostic`, `logs_exp07_A/B`, `exp07_log_C`) | `analysis/analyze_exp14.py` | V-E |
| 15 | Label-noise rate: 10% and 40% (seeds 0-5) | `EXP15_RUNBOOK.md` | `run_exp15_label_noise_rates.sh` | `exp15_logs_1` ... `exp15_logs_4` | `analysis/analyze_exp15.py` | V-E |
| 16 | Does the noise proxy track gradient noise? First diagnostic (K=16, 2 seeds) | `EXP16_RUNBOOK.md` | `noise_proxy_diagnostic.py`, `run_exp16_proxy_diagnostic.sh` | `exp16` | `analysis/analyze_exp16.py` | V-G |
| 17 | Second architecture: ViT-tiny, clean and 20% noise, Adam / Adam-NV / constant 0.85 (5 seeds) | `EXP17_RUNBOOK.md` | `models_alt.py`, `part_b_resnet_v8_arch.py`, `run_exp17_arch.sh` | `exp17_logs_1` ... `exp17_logs_4` | `analysis/analyze_exp17.py` | V-H |
| 18 | ViT constant-beta1 sweep (0.70, 0.80, 0.92, 0.95; 5 seeds) | `EXP18_19_20_RUNBOOK.md` | (uses the Exp. 17 launcher) | `exp18_logs_1` ... `exp18_logs_4` (one constant each) | `analysis/analyze_exp18.py` | V-H |
| 19 | Null run: Adam-NV code with gamma=0, beta1=0.9 against Adam (seeds 0-7) | `EXP18_19_20_RUNBOOK.md` | `run_exp19_null.sh` | `exp19_logs_1`, `exp19_logs_2` | `analysis/analyze_exp19.py` | III-B, V-E |
| 20 | Constant beta1 = 0.79 on the one-dimensional tensors only (seeds 0-7) | `EXP18_19_20_RUNBOOK.md` | `adam_const_subset.py`, `part_b_resnet_v8_constsubset.py`, `run_exp20_constsubset.sh` | `exp20_logs_1`, `exp20_logs_2` | `analysis/analyze_exp20.py` | V-E |
| 21 | Noise-proxy diagnostic v2 (K=64, reliability, absolute scale, lambda=0; 5 seeds) | `EXP21_RUNBOOK.md` | `noise_proxy_diagnostic_v2.py`, `run_exp21_proxy_v2.sh` | `exp21_logs_A`, `exp21_logs_B` (the `.npz` sample files are not included) | `analysis/analyze_exp21.py` | V-G |

Each log archive contains the CSV logs, `*_metadata.json`, stdout, the status files and `_code_hashes_*.txt`. The verification script of experiment NN is
`tools/verify_eNN_numbers.py` inside `ResearchPaperSupplements/adamnv_paper.zip`; `tools/reproduce_paper_numbers.py` runs all of them.

## Earlier experiments (1-13)

Run books `EXP01_RUNBOOK.md` ... `EXP13_RUNBOOK.md` and `TIMING_RUNBOOK.md` in `adam-nv/` describe the clean benchmark, the learning-rate and constant-beta1
experiments, the non-stationary noise schedules, the controller-direction and CIFAR-100 experiments, and the label-noise controls (constants, smooth
response, beta1 shuffling and tensor mean). Their analysis scripts are in `adam-nv/analysis/` and their logs in `adam-nv/e1_convergence/logs/` and
`adam-nv/e2_noisy_gradient/logs/`.
