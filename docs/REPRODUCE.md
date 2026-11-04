# Reproduction map

All commands run from the repository root unless a `cd` is shown. "GPU" = needs the original
type of hardware; **none of the GPU runs below were executed in the repair.**

| Result | Inputs | Command | Needs GPU |
|---|---|---|---|
| Provenance table | `*_metadata.json`, `_code_hashes_*.txt`, zips | `python tools/provenance.py verify --csv reports/provenance_verification.csv` | no |
| Log completeness | all per-epoch CSVs | `python tools/check_runs.py --csv reports/run_completeness.csv` | no |
| E1 table / paired tests | `e1_convergence/logs`, `adam-nv/e1_convergence/{AdditionalsLogs,logs}` | `python analysis/analyze_e1.py [--metric final] [--use-supplement]` | no |
| E13--E15 table | `e2_noisy_gradient/logs/{dynamic_noise,Intermittent,response_lag,...}` | `python analysis/analyze_e13_e15.py [--metric final]` | no |
| EXP06 (lr 3e-3 sanity) | `exp05_logs_*.zip` (extract first) | `python adam-nv/analysis/analyze_exp06.py --log-dirs <dirs> --tag _lr3e-3 --opts adam adam_nv --ref adam` | no |
| EXP07 / EXP08 | `adam-nv/e2_noisy_gradient/logs/*.zip` (extract first) + `e2_noisy_gradient/logs/Label_Diagnostic` | see command blocks in `adam-nv/EXP07_RUNBOOK.md` and `EXP08_RUNBOOK.md` | no |
| E1 training | CIFAR-10 | `cd adam-nv/e1_convergence/src && python run_e1_suite.py ...` | yes |
| E2/E13--E16 training | CIFAR-10/100 | `cd e2_noisy_gradient/src && python part_b_resnet_v8.py ...` or the `run_*_suite.py` scripts | yes |
| EXP09 smooth-vs-distrust (label noise) | `adam-nv/e2_noisy_gradient/logs/exp09_logs_{A,B}.zip` (extract first) | `python adam-nv/analysis/analyze_exp09.py --orig-dir e2_noisy_gradient/logs/Label_Diagnostic --exp07-dirs <A B C> --smooth-dirs <A B>` | no |
| EXP10 (lr 3e-3, 10 seeds) | `adam-nv/e1_convergence/exp10_logs_{A,B}.zip`, `adam-nv/e2_noisy_gradient/logs/exp10_logs_C.zip`, EXP05 zips; **Exp.6 logs (beta1 0.83/0.80 seeds 0-3) absent** | `python adam-nv/analysis/analyze_exp10.py --log-dirs ...` | no |
| EXP11 (beta1 shuffle) | `adam-nv/e2_noisy_gradient/logs/exp11_logs_{A,B,C}.zip` | `python adam-nv/analysis/analyze_exp11.py ... --strict-seeds` (see `tools/validate_all.py` step 13e) | no |
| EXP12 (constant beta1 0.92/0.95) | `adam-nv/e2_noisy_gradient/logs/exp12_logs_{1,2,3,4}.zip` (file names carry no beta1; pass each folder explicitly) | `python adam-nv/analysis/analyze_exp12.py ... --strict-seeds` (step 13f) | no |
| Timing v2 | `adam-nv/benchmarks/timing_v2_results.zip` | summary only; re-measuring needs one free T4 (`adam-nv/TIMING_RUNBOOK.md`) | yes |
| Paper-draft claim check | all logs above + `adam-nv/paperdraft/adamnv_paper_draft.pdf` | `python reports/_build_paper_check.py` | no |
| All tests | -- | `python tools/run_all_tests.py` | no |

EXP03 and EXP05 analyses (`analyze_exp03/05`) and `PROTOCOL_V2.md` are referenced by runbooks but
are **not in the repository**; their numbers cannot be regenerated from repository code.
