# Adam-NV: a controlled negative-result study

Code, logs, run books, pre-registered decision rules and verification scripts for the paper
**"Does Noise-Adaptive Momentum Help? A Controlled Negative-Result Study of Adam-NV"**.

Everything the paper reports can be recomputed from the raw logs stored here, without a GPU:

```bash
pip install -r requirements.txt          # numpy, pandas, scipy are enough for the next command
python tools/reproduce_paper_numbers.py  # recomputes the Experiment 14-21 numbers of the paper from the raw logs (expects: ALL VERIFIERS PASS)
python tools/run_all_tests.py            # unit tests of every optimizer variant and analysis script (needs PyTorch)
sha256sum -c MANIFEST.sha256             # every file in the repository against its recorded checksum
```

## Layout

| Folder | Content |
|---|---|
| `adam-nv/` | Experiments 1-21: run books (`EXP*_RUNBOOK.md`), source (`*/src`), analysis scripts (`analysis/`), tests (`tests/`) and the raw logs as zip archives (`e2_noisy_gradient/logs/`) |
| `e2_noisy_gradient/` | the earlier source tree and the `Label_Diagnostic` baseline logs (Adam and Adam-NV, seeds 0-3, 20% label noise) |
| `ResearchPaperSupplements/adamnv_paper.zip` | the paper source (`main.tex`, figures, bibliography) and its number-verification scripts (`tools/verify_e14 ... verify_e21`) |
| `docs/` | `EXPERIMENT_INDEX.md` (which experiment is where), `REPRODUCE.md`, `REPO_LAYOUT.md` |
| `provenance/` | archive of the exact source files whose SHA-256 values the logs record, and the registry that ties them to the logs |
| `tools/` | provenance, manifest, test runner and reproduction scripts |
| `tests/` | tests of the repository tooling |
| `environment/` | the software environment of the experiments |
| `reports/` | historical audit reports of an earlier repair pass (see `reports/README_HISTORICAL.md`) |

`docs/EXPERIMENT_INDEX.md` maps every experiment to its run book, scripts, logs and analysis. The run books contain the decision rules,
written before the runs they govern.

## Environment

The experiments ran in Python 3.12.13, PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128, on two Tesla T4 GPUs with deterministic mode enabled
(see `environment/ENVIRONMENT.md`). The statistics and the verification scripts need only numpy, pandas and scipy.

## Provenance of the code that produced the logs

Each log folder from Experiment 14 onward records the SHA-256 of the source files it ran (`_code_hashes_*.txt`). Every one of those hashes matches a
file stored in this repository byte for byte. Two points matter when you edit or copy files:

* The hash-pinned sources keep their original bytes, including Windows line endings in some files (`.gitattributes` sets `*.py -text`). An editor or
  `git` that rewrites line endings changes the SHA-256 and breaks the link between a log and its code.
* `python tools/provenance.py verify` classifies every hash claim in the repository. For Experiments 14-21 every claim is verified (byte-exact copies are
  archived in `provenance/source_archive`). A remaining group of claims, from the very first experiments, names four early source versions
  (`train_resnet.py` twice, an early `adam_nv.py` and `gono.py`) that are not in this repository, and the earliest logs recorded no hashes at all;
  the paper states this and that the relevant runs were repeated with the final code.

## What is not in the repository

* The raw per-element sample files (`proxy2_sample_*.npz`, about 150 MB) of Experiment 21 were left out because of their size. Every statistic of the
  paper is computed from the smaller `proxy2_diag_*.jsonl` files that are included.
* Model checkpoints and the CIFAR data (downloaded automatically by the training scripts).

## License

Apache License 2.0 (see `LICENSE`).
