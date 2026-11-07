# Reproducing the results

## 1. Check the paper's numbers (no GPU, minutes)

```bash
pip install numpy pandas scipy
python tools/reproduce_paper_numbers.py
```

This unpacks the log archives of Experiments 7 and 14-21 into a temporary folder, extracts the paper's verification scripts from
`ResearchPaperSupplements/adamnv_paper.zip` and recomputes every number typed in the paper's sections on these experiments (means, paired differences,
t-based 95% intervals, Holm-adjusted p-values, shares, correlations). It prints one line per experiment and `ALL VERIFIERS PASS` if there are no mismatches.
Pass `--keep --work DIR` to keep the unpacked logs for your own analysis.

## 2. Re-run the statistics of one experiment

Each run book names its analysis script and command, for example
`python adam-nv/analysis/analyze_exp14.py --orig-dir ... --exp07-dirs ... --onedim-dirs ... --rest-dirs ...`.
Unpack the needed log archives first (`unzip adam-nv/e2_noisy_gradient/logs/<archive>.zip -d <folder>`); the run book lists the expected folders.

## 3. Run the unit tests

```bash
python tools/run_all_tests.py      # needs PyTorch, torchvision, adabelief-pytorch, numpy, pandas, scipy, matplotlib
```

## 4. Re-run an experiment (GPU)

Every run book contains the exact launch commands (`bash run_expNN_*.sh` with the environment variables it documents). The launchers check the software
environment against the one used here and refuse to start otherwise, record the SHA-256 of every source file they run, and verify the logs they write.
One 60-epoch ResNet-18 run takes about 65 minutes on a Tesla T4; a ViT-tiny run takes 40-55 minutes.

## 5. Verify that the code in the repository is the code that produced the logs

```bash
python tools/provenance.py verify
```

The paper's claim is narrower and stronger for Experiments 14-21: every `_code_hashes_*.txt` inside those log archives lists SHA-256 values that match a
source file in this repository byte for byte. `docs/EXPERIMENT_INDEX.md` lists which archive belongs to which experiment.
