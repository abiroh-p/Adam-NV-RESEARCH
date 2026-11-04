# Environments

## Declared (requirements.txt)
`torch>=2.2,<3`, `torchvision>=0.17,<1`, `adabelief-pytorch==0.2.1`, `numpy>=1.24,<3`,
`pandas>=2.0,<3`, `scipy>=1.10,<2`, `matplotlib>=3.7,<4`.

## Environment the experiments ran in (from run metadata; Kaggle, 2x T4)
`torch 2.10.0+cu128`, `torchvision 0.25.0+cu128` (also the defaults hard-coded in
`adam-nv/e1_convergence/src/check_logs.py`). Python/CUDA details are in each `*_metadata.json`.

## Environment used to validate this repair (CPU only, NO GPU)
Python 3.12.3, torch 2.4.1+cu121 (CPU execution), torchvision 0.19.1, numpy 2.0.2, pandas 2.3.3,
scipy 1.17.1, matplotlib 3.10.8, adabelief-pytorch 0.2.1.
This is NOT the experiment environment. Unit tests and statistics are numerically independent of
the GPU; no training run was reproduced here, so bit-for-bit agreement of training logs on the
original hardware stack is untested.

## Reproduce the validation environment
    python -m venv .venv && . .venv/bin/activate
    pip install "torch==2.4.1" "torchvision==0.19.1" "numpy<2.1" "pandas>=2.0,<3" "scipy>=1.10,<2" \
                matplotlib adabelief-pytorch==0.2.1
    python tools/run_all_tests.py
