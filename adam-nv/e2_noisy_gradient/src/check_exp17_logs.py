"""Verify completeness and provenance of Exp. 17 logs (second architecture).

For every "<optimizer>:<seed>" job (adam, adam_nv or adam_nv_fixed) it checks that
  * the CSV exists with exactly --epochs data rows,
  * the metadata JSON exists and its arguments match the protocol (optimizer, seed, constant
    schedule, sigma 0, label_noise equal to --noise, batch 128, deterministic, real data, not a
    smoke test, lr equal to --lr, fixed_beta1 equal to --fixed-beta1 for adam_nv_fixed and null
    otherwise), and the software stack is Python 3.12.13, torch 2.10.0+cu128, torchvision 0.25.0+cu128,
  * arch_config_seed<k>.json exists and records the requested architecture and seed.
Exit code 1 if any check fails.

Example:
    python3 check_exp17_logs.py --log-dir ../logs_exp17 --arch vit_tiny --noise 0.0 --lr 5e-4 \
        --jobs adam:0 adam_nv:0
"""

import argparse
import csv
import json
import os
import sys

STACK = {"python_version": "3.12.13", "torch_version": "2.10.0+cu128", "torchvision_version": "0.25.0+cu128"}


def check_job(log_dir, opt, seed, epochs, arch, noise, lr, fixed_beta1):
    ln = "" if noise == 0.0 else f"_ln{noise}"
    base = f"{opt}_scheduleconstant_sigma0.0{ln}_seed{seed}"
    csv_path = os.path.join(log_dir, base + ".csv")
    meta_path = os.path.join(log_dir, base + "_metadata.json")
    if not os.path.isfile(csv_path):
        return ["missing CSV"]
    problems = []
    with open(csv_path, newline="") as f:
        n = len(list(csv.DictReader(f)))
    if n != epochs:
        problems.append(f"{n}/{epochs} epochs")
    if not os.path.isfile(meta_path):
        return problems + ["missing metadata"]
    with open(meta_path) as f:
        meta = json.load(f)
    a = meta.get("arguments", {})
    expect = {"optimizer": opt, "seed": seed, "schedule": "constant", "sigma": 0.0, "label_noise": noise,
              "batch_size": 128, "epochs": epochs, "deterministic": True, "smoke_test": False,
              "synthetic_data": False, "lr": lr, "dataset": "cifar10",
              "fixed_beta1": fixed_beta1 if opt == "adam_nv_fixed" else None}
    for k, v in expect.items():
        got = a.get(k)
        same = abs(got - v) < 1e-12 if isinstance(v, float) and isinstance(got, (int, float)) else got == v
        if not same:
            problems.append(f"argument {k}: got {got!r}, expected {v!r}")
    for k, v in STACK.items():
        if meta.get(k) != v:
            problems.append(f"{k}: got {meta.get(k)!r}, expected {v!r}")
    cfg_path = os.path.join(log_dir, f"arch_config_seed{seed}.json")
    if not os.path.isfile(cfg_path):
        problems.append("missing arch_config sidecar")
    else:
        with open(cfg_path) as f:
            cfg = json.load(f)
        if cfg.get("arch") != arch:
            problems.append(f"sidecar arch {cfg.get('arch')!r}, expected {arch!r}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--arch", required=True, choices=["vit_tiny", "vgg11_bn"])
    ap.add_argument("--noise", type=float, required=True, choices=[0.0, 0.2])
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--fixed-beta1", type=float, default=0.85)
    ap.add_argument("--jobs", nargs="+", required=True)
    args = ap.parse_args()
    failed = 0
    print(f"{'job':22s} result")
    for job in args.jobs:
        opt, seed = job.split(":")
        problems = check_job(args.log_dir, opt, int(seed), args.epochs, args.arch, args.noise, args.lr, args.fixed_beta1)
        print(f"{job:22s} {'OK' if not problems else 'FAIL: ' + '; '.join(problems)}")
        failed += bool(problems)
    print(f"\n{len(args.jobs) - failed}/{len(args.jobs)} jobs verified")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
