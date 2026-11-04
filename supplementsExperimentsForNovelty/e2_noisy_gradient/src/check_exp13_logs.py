"""Verify completeness and provenance of Exp. 13 logs (Adam-NV tensor-mean control).

In addition to the Exp. 7 checks (CSV with --epochs rows; metadata arguments: adam_nv, constant
schedule, sigma 0, label noise 0.2, batch 128, deterministic, real data, default lr, software stack)
it checks, for every seed, that
  * tensormean_config_seed<k>.json exists and records the requested variant and seed,
  * beta1_dist_<variant>_seed<k>.jsonl and beta1_tensors_<variant>_seed<k>.jsonl exist with one
    record per epoch.
Exit code 1 if any check fails.

Example:
    python3 check_exp13_logs.py --log-dir ../logs_exp13_tm_A --variant tensor_mean --seeds 0 1 2 3
"""

import argparse
import json
import os
import sys

from check_exp07_logs import check_job


def _lines(path):
    with open(path) as fh:
        return sum(1 for line in fh if line.strip())


def check_seed(log_dir, seed, epochs, variant):
    problems = check_job(log_dir, "adam_nv", seed, epochs, None)
    cfg_path = os.path.join(log_dir, f"tensormean_config_seed{seed}.json")
    if not os.path.isfile(cfg_path):
        problems.append("missing tensormean_config sidecar")
    else:
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        if cfg.get("variant") != variant:
            problems.append(f"sidecar variant {cfg.get('variant')!r}, expected {variant!r}")
        if cfg.get("seed") != seed:
            problems.append(f"sidecar seed {cfg.get('seed')!r}, expected {seed}")
    for kind in ("dist", "tensors"):
        path = os.path.join(log_dir, f"beta1_{kind}_{variant}_seed{seed}.jsonl")
        if not os.path.isfile(path):
            problems.append(f"missing beta1 {kind} log")
        elif _lines(path) != epochs:
            problems.append(f"beta1 {kind} log has {_lines(path)} records, expected {epochs}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--variant", required=True, choices=["tensor_mean", "none"])
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    args = ap.parse_args()
    failed = 0
    print(f"{'job':26s} result")
    for seed in args.seeds:
        job = f"adam_nv_{args.variant}:{seed}"
        problems = check_seed(args.log_dir, seed, args.epochs, args.variant)
        print(f"{job:26s} {'OK' if not problems else 'FAIL: ' + '; '.join(problems)}")
        failed += bool(problems)
    print(f"\n{len(args.seeds) - failed}/{len(args.seeds)} jobs verified")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
