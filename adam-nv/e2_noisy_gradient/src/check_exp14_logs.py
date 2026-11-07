"""Verify completeness and provenance of Exp. 14 logs (Adam-NV controller on a subset of tensors).

In addition to the Exp. 7 checks (CSV with --epochs rows; metadata arguments: adam_nv, constant
schedule, sigma 0, label noise 0.2, batch 128, deterministic, real data, default lr, software
stack) it checks that subset_config_seed<k>.json exists and records the requested subset and seed.
Exit code 1 if any check fails.

Example:
    python3 check_exp14_logs.py --log-dir ../logs_exp14_onedim_A --subset onedim --seeds 0 1 2
"""

import argparse
import json
import os
import sys

from check_exp07_logs import check_job


def check_seed(log_dir, seed, epochs, subset):
    problems = check_job(log_dir, "adam_nv", seed, epochs, None)
    path = os.path.join(log_dir, f"subset_config_seed{seed}.json")
    if not os.path.isfile(path):
        problems.append("missing subset_config sidecar")
    else:
        with open(path) as fh:
            cfg = json.load(fh)
        if cfg.get("subset") != subset:
            problems.append(f"sidecar subset {cfg.get('subset')!r}, expected {subset!r}")
        if cfg.get("seed") != seed:
            problems.append(f"sidecar seed {cfg.get('seed')!r}, expected {seed}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--subset", required=True, choices=["onedim", "rest"])
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    args = ap.parse_args()
    failed = 0
    print(f"{'job':26s} result")
    for seed in args.seeds:
        job = f"adam_nv_{args.subset}:{seed}"
        problems = check_seed(args.log_dir, seed, args.epochs, args.subset)
        print(f"{job:26s} {'OK' if not problems else 'FAIL: ' + '; '.join(problems)}")
        failed += bool(problems)
    print(f"\n{len(args.seeds) - failed}/{len(args.seeds)} jobs verified")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
