"""Verify completeness and provenance of Exp. 20 logs (constant beta_1 on a subset of tensors).

Exp. 7 checks (CSV with --epochs rows; metadata arguments: adam_nv, constant schedule, sigma 0,
label noise 0.2, batch 128, deterministic, real data, default lr, software stack) plus the sidecar
constsubset_config_seed<k>.json with the requested subset, constant and seed. Exit code 1 on any failure.

Example:
    python3 check_exp20_logs.py --log-dir ../logs_exp20_onedim_c079_A --subset onedim --const-beta1 0.79 --seeds 0 1 2 3
"""
import argparse
import json
import os
import sys

from check_exp07_logs import check_job


def check_seed(log_dir, seed, epochs, subset, c):
    problems = check_job(log_dir, "adam_nv", seed, epochs, None)
    path = os.path.join(log_dir, f"constsubset_config_seed{seed}.json")
    if not os.path.isfile(path):
        problems.append("missing constsubset_config sidecar")
    else:
        cfg = json.load(open(path))
        if cfg.get("subset") != subset:
            problems.append(f"sidecar subset {cfg.get('subset')!r}, expected {subset!r}")
        if abs(float(cfg.get("const_beta1", -1)) - c) > 1e-12:
            problems.append(f"sidecar const_beta1 {cfg.get('const_beta1')!r}, expected {c}")
        if cfg.get("seed") != seed:
            problems.append(f"sidecar seed {cfg.get('seed')!r}, expected {seed}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--subset", required=True, choices=["onedim", "rest"])
    ap.add_argument("--const-beta1", type=float, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    a = ap.parse_args()
    failed = 0
    print(f"{'job':30s} result")
    for s in a.seeds:
        pr = check_seed(a.log_dir, s, a.epochs, a.subset, a.const_beta1)
        print(f"{'adam_const_' + a.subset + ':' + str(s):30s} {'OK' if not pr else 'FAIL: ' + '; '.join(pr)}")
        failed += bool(pr)
    print(f"\n{len(a.seeds) - failed}/{len(a.seeds)} jobs verified")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
