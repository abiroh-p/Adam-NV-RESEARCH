"""Exp. 16 analysis: does Adam-NV's score track directly measured gradient noise?

Reads proxy_diag_ln<noise>_seed<k>.jsonl from the given folders and prints, per label-noise condition,
the mean over seeds of every diagnostic at each checkpoint, and then applies the pre-registered rule.

Quantities (per element, from K minibatch gradients at fixed weights):
  q      direct noise fraction = unbiased variance / mean square of the minibatch gradients
  s      Adam-NV's proxy from its moments m and v:  max(v - m^2, 0) / (v + lam*|m| + eps)
  scale  sqrt(mean square of the minibatch gradients) / sqrt(v)  (current scale relative to v)

Pre-registered rule (written before any Exp. 16 run). For each label-noise condition let
r = mean, over seeds and over checkpoints from epoch 5 on, of the pooled Spearman correlation between s and q.
  r >= 0.5          : the proxy TRACKS direct gradient noise
  0.2 <= r < 0.5    : it tracks it WEAKLY
  r < 0.2           : it DOES NOT track it
Confound check: with c = the same mean of the Spearman correlation between s and scale,
  |c| > r           : the proxy is MORE related to the gradient-scale ratio than to direct noise (SCALE-CONFOUNDED).
Descriptive additions (no threshold): mean s, mean q, and the correlations by tensor rank.

Example:
    python3 analyze_exp16.py --dirs logs_exp16_A logs_exp16_B
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

KEYS = ["mean_s", "mean_rho", "mean_q", "spearman_s_q", "spearman_rho_q", "spearman_s_scale", "spearman_q_scale",
        "median_conv_tensor_spearman_s_q", "spearman_s_q_rank1", "spearman_s_q_rank2", "spearman_s_q_rank4"]


def nanmean(x):
    x = [v for v in x if v is not None and v == v]
    return float(np.mean(x)) if x else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--min-epoch", type=int, default=5, help="first checkpoint used by the rule")
    args = ap.parse_args()

    runs = defaultdict(dict)  # ln -> seed -> records
    for d in args.dirs:
        for f in sorted(glob.glob(os.path.join(d, "proxy_diag_ln*_seed*.jsonl"))):
            m = re.search(r"proxy_diag_ln([0-9.]+)_seed(\d+)\.jsonl$", os.path.basename(f))
            runs[m.group(1)][int(m.group(2))] = [json.loads(x) for x in open(f) if x.strip()]
    if not runs:
        print("No diagnostic files found.")
        return

    summary = {}
    for ln in sorted(runs):
        seeds = sorted(runs[ln])
        print(f"\n######## label noise {float(ln):.0%} (seeds {seeds}) ########")
        epochs = sorted({r["epoch"] for recs in runs[ln].values() for r in recs})
        print("epoch " + " ".join(f"{k[:22]:>22s}" for k in KEYS[:8]))
        for e in epochs:
            row = []
            for k in KEYS[:8]:
                row.append(nanmean([r.get(k) for recs in runs[ln].values() for r in recs if r["epoch"] == e]))
            print(f"{e:5d} " + " ".join(f"{v:22.3f}" for v in row))
        print("by tensor rank (mean over seeds, checkpoints >= %d): " % args.min_epoch +
              ", ".join(f"rank{r}: {nanmean([rec.get(f'spearman_s_q_rank{r}') for recs in runs[ln].values() for rec in recs if rec['epoch'] >= args.min_epoch]):.3f}" for r in (1, 2, 4)))
        sel = [rec for recs in runs[ln].values() for rec in recs if rec["epoch"] >= args.min_epoch]
        summary[ln] = (nanmean([r["spearman_s_q"] for r in sel]), nanmean([r["spearman_s_scale"] for r in sel]),
                       nanmean([r["mean_q"] for r in sel]), nanmean([r["mean_s"] for r in sel]), len(seeds))

    print("\n=== Pre-registered decision ===")
    for ln, (r, c, mq, ms, n) in summary.items():
        verdict = "TRACKS direct gradient noise" if r >= 0.5 else ("tracks it WEAKLY" if r >= 0.2 else "DOES NOT track it")
        print(f"  label noise {float(ln):.0%}: r = Spearman(s, q) = {r:+.3f} ({n} seeds) -> {verdict}")
        print(f"      c = Spearman(s, scale) = {c:+.3f} -> " +
              ("SCALE-CONFOUNDED (|c| > r)" if abs(c) > r else "not more related to scale than to direct noise (|c| <= r)") +
              f"; mean q = {mq:.3f}, mean s = {ms:.3f}")


if __name__ == "__main__":
    main()
