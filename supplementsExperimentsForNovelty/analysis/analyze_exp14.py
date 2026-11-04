"""Exp. 14 analysis: Adam-NV controller on a subset of tensors under 20% label noise (seeds 0-7).

Arms (paired by seed):
  adam      Adam, beta1 = 0.9                              (Label_Diagnostic 0-3 + Exp. 7 seeds 4-7)
  adam_nv   Adam-NV on all tensors                         (Label_Diagnostic 0-3 + Exp. 7 seeds 4-7)
  onedim    controller only on tensors of rank <= 1        (Exp. 14; 41 of 62 tensors, 9,610 elements)
  rest      controller only on tensors of rank >= 2        (Exp. 14; 21 tensors, 99.9% of the elements)

Primary metric: last-5-epoch mean of train_corrupt_noisy_label_acc (memorization), in pp.
M = adam_nv - adam; F1 = onedim - adam, F2 = rest - adam; D1 = adam_nv - onedim, D2 = adam_nv - rest.
Holm over {F1, F2, D1, D2}.

Pre-registered rule (written before any Exp. 14 run):
  0. If the 95% CI of M is not above 0, the rule does not apply.
  1. ONE-DIMENSIONAL TENSORS DRIVE THE EFFECT: F1 >= 0.75*M, the 95% CI of D1 includes 0, and F2 < 0.5*M.
  2. CONVOLUTION AND CLASSIFIER TENSORS DRIVE THE EFFECT: F2 >= 0.75*M, the 95% CI of D2 includes 0,
     and F1 < 0.5*M.
  3. otherwise BOTH SUBSETS CONTRIBUTE (or neither is sufficient); report F1, F2 and F1 + F2 against M.

Example:
    python3 analyze_exp14.py --orig-dir Label_Diagnostic --exp07-dirs logs_exp07_A logs_exp07_B logs_exp07_C \
        --onedim-dirs logs_exp14_onedim_A logs_exp14_onedim_C --rest-dirs logs_exp14_rest_B logs_exp14_rest_C
"""

import argparse
import os

import numpy as np
import pandas as pd

import analyze_exp07 as e7  # same folder


def load_arm(opt, dirs, seeds):
    out = {m: {} for m in e7.METRICS}
    for s in seeds:
        for d in dirs:
            p = os.path.join(d, e7.NAME.format(opt=opt, seed=s))
            if os.path.isfile(p):
                df = pd.read_csv(p)
                if len(df) != 60:
                    print(f"[skip] {p}: {len(df)}/60 epochs")
                    break
                for m, (_, fn) in e7.METRICS.items():
                    out[m][s] = fn(df)
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-dir", required=True)
    ap.add_argument("--exp07-dirs", nargs="+", required=True)
    ap.add_argument("--onedim-dirs", nargs="+", default=[])
    ap.add_argument("--rest-dirs", nargs="+", default=[])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    args = ap.parse_args()

    base = [args.orig_dir] + args.exp07_dirs
    A = load_arm("adam", base, args.seeds)
    N = load_arm("adam_nv", base, args.seeds)
    O = load_arm("adam_nv", args.onedim_dirs, args.seeds)
    R = load_arm("adam_nv", args.rest_dirs, args.seeds)
    label = {"adam": "Adam (0.90)", "adam_nv": "Adam-NV (all tensors)", "onedim": "controller on rank<=1 only",
             "rest": "controller on rank>=2 only"}

    for m, (title, _) in e7.METRICS.items():
        print(f"\n=== {title} ===")
        table = {"adam": A[m], "adam_nv": N[m], "onedim": O[m], "rest": R[m]}
        for k, t in table.items():
            v = np.array([t[s] for s in sorted(t)])
            if len(v) > 1:
                print(f"  {label[k]:28s} n={len(v)} mean={v.mean():.2f} sd={v.std(ddof=1):.2f}")
            else:
                print(f"  {label[k]:28s} n={len(v)} (not available)")
        pairs = [("adam_nv", "adam"), ("onedim", "adam"), ("rest", "adam"), ("adam_nv", "onedim"), ("adam_nv", "rest")]
        if min(len(t) for t in table.values()) > 1:
            rs = [e7.contrast(table, a, b) for a, b in pairs]
            fam = rs[1:]
            adj = dict(zip([id(r) for r in fam], e7.holm([r["p"] for r in fam])))
            for (a, b), r in zip(pairs, rs):
                ph = adj.get(id(r))
                ph_txt = f"  p_Holm={ph:.4f}" if ph is not None else "  (reference contrast)"
                print(f"  {label[a]} - {label[b]}: {r['mean']:+.3f} pp  95% CI [{r['lo']:+.3f}, {r['hi']:+.3f}]  "
                      f"p={r['p']:.4f}{ph_txt}  sign-flip p={r['pflip']:.4f}  higher in {r['up']}/{r['n']}")

    print("\n=== Pre-registered decision (primary metric) ===")
    table = {"adam": A["mem5"], "adam_nv": N["mem5"], "onedim": O["mem5"], "rest": R["mem5"]}
    if min(len(t) for t in table.values()) < 2:
        print("  Not enough complete arms/seeds to apply the rule.")
        return
    M = e7.contrast(table, "adam_nv", "adam")
    F1 = e7.contrast(table, "onedim", "adam")
    F2 = e7.contrast(table, "rest", "adam")
    D1 = e7.contrast(table, "adam_nv", "onedim")
    D2 = e7.contrast(table, "adam_nv", "rest")
    print(f"  M = {M['mean']:+.2f} pp [{M['lo']:+.2f}, {M['hi']:+.2f}]; F1 = {F1['mean']:+.2f} ({100 * F1['mean'] / M['mean']:.0f}% of M); "
          f"F2 = {F2['mean']:+.2f} ({100 * F2['mean'] / M['mean']:.0f}% of M); F1 + F2 = {F1['mean'] + F2['mean']:+.2f}")
    print(f"  D1 = {D1['mean']:+.2f} [{D1['lo']:+.2f}, {D1['hi']:+.2f}]; D2 = {D2['mean']:+.2f} [{D2['lo']:+.2f}, {D2['hi']:+.2f}]")
    if M["lo"] <= 0:
        print("  Adam-NV effect not reproduced in this pooled set; the rule does not apply.")
    elif F1["mean"] >= 0.75 * M["mean"] and D1["lo"] <= 0 <= D1["hi"] and F2["mean"] < 0.5 * M["mean"]:
        print("  OUTCOME: ONE-DIMENSIONAL TENSORS DRIVE THE EFFECT.")
    elif F2["mean"] >= 0.75 * M["mean"] and D2["lo"] <= 0 <= D2["hi"] and F1["mean"] < 0.5 * M["mean"]:
        print("  OUTCOME: CONVOLUTION AND CLASSIFIER TENSORS DRIVE THE EFFECT.")
    else:
        print("  OUTCOME: BOTH SUBSETS CONTRIBUTE (or neither is sufficient); compare F1 + F2 with M.")


if __name__ == "__main__":
    main()
