"""Exp. 13 analysis: tensor-mean beta_1 control for Adam-NV under 20% label noise (seeds 0-7).

Arms (paired by seed):
  adam         Adam, beta1 = 0.9                              (Label_Diagnostic 0-3 + Exp. 7 seeds 4-7)
  adam_nv      Adam-NV                                        (Label_Diagnostic 0-3 + Exp. 7 seeds 4-7)
  tensor_mean  Adam-NV with every element of a tensor set to that tensor's mean beta_1  (Exp. 13)
  step, fixed  Adam-NV with beta_1 permuted within each tensor (Exp. 11), for comparison only
  none         Adam-NV through the Exp. 13 wrapper without change (Exp. 13, seeds 0-1): canary and
               source of per-tensor beta_1 statistics.

Primary metric: last-5-epoch mean of train_corrupt_noisy_label_acc (memorization), in pp.
M = adam_nv - adam; F = tensor_mean - adam; D = adam_nv - tensor_mean.
Holm over {F, D, tensor_mean - step, tensor_mean - fixed}.

Pre-registered rule (written before any Exp. 13 run):
  0. If the 95% CI of M is not above 0, the rule does not apply.
  1. TENSOR-LEVEL ADAPTATION SUFFICES   : F >= 0.75*M and the 95% CI of D includes 0
                                          (within-tensor spread of beta_1 is not needed).
  2. WITHIN-TENSOR SPREAD MATTERS        : F < 0.5*M and the 95% CI of D is above 0.
  3. otherwise PARTIAL.
Reading together with Exp. 11 (permutation within tensors kept 61-72% of M):
  rule 1 -> the effect is layer-level adaptation: the controller lowers beta_1 on some tensors;
  rule 2 -> the within-tensor structure carries a substantial part of the effect;
  rule 3 -> both levels contribute; report all contrasts.

Per-tensor statistics: from the `none` runs, beta_1 means by tensor rank (1-D tensors such as
normalization parameters and biases, 2-D, 4-D convolution tensors) at epochs 1, 20, 40 and 60.

Example:
    python3 analyze_exp13.py --orig-dir Label_Diagnostic --exp07-dirs logs_exp07_A logs_exp07_B logs_exp07_C \
        --tm-dirs logs_exp13_tm_A logs_exp13_tm_B --none-dirs logs_exp13_none_C \
        --step-dirs logs_exp11_step_A logs_exp11_step_C --fixed-dirs logs_exp11_fixed_B logs_exp11_fixed_C
"""

import argparse
import glob
import json
import os
from collections import defaultdict

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


def canary(orig_dirs, none_dirs):
    print("\nCanary: Exp. 13 wrapper with variant=none against the existing Adam-NV logs")
    for d in none_dirs:
        for f in sorted(glob.glob(os.path.join(d, "adam_nv_scheduleconstant_sigma0.0_ln0.2_seed?.csv"))):
            base = os.path.basename(f)
            ref = next((os.path.join(o, base) for o in orig_dirs if os.path.isfile(os.path.join(o, base))), None)
            if ref is None:
                print(f"  {base}: no reference log")
                continue
            a, b = pd.read_csv(ref), pd.read_csv(f)
            cols = [c for c in a.columns if c != "epoch_time_sec" and c in b.columns]
            if len(a) != len(b):
                print(f"  {base}: different length {len(a)} vs {len(b)}")
                continue
            diff = float(np.nanmax(np.abs(a[cols].to_numpy(float) - b[cols].to_numpy(float))))
            print(f"  {base}: max abs difference over {len(cols)} columns = {diff:.3e} "
                  f"({'IDENTICAL' if diff == 0 else 'NOT identical'})")


def tensor_summary(dirs):
    files = [f for d in dirs for f in glob.glob(os.path.join(d, "beta1_tensors_none_seed?.jsonl"))]
    if not files:
        return
    runs = [[json.loads(x) for x in open(f) if x.strip()] for f in files]
    n = min(len(r) for r in runs)
    print(f"\nPer-tensor beta_1 of the controller by tensor rank (mean over {len(runs)} runs)")
    print("  epoch  rank  tensors     elements  mean of tensor means  element-weighted  lowest tensor mean")
    for e in [1, 20, 40, 60]:
        if e > n:
            continue
        agg = defaultdict(lambda: [0, 0, [], 0.0, []])
        for run in runs:
            for t in run[e - 1]["tensors"]:
                k = len(t["shape"]) if t["shape"] is not None else -1
                a = agg[k]
                a[0] += 1
                a[1] += t["numel"]
                a[2].append(t["mean"])
                a[3] += t["mean"] * t["numel"]
                a[4].append(t["mean"])
        for k, a in sorted(agg.items()):
            print(f"  {e:5d} {k:5d} {a[0] // len(runs):8d} {a[1] // len(runs):12d} "
                  f"{np.mean(a[2]):21.4f} {a[3] / a[1]:17.4f} {min(a[4]):19.4f}")
        all_t = [t for run in runs for t in run[e - 1]["tensors"]]
        print(f"  {e:5d}   all {len(all_t) // len(runs):8d} {sum(t['numel'] for t in all_t) // len(runs):12d} "
              f"{np.mean([t['mean'] for t in all_t]):21.4f} "
              f"{sum(t['mean'] * t['numel'] for t in all_t) / sum(t['numel'] for t in all_t):17.4f} "
              f"{min(t['mean'] for t in all_t):19.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-dir", required=True)
    ap.add_argument("--exp07-dirs", nargs="+", required=True)
    ap.add_argument("--tm-dirs", nargs="+", default=[])
    ap.add_argument("--none-dirs", nargs="+", default=[])
    ap.add_argument("--step-dirs", nargs="+", default=[])
    ap.add_argument("--fixed-dirs", nargs="+", default=[])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    args = ap.parse_args()

    base = [args.orig_dir] + args.exp07_dirs
    A = load_arm("adam", base, args.seeds)
    N = load_arm("adam_nv", base, args.seeds)
    T = load_arm("adam_nv", args.tm_dirs, args.seeds)
    S = load_arm("adam_nv", args.step_dirs, args.seeds)
    F = load_arm("adam_nv", args.fixed_dirs, args.seeds)
    label = {"adam": "Adam (0.90)", "adam_nv": "Adam-NV", "tm": "Adam-NV, tensor-mean",
             "step": "Adam-NV, shuffle/step", "fixed": "Adam-NV, shuffle/fixed"}

    for m, (title, _) in e7.METRICS.items():
        print(f"\n=== {title} ===")
        table = {"adam": A[m], "adam_nv": N[m], "tm": T[m], "step": S[m], "fixed": F[m]}
        for k, t in table.items():
            v = np.array([t[s] for s in sorted(t)])
            if len(v) > 1:
                print(f"  {label[k]:24s} n={len(v)} mean={v.mean():.2f} sd={v.std(ddof=1):.2f}")
            else:
                print(f"  {label[k]:24s} n={len(v)} (not available)")
        pairs = [("adam_nv", "adam"), ("tm", "adam"), ("adam_nv", "tm"), ("tm", "step"), ("tm", "fixed")]
        if min(len(table[k]) for k in ("adam", "adam_nv", "tm")) > 1:
            rs = []
            for a, b in pairs:
                rs.append(e7.contrast(table, a, b) if min(len(table[a]), len(table[b])) > 1 else None)
            fam = rs[1:]
            adj = dict(zip([id(r) for r in fam], e7.holm([r["p"] for r in fam]))) if all(fam) else {}
            for (a, b), r in zip(pairs, rs):
                if r is None:
                    print(f"  {label[a]} - {label[b]}: not enough paired seeds")
                    continue
                ph = adj.get(id(r))
                ph_txt = f"  p_Holm={ph:.4f}" if ph is not None else "  (reference contrast)"
                print(f"  {label[a]} - {label[b]}: {r['mean']:+.3f} pp  95% CI [{r['lo']:+.3f}, {r['hi']:+.3f}]  "
                      f"p={r['p']:.4f}{ph_txt}  sign-flip p={r['pflip']:.4f}  higher in {r['up']}/{r['n']}")

    canary(base, args.none_dirs)
    tensor_summary(args.none_dirs)

    print("\n=== Pre-registered decision (primary metric) ===")
    table = {"adam": A["mem5"], "adam_nv": N["mem5"], "tm": T["mem5"]}
    if min(len(t) for t in table.values()) < 2:
        print("  Not enough complete arms/seeds to apply the rule.")
        return
    M = e7.contrast(table, "adam_nv", "adam")
    Fc = e7.contrast(table, "tm", "adam")
    D = e7.contrast(table, "adam_nv", "tm")
    print(f"  M = {M['mean']:+.2f} pp [{M['lo']:+.2f}, {M['hi']:+.2f}]; F = {Fc['mean']:+.2f} pp "
          f"({100 * Fc['mean'] / M['mean']:.0f}% of M); D = {D['mean']:+.2f} pp [{D['lo']:+.2f}, {D['hi']:+.2f}]")
    if M["lo"] <= 0:
        print("  Adam-NV effect not reproduced in this pooled set; the rule does not apply.")
    elif Fc["mean"] >= 0.75 * M["mean"] and D["lo"] <= 0 <= D["hi"]:
        print("  OUTCOME: TENSOR-LEVEL ADAPTATION SUFFICES (F >= 0.75*M, D CI includes 0): the effect is layer-level.")
    elif Fc["mean"] < 0.5 * M["mean"] and D["lo"] > 0:
        print("  OUTCOME: WITHIN-TENSOR SPREAD MATTERS (F < 0.5*M, D CI above 0).")
    else:
        print("  OUTCOME: PARTIAL; both levels contribute. Report all contrasts.")


if __name__ == "__main__":
    main()
