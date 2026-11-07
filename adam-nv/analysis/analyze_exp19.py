"""Exp. 19: null run. Adam-NV code with gamma = 0 and beta1 = 0.9 (adam_nv_fixed --fixed-beta1 0.9) against torch Adam,
ResNet-18, 20% label noise, seeds 0-7, paired by seed.
Metric: last-5-epoch memorization (pp); also validation accuracy. N0 = null - adam.
Pre-registered rule (primary metric), equivalence margin +/-0.5 pp (about 13% of the +3.91 pp Adam-NV effect):
  CODE PATH EQUIVALENT TO ADAM : the whole 95% CI of the mean of N0 lies inside (-0.5, +0.5)
  CODE PATH DIFFERS            : the 95% CI excludes 0 and |mean| > 0.5 -> the Exp. 14 contrasts must be re-based on this null arm
  otherwise INCONCLUSIVE (the paired SD is reported as the trajectory floor; more seeds needed).
The paired SD of N0 is the trajectory-level noise floor of the code path and is compared with that of M (Adam-NV - Adam) when --nv-dirs is given.
Example:
  python3 analyze_exp19.py --adam-dirs Label_Diagnostic logs_exp07_A logs_exp07_B --null-dirs logs_exp19_null_A logs_exp19_null_B --nv-dirs Label_Diagnostic exp07_log_C/log_exp07_C
"""
import argparse, os
import numpy as np, pandas as pd
import analyze_exp07 as e7

NAME = "{opt}_scheduleconstant_sigma0.0_ln0.2_seed{seed}.csv"


def load(opt, dirs, seeds):
    out = {}
    for s in seeds:
        for d in dirs:
            p = os.path.join(d, NAME.format(opt=opt, seed=s))
            if os.path.isfile(p):
                df = pd.read_csv(p)
                if len(df) == 60: out[s] = df; break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adam-dirs", nargs="+", required=True); ap.add_argument("--null-dirs", nargs="+", required=True)
    ap.add_argument("--nv-dirs", nargs="*", default=[]); ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--null-opt", default="adam_nv_fixed", help="optimizer name in the null folders (adam_nv_fixed for Exp. 19)")
    a = ap.parse_args()
    A = load("adam", a.adam_dirs, a.seeds); Z = load(a.null_opt, a.null_dirs, a.seeds); N = load("adam_nv", a.nv_dirs, a.seeds) if a.nv_dirs else {}
    print("seeds: adam", sorted(A), "| null", sorted(Z), "| adam_nv", sorted(N))
    for m, (title, fn) in {"mem5": e7.METRICS["mem5"], "val5": e7.METRICS["val5"]}.items():
        t = {"adam": {s: fn(d) for s, d in A.items()}, "null": {s: fn(d) for s, d in Z.items()}}
        r = e7.contrast(t, "null", "adam"); print(f"\n{title}\n  null - Adam: {r['mean']:+.3f} pp  95% CI [{r['lo']:+.3f},{r['hi']:+.3f}]  p={r['p']:.4f}  sign-flip={r['pflip']:.4f}  higher in {r['up']}/{r['n']}")
        x = np.array([t['null'][s] - t['adam'][s] for s in sorted(set(t['null']) & set(t['adam']))]); print(f"  per-seed differences: {np.round(x, 2).tolist()}  (paired SD = trajectory floor = {x.std(ddof=1):.2f} pp)")
        if m == "mem5":
            if r["lo"] > -a.margin and r["hi"] < a.margin: out = "CODE PATH EQUIVALENT TO ADAM (CI inside the margin)"
            elif (r["lo"] > 0 or r["hi"] < 0) and abs(r["mean"]) > a.margin: out = "CODE PATH DIFFERS (re-base the Exp. 14 contrasts on this null arm)"
            else: out = "INCONCLUSIVE (interval not inside the margin; more seeds needed)"
            print(f"  PRE-REGISTERED OUTCOME (margin +/-{a.margin} pp): {out}")
            if N:
                t["nv"] = {s: fn(d) for s, d in N.items()}; M = e7.contrast(t, "nv", "adam")
                xm = np.array([t['nv'][s] - t['adam'][s] for s in sorted(set(t['nv']) & set(t['adam']))])
                print(f"  for scale: Adam-NV - Adam = {M['mean']:+.2f} pp, paired SD {xm.std(ddof=1):.2f}; null paired SD {x.std(ddof=1):.2f}; ratio null/NV SD = {x.std(ddof=1)/xm.std(ddof=1):.2f}")


if __name__ == "__main__":
    main()
