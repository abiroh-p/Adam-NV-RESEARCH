"""Exp. 18: constant-beta1 sweep on the ViT-tiny under 20% label noise (seeds 0-4).

Reference arms come from Exp. 17 (same code, same seeds): adam and adam_nv in the Exp. 17 noisy folder and the
constant 0.85 control. New constants are given as VALUE=DIR pairs.
Metric: last-5-epoch memorization (pp). M = adam_nv - adam; for each constant c: F_c = const_c - adam, D_c = adam_nv - const_c.
Holm family (pre-registered): F_c and D_c for the CORE constants 0.70, 0.80, 0.95 (six contrasts). Other constants (0.85 from Exp. 17, 0.92)
are reported outside the family.
Pre-registered rule per constant (same thresholds as Exp. 7 and Exp. 17; requires M's 95% CI above 0):
  EXPLAINED BY THE CONSTANT      : F_c >= 0.5 M and the 95% CI of D_c includes 0
  MOSTLY EXPLAINED, SMALL RESIDUAL : F_c >= 0.5 M and the 95% CI of D_c is above 0
  ADAPTATION BEYOND THE CONSTANT : F_c < 0.25 M and the 95% CI of D_c is above 0
  otherwise INCONCLUSIVE.
Overall outcome = the outcome of the matched constant 0.80. Everything else is descriptive.
Example:
  python3 analyze_exp18.py --adam-dir exp17_logs_3 --const 0.85=logs_exp17_vit_noisy_2 0.70=logs_exp18_vit_b070 0.80=logs_exp18_vit_b080 0.95=logs_exp18_vit_b095
"""
import argparse, os
import numpy as np, pandas as pd
import analyze_exp07 as e7

NAME = "{opt}_scheduleconstant_sigma0.0_ln0.2_seed{seed}.csv"
mem5 = e7.METRICS["mem5"][1]; val5 = e7.METRICS["val5"][1]


def load(opt, d, seeds):
    out = {}
    for s in seeds:
        p = os.path.join(d, NAME.format(opt=opt, seed=s))
        if os.path.isfile(p):
            df = pd.read_csv(p)
            if len(df) == 60: out[s] = df
            else: print(f"[skip] {p}: {len(df)}/60 epochs")
    return out


def rule(M, F, D):
    if not M["lo"] > 0: return "effect not reproduced on the ViT (M CI not above 0); the control question is moot"
    if F["mean"] >= 0.5 * M["mean"] and D["lo"] <= 0 <= D["hi"]: return "EXPLAINED BY THE CONSTANT"
    if F["mean"] >= 0.5 * M["mean"] and D["lo"] > 0: return "MOSTLY EXPLAINED, SMALL RESIDUAL"
    if D["lo"] > 0 and F["mean"] < 0.25 * M["mean"]: return "ADAPTATION BEYOND THE CONSTANT"
    return "INCONCLUSIVE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adam-dir", required=True)
    ap.add_argument("--const", nargs="+", required=True, help="VALUE=DIR pairs, e.g. 0.80=logs_exp18_vit_b080")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    ap.add_argument("--core", type=float, nargs="+", default=[0.70, 0.80, 0.95])
    a = ap.parse_args()
    A = load("adam", a.adam_dir, a.seeds); N = load("adam_nv", a.adam_dir, a.seeds)
    consts = {}
    for kv in a.const:
        v, d = kv.split("="); consts[float(v)] = load("adam_nv_fixed", d, a.seeds)
    print("seeds:", {"adam": sorted(A), "adam_nv": sorted(N), **{f"const{c}": sorted(x) for c, x in consts.items()}})
    tab = {"adam": {s: mem5(d) for s, d in A.items()}, "adam_nv": {s: mem5(d) for s, d in N.items()}}
    for c, x in consts.items(): tab[f"c{c}"] = {s: mem5(d) for s, d in x.items()}
    for k, v in tab.items(): print(f"  {k:10s} n={len(v)} mean={np.mean(list(v.values())):.2f} sd={np.std(list(v.values()), ddof=1):.2f}")
    M = e7.contrast(tab, "adam_nv", "adam")
    print(f"\nM = Adam-NV - Adam: {M['mean']:+.2f} pp [{M['lo']:+.2f},{M['hi']:+.2f}] p={M['p']:.4f} up {M['up']}/{M['n']}")
    rows = []
    for c in sorted(consts):
        F = e7.contrast(tab, f"c{c}", "adam"); D = e7.contrast(tab, "adam_nv", f"c{c}"); rows.append((c, F, D))
    core = [(c, F, D) for c, F, D in rows if c in a.core]
    fam = [r for c, F, D in core for r in (F, D)]
    adj = dict(zip([id(r) for r in fam], e7.holm([r["p"] for r in fam]))) if len(fam) == 2 * len(a.core) else {}
    print("\nconstant   F_c=const-Adam [95% CI]  (Holm, family of core constants)   D_c=NV-const [95% CI] (Holm)   F_c/M   rule outcome")
    for c, F, D in rows:
        hf = f"{adj[id(F)]:.3f}" if id(F) in adj else "n/a"; hd = f"{adj[id(D)]:.3f}" if id(D) in adj else "n/a"
        print(f"  {c:4.2f}   {F['mean']:+.2f} [{F['lo']:+.2f},{F['hi']:+.2f}] ({hf})   {D['mean']:+.2f} [{D['lo']:+.2f},{D['hi']:+.2f}] ({hd})   {F['mean']/M['mean']:.2f}   {rule(M, F, D)}{'' if c in a.core else '   [outside Holm family]'}")
    if 0.80 in consts:
        c, F, D = [r for r in rows if r[0] == 0.80][0]; print(f"\nOVERALL OUTCOME (matched constant 0.80): {rule(M, F, D)}")
    else:
        print("\nOVERALL OUTCOME: the matched constant 0.80 is missing; the rule cannot be applied.")
    print("\nSecondary (descriptive): validation accuracy last-5, Adam-NV/constant minus Adam")
    V = {"adam": {s: val5(d) for s, d in A.items()}, "adam_nv": {s: val5(d) for s, d in N.items()}}
    for c, x in consts.items(): V[f"c{c}"] = {s: val5(d) for s, d in x.items()}
    for k in [k for k in V if k != "adam"]:
        r = e7.contrast(V, k, "adam"); print(f"  {k:8s}: {r['mean']:+.2f} [{r['lo']:+.2f},{r['hi']:+.2f}]")
    pts = [(c, tab[f"c{c}"][s]) for c in consts for s in tab[f"c{c}"]]
    if len(pts) > 3:
        x, y = np.array(pts).T; sl = np.polyfit(x, y, 1)[0]; print(f"\nDescriptive only: pooled slope of memorization on constant beta1 = {sl:+.1f} pp per unit (n={len(pts)} runs from {len(consts)} constants)")


if __name__ == "__main__":
    main()
