"""Exp. 20: constant beta1 on the one-dimensional tensors only (plain Adam elsewhere) vs the adaptive controller on the same tensors
(Exp. 14 'onedim' arm), ResNet-18, 20% label noise, seeds 0-7, paired by seed. Metric: last-5-epoch memorization (pp).
F1 = onedim(adaptive) - adam (from Exp. 14); C = const1d - adam; E = onedim(adaptive) - const1d.
Holm family (pre-registered): {C, E}. F1 is the Exp. 14 reference and is outside the family.
Pre-registered rule (requires F1's CI above 0):
  LOWERING beta1 ON THESE TENSORS SUFFICES : C >= 0.75 F1 and the 95% CI of E includes 0
  ADAPTIVITY NEEDED                        : C < 0.5 F1 and the 95% CI of E is above 0
  otherwise PARTLY / INCONCLUSIVE.
Example:
  python3 analyze_exp20.py --adam-dirs Label_Diagnostic logs_exp07_A logs_exp07_B --onedim-dirs logs_exp14_onedim_A logs_exp14_onedim_C --const-dirs logs_exp20_onedim_c079_A logs_exp20_onedim_c079_B
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
    ap.add_argument("--adam-dirs", nargs="+", required=True); ap.add_argument("--onedim-dirs", nargs="+", required=True)
    ap.add_argument("--const-dirs", nargs="+", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    a = ap.parse_args()
    A = load("adam", a.adam_dirs, a.seeds); O = load("adam_nv", a.onedim_dirs, a.seeds); C = load("adam_nv", a.const_dirs, a.seeds)
    print("seeds: adam", sorted(A), "| onedim adaptive", sorted(O), "| const 1-D", sorted(C))
    for m in ("mem5", "val5", "clean5"):
        fn = e7.METRICS[m][1]; t = {"adam": {s: fn(d) for s, d in A.items()}, "one": {s: fn(d) for s, d in O.items()}, "c": {s: fn(d) for s, d in C.items()}}
        F1 = e7.contrast(t, "one", "adam"); Cc = e7.contrast(t, "c", "adam"); E = e7.contrast(t, "one", "c")
        adj = e7.holm([Cc["p"], E["p"]]); print(f"\n{e7.METRICS[m][0]}")
        print(f"  F1 onedim(adaptive) - Adam  : {F1['mean']:+.3f} [{F1['lo']:+.3f},{F1['hi']:+.3f}] (reference, outside the family)")
        print(f"  C  const 1-D - Adam         : {Cc['mean']:+.3f} [{Cc['lo']:+.3f},{Cc['hi']:+.3f}] p={Cc['p']:.4f} Holm={adj[0]:.4f} higher in {Cc['up']}/{Cc['n']}")
        print(f"  E  onedim(adaptive) - const : {E['mean']:+.3f} [{E['lo']:+.3f},{E['hi']:+.3f}] p={E['p']:.4f} Holm={adj[1]:.4f} higher in {E['up']}/{E['n']}")
        if m == "mem5":
            if not F1["lo"] > 0: out = "Exp. 14 onedim effect not reproduced on these seeds (F1 CI not above 0); the control question is moot"
            elif Cc["mean"] >= 0.75 * F1["mean"] and E["lo"] <= 0 <= E["hi"]: out = "LOWERING beta1 ON THESE TENSORS SUFFICES"
            elif Cc["mean"] < 0.5 * F1["mean"] and E["lo"] > 0: out = "ADAPTIVITY NEEDED"
            else: out = "PARTLY / INCONCLUSIVE"
            print(f"  C/F1 = {Cc['mean']/F1['mean']:.2f}\n  PRE-REGISTERED OUTCOME: {out}")


if __name__ == "__main__":
    main()
