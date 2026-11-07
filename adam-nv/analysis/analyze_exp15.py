"""Exp. 15 analysis: Adam versus Adam-NV at label-noise rates 10% and 40% (seeds 0-5).

For each noise rate (paired by seed):
  adam      Adam, beta1 = 0.9
  adam_nv   Adam-NV
Primary metric: last-5-epoch mean of train_corrupt_noisy_label_acc (memorization), in pp.
Secondary: validation accuracy, clean-subset accuracy and peak validation accuracy.
M_r = adam_nv - adam at noise rate r. Holm over the two primary contrasts {M_0.1, M_0.4}.
Reference: at 20% noise (Exp. 7, 8 seeds) M = +3.91 pp.

Pre-registered rule (written before any Exp. 15 run), applied at each rate:
  * EFFECT PRESENT : the 95% CI of M_r is above 0
  * NO EFFECT      : the 95% CI of M_r includes 0
  * REVERSED       : the 95% CI of M_r is below 0
Overall: the memorization effect GENERALIZES across noise rates if it is present at both rates.

Example:
    python3 analyze_exp15.py --ln01-dirs logs_exp15_ln01_A logs_exp15_ln01_B \
        --ln04-dirs logs_exp15_ln04_A logs_exp15_ln04_B
"""

import argparse
import os

import numpy as np
import pandas as pd

import analyze_exp07 as e7  # same folder

NAME = "{opt}_scheduleconstant_sigma0.0_ln{ln}_seed{seed}.csv"


def load_arm(opt, ln, dirs, seeds):
    out = {m: {} for m in e7.METRICS}
    for s in seeds:
        for d in dirs:
            p = os.path.join(d, NAME.format(opt=opt, ln=ln, seed=s))
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
    ap.add_argument("--ln01-dirs", nargs="+", default=[])
    ap.add_argument("--ln04-dirs", nargs="+", default=[])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    args = ap.parse_args()

    results = {}
    for ln, dirs in [("0.1", args.ln01_dirs), ("0.4", args.ln04_dirs)]:
        A = load_arm("adam", ln, dirs, args.seeds)
        N = load_arm("adam_nv", ln, dirs, args.seeds)
        print(f"\n######## label noise {float(ln):.0%} ########")
        for m, (title, _) in e7.METRICS.items():
            print(f"\n=== {title} ===")
            table = {"adam": A[m], "adam_nv": N[m]}
            for k, t in table.items():
                v = np.array([t[s] for s in sorted(t)])
                if len(v) > 1:
                    print(f"  {k:8s} n={len(v)} mean={v.mean():.2f} sd={v.std(ddof=1):.2f}")
                else:
                    print(f"  {k:8s} n={len(v)} (not available)")
            if min(len(t) for t in table.values()) > 1:
                r = e7.contrast(table, "adam_nv", "adam")
                print(f"  Adam-NV - Adam: {r['mean']:+.3f} pp  95% CI [{r['lo']:+.3f}, {r['hi']:+.3f}]  p={r['p']:.4f}  "
                      f"sign-flip p={r['pflip']:.4f}  higher in {r['up']}/{r['n']}")
                if m == "mem5":
                    results[ln] = r

    print("\n=== Pre-registered decision (primary metric) ===")
    if set(results) != {"0.1", "0.4"}:
        print("  Not enough complete arms/seeds at both noise rates to apply the rule.")
        return
    adj = e7.holm([results["0.1"]["p"], results["0.4"]["p"]])
    verdicts = {}
    for (ln, r), ph in zip(results.items(), adj):
        verdicts[ln] = "EFFECT PRESENT" if r["lo"] > 0 else ("REVERSED" if r["hi"] < 0 else "NO EFFECT")
        print(f"  noise {float(ln):.0%}: M = {r['mean']:+.2f} pp [{r['lo']:+.2f}, {r['hi']:+.2f}], p_Holm = {ph:.4f} -> {verdicts[ln]} "
              f"({100 * r['mean'] / 3.91:.0f}% of the 20%-noise effect of +3.91 pp)")
    if all(v == "EFFECT PRESENT" for v in verdicts.values()):
        print("  OUTCOME: the memorization effect GENERALIZES across noise rates (present at 10% and 40%).")
    elif any(v == "EFFECT PRESENT" for v in verdicts.values()):
        print("  OUTCOME: the effect is present at one noise rate only; report the dependence on the rate.")
    else:
        print("  OUTCOME: the effect is not detected at 10% or 40%; it may be specific to 20% noise.")


if __name__ == "__main__":
    main()
