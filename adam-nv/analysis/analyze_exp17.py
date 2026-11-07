"""Exp. 17 analysis: Adam, Adam-NV and a constant-beta1 control on a second architecture.

Run once per architecture. Arms (paired by seed, seeds 0-4): adam, adam_nv, adam_nv_fixed (beta1 = 0.85).
Settings: clean labels and 20% label noise (separate log folders).

Clean setting, primary metric: last-5-epoch mean validation accuracy (pp). Secondary: final-epoch
accuracy and peak accuracy.
Noisy setting, primary metric: last-5-epoch memorization (train_corrupt_noisy_label_acc, pp).
Contrasts per setting: NV - Adam, fixed - Adam, NV - fixed, Holm over the three.

Pre-registered rules (written before any Exp. 17 run). M = adam_nv - adam.
Clean accuracy:
  * ADAM-NV BETTER      : the 95% CI of M is above 0
  * ADAM-NV WORSE       : the 95% CI of M is below 0
  * NO DETECTABLE DIFF. : otherwise (flagged EQUIVALENT only if the whole CI lies within +-0.15 pp)
Memorization under label noise:
  * EFFECT REPRODUCED   : the CI of M is above 0 and F = fixed - adam < 0.5*M
  * EFFECT ALSO IN A CONSTANT beta1 : the CI of M is above 0 and F >= 0.5*M
  * NO EFFECT           : the CI of M includes 0
  * REVERSED            : the CI of M is below 0

Example:
    python3 analyze_exp17.py --clean-dirs logs_exp17_vit_clean_1 logs_exp17_vit_clean_2 \
        --noisy-dirs logs_exp17_vit_noisy_1 logs_exp17_vit_noisy_2
"""

import argparse
import os

import numpy as np
import pandas as pd

import analyze_exp07 as e7  # same folder

ARMS = ["adam", "adam_nv", "adam_nv_fixed"]
LABEL = {"adam": "Adam", "adam_nv": "Adam-NV", "adam_nv_fixed": "Adam const. beta1=0.85"}


def name(opt, noisy, seed):
    return f"{opt}_scheduleconstant_sigma0.0{'_ln0.2' if noisy else ''}_seed{seed}.csv"


def load(dirs, noisy, seeds):
    out = {a: {m: {} for m in e7.METRICS} for a in ARMS}
    for a in ARMS:
        for s in seeds:
            for d in dirs:
                p = os.path.join(d, name(a, noisy, s))
                if os.path.isfile(p):
                    df = pd.read_csv(p)
                    if len(df) != 60:
                        print(f"[skip] {p}: {len(df)}/60 epochs")
                        break
                    for m, (_, fn) in e7.METRICS.items():
                        if m == "mem5" and not noisy:
                            continue
                        if m in ("clean5",) and not noisy:
                            continue
                        out[a][m][s] = fn(df)
                    out[a].setdefault("final", {})[s] = float(df["val_acc"].iloc[-1] * 100)
                    break
    return out


def report(title, data, metrics):
    print(f"\n######## {title} ########")
    results = {}
    for m in metrics:
        label = e7.METRICS[m][0] if m in e7.METRICS else "Final-epoch validation accuracy (pp)"
        print(f"\n=== {label} ===")
        table = {a: data[a].get(m, {}) for a in ARMS}
        for a in ARMS:
            v = np.array([table[a][s] for s in sorted(table[a])])
            print(f"  {LABEL[a]:24s} n={len(v)}" + (f" mean={v.mean():.2f} sd={v.std(ddof=1):.2f}" if len(v) > 1 else " (not available)"))
        pairs = [("adam_nv", "adam"), ("adam_nv_fixed", "adam"), ("adam_nv", "adam_nv_fixed")]
        if min(len(table[a]) for a in ARMS) > 1:
            rs = [e7.contrast(table, a, b) for a, b in pairs]
            adj = e7.holm([r["p"] for r in rs])
            for (a, b), r, ph in zip(pairs, rs, adj):
                print(f"  {LABEL[a]} - {LABEL[b]}: {r['mean']:+.3f} pp  95% CI [{r['lo']:+.3f}, {r['hi']:+.3f}]  p={r['p']:.4f}  "
                      f"p_Holm={ph:.4f}  sign-flip p={r['pflip']:.4f}  higher in {r['up']}/{r['n']}")
            results[m] = rs
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dirs", nargs="+", default=[])
    ap.add_argument("--noisy-dirs", nargs="+", default=[])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    args = ap.parse_args()

    clean = load(args.clean_dirs, False, args.seeds) if args.clean_dirs else None
    noisy = load(args.noisy_dirs, True, args.seeds) if args.noisy_dirs else None
    rc = report("CLEAN LABELS", clean, ["val5", "final", "valpeak"]) if clean else {}
    rn = report("20% LABEL NOISE", noisy, ["mem5", "val5", "clean5", "valpeak"]) if noisy else {}

    print("\n=== Pre-registered decisions ===")
    if "val5" in rc:
        M = rc["val5"][0]
        verdict = "ADAM-NV BETTER" if M["lo"] > 0 else ("ADAM-NV WORSE" if M["hi"] < 0 else "NO DETECTABLE DIFFERENCE")
        if verdict == "NO DETECTABLE DIFFERENCE" and M["lo"] >= -0.15 and M["hi"] <= 0.15:
            verdict += " (EQUIVALENT within +-0.15 pp)"
        print(f"  clean accuracy: M = {M['mean']:+.3f} pp [{M['lo']:+.3f}, {M['hi']:+.3f}] -> {verdict}")
    else:
        print("  clean accuracy: not enough complete data.")
    if "mem5" in rn:
        M, F = rn["mem5"][0], rn["mem5"][1]
        if M["lo"] > 0 and F["mean"] < 0.5 * M["mean"]:
            verdict = "EFFECT REPRODUCED (not matched by the constant beta1)"
        elif M["lo"] > 0:
            verdict = "EFFECT ALSO IN A CONSTANT beta1 (F >= 0.5*M)"
        elif M["hi"] < 0:
            verdict = "REVERSED"
        else:
            verdict = "NO EFFECT"
        print(f"  memorization: M = {M['mean']:+.3f} pp [{M['lo']:+.3f}, {M['hi']:+.3f}], F = {F['mean']:+.3f} -> {verdict}")
    else:
        print("  memorization: not enough complete data.")


if __name__ == "__main__":
    main()
