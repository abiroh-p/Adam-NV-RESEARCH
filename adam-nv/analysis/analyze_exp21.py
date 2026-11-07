"""Exp. 21 analysis: does Adam-NV's score track gradient noise WITHIN tensors? (seeds are the unit of replication)

Per run and checkpoint: over the 20 convolution tensors (rank 4) the MEDIAN of the within-tensor Spearman correlations
  R     = r(s, qu)                 score vs direct noise fraction (unclamped, K = 64)
  rel   = split-half reliability of qu (Spearman-Brown), the empirical ceiling of the direct estimate
  R*    = R / sqrt(rel)            reliability-corrected correlation (only if rel >= 0.2, else unresolved)
  R0*   = r(s0, qu) / sqrt(rel)    the same for the lambda = 0 score
  Rrms  = r(s, rms)                score vs absolute gradient scale;   P* = partial r(s, qu | rms) / sqrt(rel)
Per seed: mean over the primary checkpoints (default epochs 30, 45, 60). Across seeds: mean and t-based 95% CI (n = number of seeds).

PRE-REGISTERED RULE (per label-noise condition):
  UNRESOLVED   : mean reliability < 0.2 (the direct estimate cannot rank elements within tensors; no verdict)
  TRACKS       : otherwise, lower 95% CI limit of mean R* >= 0.3
  DOES NOT TRACK : otherwise, upper 95% CI limit of mean R* < 0.1
  INCONCLUSIVE : otherwise.
  SCALE-DOMINATED : mean |Rrms| > mean R* (the score is more related to absolute gradient scale than to noise).
  lambda effect  : paired difference (R0* - R*) over seeds with its 95% CI (a CI above 0 means the lambda|m| term lowers tracking).
Between tensors (descriptive): Spearman over the 62 tensors of mean s with mean qu, mean s0 with mean qu, mean s with mean rms.
Pooled-over-elements correlations (the Exp. 16 statistic) are reported for comparison only.
Example:
  python3 analyze_exp21.py --log-dirs logs_exp21_A logs_exp21_B
"""
import argparse, json, os
import numpy as np
from scipy import stats

CK = [1, 5, 10, 20, 30, 45, 60]


def load(dirs, ln, seed):
    for d in dirs:
        p = os.path.join(d, f"proxy2_diag_ln{ln}_seed{seed}.jsonl")
        if os.path.isfile(p): return {r["epoch"]: r for r in (json.loads(x) for x in open(p) if x.strip())}
    return None


def med(rec, key, rank=4):
    v = [t[key] for t in rec["tensors"] if t["rank"] == rank and t[key] == t[key]]
    return float(np.median(v)) if v else float("nan")


def stats_row(rec):
    rel = med(rec, "reliability"); ok = rel == rel and rel >= 0.2; sq = np.sqrt(rel) if ok else float("nan")
    o = {"R": med(rec, "r_s_qu"), "rel": rel, "R0": med(rec, "r_s0_qu"), "Rrms": med(rec, "r_s_rms"), "P": med(rec, "pr_s_qu_given_rms"), "Rstream": med(rec, "r_s_qstream"),
         "pooled": rec["spearman_s_q16"] if "spearman_s_q16" in rec else float("nan"), "pooled_qu": rec["spearman_s_qu"], "mean_qu": rec["mean_qu"], "clamp16": rec["frac_q16_eq1"]}
    o["Rstar"] = o["R"] / sq; o["R0star"] = o["R0"] / sq; o["Pstar"] = o["P"] / sq
    tn = [t for t in rec["tensors"]]
    ms, mq, mr, m0 = (np.array([t[k] for t in tn]) for k in ("mean_s", "mean_qu", "mean_rms", "mean_s0"))
    o["between_s_qu"] = float(stats.spearmanr(ms, mq)[0]); o["between_s0_qu"] = float(stats.spearmanr(m0, mq)[0]); o["between_s_rms"] = float(stats.spearmanr(ms, mr)[0])
    return o


def ci(x):
    x = np.asarray([v for v in x if v == v], float); n = len(x)
    if n < 2: return (float(x.mean()) if n else float("nan"), float("nan"), float("nan"), n)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(n); t = stats.t.ppf(.975, n - 1); return m, m - t * se, m + t * se, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dirs", nargs="+", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    ap.add_argument("--conditions", nargs="+", default=["0.0", "0.2"]); ap.add_argument("--primary", type=int, nargs="+", default=[30, 45, 60])
    a = ap.parse_args()
    for ln in a.conditions:
        runs = {s: load(a.log_dirs, ln, s) for s in a.seeds}; runs = {s: r for s, r in runs.items() if r}
        print(f"\n================ label noise {float(ln):.0%}: seeds {sorted(runs)} ================")
        if not runs: continue
        S = {s: {e: stats_row(r[e]) for e in r} for s, r in runs.items()}
        print("checkpoint (mean over seeds): R(s,qu) rel R* | R0*(lambda=0) | Rrms P* | pooled(s,q16) pooled(s,qu) | mean qu | between-tensor(s,qu)")
        for e in CK:
            rows = [S[s][e] for s in S if e in S[s]]
            if not rows: continue
            f = lambda k: np.nanmean([r[k] for r in rows])
            print(f"  ep{e:2d}: R {f('R'):+.3f} rel {f('rel'):.3f} R* {f('Rstar'):+.3f} | R0* {f('R0star'):+.3f} | Rrms {f('Rrms'):+.3f} P* {f('Pstar'):+.3f} | pooled {f('pooled'):+.3f} {f('pooled_qu'):+.3f} | qu {f('mean_qu'):.3f} | between {f('between_s_qu'):+.3f}")
        per = {k: [np.nanmean([S[s][e][k] for e in a.primary if e in S[s]]) for s in S] for k in ("R", "rel", "Rstar", "R0star", "Rrms", "Pstar", "Rstream", "between_s_qu")}
        print(f"\nprimary checkpoints {a.primary}; per-seed values of R*: {np.round(per['Rstar'], 3).tolist()}")
        for k, lab in (("rel", "reliability"), ("R", "within-tensor R (uncorrected)"), ("Rstar", "R* (reliability-corrected)"), ("R0star", "R0* (lambda = 0)"), ("Rrms", "r(s, absolute scale)"), ("Pstar", "P* (partial given scale)"), ("Rstream", "r(s, stream estimate) uncorrected"), ("between_s_qu", "between tensors r(s,qu)")):
            m, lo, hi, n = ci(per[k]); print(f"  {lab:38s}: {m:+.3f} [{lo:+.3f},{hi:+.3f}] (n={n})")
        mrel = np.nanmean(per["rel"]); m, lo, hi, n = ci(per["Rstar"])
        if not mrel >= 0.2: out = "UNRESOLVED (mean reliability < 0.2)"
        elif lo >= 0.3: out = "TRACKS within tensors"
        elif hi < 0.1: out = "DOES NOT TRACK within tensors"
        else: out = "INCONCLUSIVE"
        dl = np.array(per["R0star"]) - np.array(per["Rstar"]); ml, ll, hl, _ = ci(dl)
        print(f"\n  PRE-REGISTERED OUTCOME: {out}")
        sd = (np.nanmean(np.abs(per['Rrms'])) > m) if m == m else "n/a (R* undefined)"
        print(f"  scale-dominated (mean |r(s, scale)| {np.nanmean(np.abs(per['Rrms'])):.3f} > mean R* {m:+.3f})? {sd}")
        print(f"  lambda effect (R0* - R*): {ml:+.3f} [{ll:+.3f},{hl:+.3f}]")


if __name__ == "__main__":
    main()
