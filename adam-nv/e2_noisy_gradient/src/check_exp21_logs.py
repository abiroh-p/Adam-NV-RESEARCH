"""Verify completeness and provenance of Exp. 21 logs (noise-proxy diagnostic v2).

Per job (label_noise:seed): proxy2_train CSV with --epochs rows; proxy2_diag jsonl with one record per checkpoint, each with
k_batches, stream_steps, 62 tensor records, no NaN pooled fields; one proxy2_sample .npz per checkpoint; config json with the
requested seed, noise, K, epochs and deterministic flag. With --reference-dirs and 20% noise the validation accuracy must equal
that of the earlier Adam-NV run with the same seed at every epoch (the diagnostics must not change training). Exit code 1 on failure.

Example:
    python3 check_exp21_logs.py --log-dir ../logs_exp21_A --jobs 0.0:0 0.2:0 0.0:1 0.2:1 --reference-dirs ../Label_Diagnostic ../exp07_log_C/log_exp07_C
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

CHECKPOINTS = [1, 5, 10, 20, 30, 45, 60]


def check_job(log_dir, ln, seed, epochs, k, refs):
    tag = f"ln{ln}_seed{seed}"; p = []
    tr = os.path.join(log_dir, f"proxy2_train_{tag}.csv"); dg = os.path.join(log_dir, f"proxy2_diag_{tag}.jsonl"); cf = os.path.join(log_dir, f"proxy2_config_{tag}.json")
    for f in (tr, dg, cf):
        if not os.path.isfile(f): return [f"missing {os.path.basename(f)}"]
    df = pd.read_csv(tr)
    if len(df) != epochs or list(df.epoch) != list(range(1, epochs + 1)): p.append(f"train csv has {len(df)} rows or non-sequential epochs")
    if df.isna().any().any(): p.append("NaN in train csv")
    cfg = json.load(open(cf))["arguments"]
    if cfg["seed"] != seed or abs(cfg["label_noise"] - float(ln)) > 1e-12 or cfg["k_batches"] != k or cfg["epochs"] != epochs or not cfg["deterministic"] or cfg.get("legacy_batches"):
        p.append("config mismatch (seed, noise, K, epochs, deterministic, legacy_batches)")
    recs = [json.loads(x) for x in open(dg) if x.strip()]
    if [r["epoch"] for r in recs] != CHECKPOINTS: p.append(f"diag checkpoints {[r['epoch'] for r in recs]}")
    for r in recs:
        if r["k_batches"] != k: p.append(f"epoch {r['epoch']}: K={r['k_batches']}")
        if r["stream_steps"] < 2: p.append(f"epoch {r['epoch']}: stream_steps={r['stream_steps']}")
        if len(r["tensors"]) != 62: p.append(f"epoch {r['epoch']}: {len(r['tensors'])} tensor records")
        if any(isinstance(v, float) and v != v for kk, v in r.items() if kk != "tensors"): p.append(f"epoch {r['epoch']}: NaN pooled field")
        npz = os.path.join(log_dir, f"proxy2_sample_{tag}_ep{r['epoch']}.npz")
        if not os.path.isfile(npz): p.append(f"missing sample file for epoch {r['epoch']}")
    if ln == "0.2":
        found = False
        for d in refs:
            ref = os.path.join(d, f"adam_nv_scheduleconstant_sigma0.0_ln0.2_seed{seed}.csv")
            if os.path.isfile(ref):
                found = True
                diff = float(np.abs(df.val_acc.values - pd.read_csv(ref).val_acc.values).max())
                if diff > 1e-6: p.append(f"validation accuracy differs from the earlier Adam-NV run (max {diff:.2e}): diagnostics changed training or the code differs")
                break
        if not found:
            print(f"  note: no earlier Adam-NV log found for seed {seed} in --reference-dirs; the training-equivalence check was SKIPPED for this job")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True); ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--k-batches", type=int, default=64)
    ap.add_argument("--jobs", nargs="+", required=True, help="label_noise:seed pairs, e.g. 0.0:0 0.2:0")
    ap.add_argument("--reference-dirs", nargs="*", default=[])
    a = ap.parse_args(); failed = 0
    print(f"{'job':14s} result")
    for j in a.jobs:
        ln, seed = j.split(":"); pr = check_job(a.log_dir, ln, int(seed), a.epochs, a.k_batches, a.reference_dirs)
        print(f"{j:14s} {'OK' if not pr else 'FAIL: ' + '; '.join(pr)}"); failed += bool(pr)
    print(f"\n{len(a.jobs) - failed}/{len(a.jobs)} jobs verified"); sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
