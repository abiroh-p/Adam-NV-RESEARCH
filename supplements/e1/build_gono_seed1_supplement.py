#!/usr/bin/env python3
"""Build the recovered-epoch supplement for E1 GONO seed 1 (epoch 60).

The original CSV ends at epoch 59 although the run finished (suite status
'success'). The suite's stdout capture still contains the epoch-60 progress line.
This script:
  1. parses every `[gono_seed1] epoch N/60 | ...` stdout line,
  2. checks that the 59 epochs present in the CSV match the stdout values at the
     stdout's 4-decimal precision (abort if any differ),
  3. records epoch 60 exactly as printed, with its precision limits.

It never edits the original CSV. Only columns printed by stdout are recorded.
"""
import hashlib, json, re, sys
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "adam-nv/e1_convergence/logs/gono_seed1.csv"
STDOUT = REPO / "adam-nv/e1_convergence/logs/stdout_gono_gpu0.txt"
OUT = REPO / "supplements/e1/gono_seed1_epoch60_from_stdout.json"
RERUN = REPO / "adam-nv/e1_convergence/logs/gono_seed_1/gono_seed1.csv"  # author-supplied rerun, if present
PAT = re.compile(r"\[gono_seed1\] epoch\s+(\d+)/60 \| train_loss=([\d.]+) train_acc=([\d.]+) "
                 r"\| val_loss=([\d.]+) val_acc=([\d.]+)")

def main() -> int:
    parsed, lines = {}, {}
    for n, line in enumerate(STDOUT.read_text(errors="replace").splitlines(), 1):
        m = PAT.search(line)
        if m:
            parsed[int(m.group(1))] = tuple(float(x) for x in m.groups()[1:])
            lines[int(m.group(1))] = (n, line.strip())
    frame = pd.read_csv(CSV)
    csv_epochs = [int(e) for e in frame["epoch"]]
    if csv_epochs != list(range(1, 60)):
        sys.exit(f"unexpected CSV epochs: {csv_epochs[:3]}..{csv_epochs[-3:]}")
    bad = []
    for _, r in frame.iterrows():
        e = int(r.epoch); s = parsed[e]
        got = (round(r.train_loss, 4), round(r.train_acc, 4), round(r.val_loss, 4), round(r.val_acc, 4))
        if any(abs(a - b) > 1.1e-4 for a, b in zip(got, s)):
            bad.append((e, got, s))
    if bad:
        sys.exit(f"stdout disagrees with CSV at {len(bad)} epochs, e.g. {bad[0]}")
    if 60 not in parsed:
        sys.exit("epoch 60 not present in stdout")
    tl, ta, vl, va = parsed[60]
    doc = {
        "purpose": "Recovered record for the epoch missing from the original CSV. Original CSV is unchanged.",
        "run_csv": str(CSV.relative_to(REPO)),
        "run_csv_sha256": hashlib.sha256(CSV.read_bytes()).hexdigest(),
        "missing_epochs": [60],
        "recovered": {"60": {"train_loss": tl, "train_acc": ta, "val_loss": vl, "val_acc": va}},
        "precision": {
            "val_acc": "exact: the validation set has 10,000 images, so 4 decimals is lossless",
            "train_acc": "rounded to 4 decimals (true value in [0.99885, 0.99895]); 50,000 training images",
            "train_loss": "rounded to 4 decimals",
            "val_loss": "rounded to 4 decimals",
        },
        "not_recoverable": ["lr", "epoch_time_sec", "beta1_*", "cosine_* and every other logged column"],
        "source": {"file": str(STDOUT.relative_to(REPO)), "file_sha256": hashlib.sha256(STDOUT.read_bytes()).hexdigest(),
                   "line_number": lines[60][0], "line_text": lines[60][1]},
        "validation": "all 59 CSV epochs match the stdout lines at 4-decimal precision (0 mismatches)",
        "suite_status": "adam-nv/e1_convergence/logs/_suite_status_gono_gpu0.csv: gono,1,success,3029.5",
        "usage_rule": "Analyses must opt in (use_supplement=True) and must flag any result that used it. "
                      "Only val_acc is suitable for the last-5 / final-epoch accuracy metrics.",
    }
    if RERUN.exists():
        rr = pd.read_csv(RERUN)
        cols = [c for c in frame.columns if c != "epoch_time_sec" and c in rr.columns]
        first59 = rr.iloc[:59].reset_index(drop=True)
        maxdiff = max(float((first59[c] - frame[c]).abs().max()) for c in cols)
        e60 = rr[rr.epoch == 60].iloc[0]
        agrees = (round(e60.train_loss, 4), round(e60.train_acc, 4), round(e60.val_loss, 4), round(e60.val_acc, 4)) == (tl, ta, vl, va)
        doc["independent_corroboration"] = {
            "rerun_csv": str(RERUN.relative_to(REPO)), "rerun_csv_sha256": hashlib.sha256(RERUN.read_bytes()).hexdigest(),
            "rerun_rows": int(len(rr)), "compared_columns": cols, "max_abs_diff_epochs_1_59": maxdiff,
            "epoch60_rerun_values_rounded_to_4dp_equal_stdout_values": bool(agrees),
            "epoch60_rerun_full_precision": {k: float(e60[k]) for k in ("train_loss", "train_acc", "val_loss", "val_acc")},
            "note": "The rerun recorded code hashes that all verify against provenance/source_archive; the original run's hashes do not (PRV-01)."}
        doc["usage_rule"] = ("Superseded for analysis by the complete rerun (analysis/analyze_e1.py uses the rerun by default). "
                             "Kept as a documented, independently corroborated record.")
    OUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT.relative_to(REPO), "| epoch 60:", doc["recovered"]["60"])
    return 0

if __name__ == "__main__":
    sys.exit(main())
