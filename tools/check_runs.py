#!/usr/bin/env python3
"""Run-completeness and integrity checker for per-epoch CSV logs.

Per CSV it reports one status:
  OK                           60 contiguous epochs, finite values, no duplicates
  INCOMPLETE                   rows missing / epochs out of range
  INCOMPLETE_SUPPLEMENT_AVAILABLE  incomplete, but a recorded supplement exists (not applied here)
  DUPLICATE_EPOCHS | NONFINITE | BAD_COLUMNS
  KNOWN_PARTIAL_SUPERSEDED     listed in reports/known_superseded_logs.json (hash must match)
  EXCLUDED_AUXILIARY           smoke / step-level / diagnostic file, not a per-epoch run log
and cross-checks (directories AND the contents of .zip files, recursively; --no-zips to skip): metadata sidecar present, byte-identical duplicates, same-name
files with different content in different folders.

Usage:  python tools/check_runs.py [--root DIR] [--epochs 60] [--csv OUT.csv] [--strict]
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, os, sys, zipfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
import pandas as pd  # noqa: E402
import run_io  # noqa: E402

AUX_NAME = ("smoke", "_steps", "finegrained", "_status", "calibration_report", "results_table",
            "overhead", "primary_statistics")
AUX_DIRS = ("part_a", "part_b_diag", "calibration", "analysis/results", "benchmarks", "supplements")

def is_aux(rel: str) -> bool:
    low = rel.lower()
    return any(t in os.path.basename(low) for t in AUX_NAME) or any(f"/{d}/" in f"/{low}" for d in AUX_DIRS)

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=REPO)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--no-zips", action="store_true", help="do not look inside .zip archives")
    ap.add_argument("--strict", action="store_true", help="exit 1 on any non-OK, non-excluded, non-known status")
    a = ap.parse_args(argv)

    known = {}
    kp = a.root / "reports" / "known_superseded_logs.json"
    if kp.exists():
        for e in json.loads(kp.read_text()):
            known[e["path"]] = e
    rows, by_name = [], defaultdict(list)

    def examine(rel, data, has_meta, container):
        """Classify one CSV given its bytes; rel is the display path (zip members use 'zip:<zip>>member')."""
        digest = hashlib.sha256(data).hexdigest()
        by_name[os.path.basename(rel)].append((rel, digest))
        if is_aux(rel):
            rows.append(dict(path=rel, status="EXCLUDED_AUXILIARY", detail="", rows="", metadata="")); return
        try:
            frame = pd.read_csv(io.BytesIO(data))
        except Exception as exc:  # noqa: BLE001
            rows.append(dict(path=rel, status="BAD_COLUMNS", detail=str(exc)[:80], rows="", metadata="")); return
        if not {"epoch", "val_acc"} <= set(frame.columns):
            rows.append(dict(path=rel, status="EXCLUDED_AUXILIARY", detail="no epoch/val_acc columns", rows=len(frame), metadata="")); return
        meta = "yes" if has_meta else "NO"
        try:
            run_io.validate_frame(frame, rel, a.epochs); status, detail = "OK", ""
        except run_io.RunIntegrityError as exc:
            msg = str(exc)
            if rel in known and known[rel]["sha256"] == digest:
                status, detail = "KNOWN_PARTIAL_SUPERSEDED", known[rel]["superseded_by"]
            elif "duplicated" in msg: status, detail = "DUPLICATE_EPOCHS", msg
            elif "non-finite" in msg: status, detail = "NONFINITE", msg
            else:
                sup = next((c for c in (a.root / "supplements").rglob("*.json")
                            if json.loads(c.read_text()).get("run_csv_sha256") == digest), None) if (a.root / "supplements").is_dir() else None
                status = "INCOMPLETE_SUPPLEMENT_AVAILABLE" if sup else "INCOMPLETE"; detail = msg.split(": ", 1)[-1]
        rows.append(dict(path=rel, status=status, detail=detail, rows=len(frame), metadata=meta))

    def walk_zip(zf, where):
        names = set(zf.namelist())
        for n in sorted(names):
            if n.endswith("/"): continue
            base = os.path.basename(n)
            if n.lower().endswith(".zip"):
                try: walk_zip(zipfile.ZipFile(io.BytesIO(zf.read(n))), f"{where}>{n}")
                except zipfile.BadZipFile: pass
            elif base.endswith(".csv") and not base.startswith("_"):
                examine(f"{where}>{n}", zf.read(n), n[:-4] + "_metadata.json" in names, "zip")

    for dp, _dn, fns in os.walk(a.root):
        d = dp.replace(os.sep, "/")
        if "/.git" in d or "__pycache__" in d or "provenance/source_archive" in d:
            continue
        for fn in sorted(fns):
            full = Path(dp) / fn; rel = str(full.relative_to(a.root)).replace(os.sep, "/")
            if fn.endswith(".csv") and not fn.startswith("_"):
                examine(rel, full.read_bytes(), full.with_name(full.stem + "_metadata.json").exists(), "dir")
            elif fn.lower().endswith(".zip") and not a.no_zips:
                try: walk_zip(zipfile.ZipFile(full), f"zip:{rel}")
                except zipfile.BadZipFile: pass
    # cross-folder name checks (informational): identical copies vs same name with DIFFERENT content
    for name, items in by_name.items():
        if len(items) > 1:
            for rel, h in items:
                same = [p.rsplit("/", 2)[-2] for p, hh in items if p != rel and hh == h]
                diff = [p.rsplit("/", 2)[-2] for p, hh in items if hh != h]
                note = ("byte-identical copy in: " + ", ".join(same) if same else "") + \
                       ("; " if same and diff else "") + \
                       ("SAME NAME, DIFFERENT CONTENT in: " + ", ".join(diff) if diff else "")
                for r in rows:
                    if r["path"] == rel and r["status"] == "OK":
                        r["detail"] = note
    counts = defaultdict(int)
    for r in rows:
        counts[r["status"]] += 1
    print(f"csv files examined: {len(rows)}")
    for k in sorted(counts):
        print(f"  {k:34s} {counts[k]}")
    nometa = sum(1 for r in rows if r.get("metadata") == "NO")
    print(f"  per-epoch runs without a metadata sidecar: {nometa}")
    for r in rows:
        if r["status"] not in ("OK", "EXCLUDED_AUXILIARY"):
            print(f"  [{r['status']}] {r['path']}  {r['detail']}")
    if a.csv:
        a.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["path", "status", "rows", "metadata", "detail"])
            w.writeheader(); w.writerows(rows)
    bad = [r for r in rows if r["status"] not in ("OK", "EXCLUDED_AUXILIARY", "KNOWN_PARTIAL_SUPERSEDED")]
    return 1 if (a.strict and bad) else 0

if __name__ == "__main__":
    sys.exit(main())
