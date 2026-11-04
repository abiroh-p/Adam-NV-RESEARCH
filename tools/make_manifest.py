#!/usr/bin/env python3
"""Write MANIFEST.md and MANIFEST.sha256 for a repaired-repository tree.

Each file is classified against a pristine clone of the original repository (--original):
  original-unchanged | original-modified | added.   Caches (__pycache__, *.pyc, .git) are excluded.
"""
import argparse, hashlib, os, sys
from collections import Counter
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, required=True); ap.add_argument("--original", type=Path, required=True)
ap.add_argument("--retained", type=Path, help="text file: paths deleted upstream but deliberately kept here")
a = ap.parse_args()
retained = set(a.retained.read_text().split()) if a.retained else set()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
rows = []
for dp, dn, fn in os.walk(a.root):
    dn[:] = [d for d in dn if d not in (".git", "__pycache__")]
    for f in sorted(fn):
        if f.endswith(".pyc") or f in ("MANIFEST.md", "MANIFEST.sha256"): continue
        p = Path(dp) / f; rel = p.relative_to(a.root).as_posix(); o = a.original / rel
        cls = "retained-after-upstream-deletion" if rel in retained else "added" if not o.exists() else ("original-unchanged" if sha(o) == sha(p) else "original-modified")
        rows.append((rel, p.stat().st_size, sha(p), cls))
rows.sort()
(a.root / "MANIFEST.sha256").write_text("".join(f"{h}  {r}\n" for r, _s, h, _c in rows), encoding="utf-8")
cnt = Counter(c for *_x, c in rows); tot = sum(s for _r, s, _h, _c in rows)
top = Counter(r.split("/")[0] if "/" in r else "(root)" for r, *_x in rows)
out = ["# Manifest", "", f"{len(rows)} files, {tot/1e6:.1f} MB (excluding this manifest and its checksum file).", "",
       "| classification | files |", "|---|---|"] + [f"| {k} | {v} |" for k, v in sorted(cnt.items())] + ["",
       "`MANIFEST.sha256` lists SHA-256 for every file (`sha256sum -c MANIFEST.sha256`). The `.git` directory is not included; the original history is at https://github.com/abiroh-p/Adam-NV (7 commits).", "",
       "## Original files that were modified", ""]
out += [f"* `{r}`" for r, _s, _h, c in rows if c == "original-modified"]
out += ["", "## Kept although deleted upstream", ""] + [f"* `{r}`" for r, _s, _h, c in rows if c == "retained-after-upstream-deletion"]
out += ["", "## Added files (excluding generated logs)", ""]
out += [f"* `{r}`" for r, _s, _h, c in rows if c == "added" and not r.startswith(("reports/validation_logs/", "analysis/results_recomputed/"))]
out += ["", "## Top-level contents", "", "| directory | files |", "|---|---|"] + [f"| {k} | {v} |" for k, v in sorted(top.items())]
(a.root / "MANIFEST.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print(dict(cnt), len(rows), "files")
