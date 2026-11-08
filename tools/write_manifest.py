#!/usr/bin/env python3
"""Write MANIFEST.sha256 and MANIFEST.md for the repository tree.   Usage: python tools/write_manifest.py   (run from anywhere)
Verify afterwards with:  sha256sum -c MANIFEST.sha256   (from the repository root). Caches and .git are excluded."""
import hashlib, os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "__pycache__", ".cache", ".pytest_cache", ".venv"}
rows = []
for dp, dn, fn in os.walk(ROOT):
    dn[:] = sorted(d for d in dn if d not in SKIP_DIRS)
    for f in sorted(fn):
        if f.endswith(".pyc") or f in ("MANIFEST.md", "MANIFEST.sha256"): continue
        p = Path(dp) / f; rows.append((p.relative_to(ROOT).as_posix(), p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest()))
rows.sort()
(ROOT / "MANIFEST.sha256").write_text("".join(f"{h}  {r}\n" for r, _s, h in rows), encoding="utf-8")
top = Counter(r.split("/")[0] if "/" in r else "(root)" for r, *_x in rows); total = sum(s for _r, s, _h in rows)
(ROOT / "MANIFEST.md").write_text("# Manifest\n\n" + f"{len(rows)} files, {total / 1e6:.1f} MB (excluding this manifest and its checksum file).\n\n"
    "Verify every file with `sha256sum -c MANIFEST.sha256` from the repository root (it writes OK for each file).\n\n| top-level entry | files |\n|---|---|\n"
    + "".join(f"| `{k}` | {v} |\n" for k, v in sorted(top.items())) + "\nHash-pinned sources keep their original bytes (see `.gitattributes`): do not let an editor or git change their line endings.\n", encoding="utf-8")
print(f"wrote MANIFEST.sha256 and MANIFEST.md ({len(rows)} files)")
