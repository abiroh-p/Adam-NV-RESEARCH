#!/usr/bin/env python3
"""Source-provenance registry and verifier for Adam-NV experiment logs.

Why this exists
---------------
The original ``check_logs.py`` only tested that a ``code_sha256`` field
*existed*. It never compared the recorded hash with any source file, so a run
could point at code that is no longer in the repository and still "pass".

This tool compares every hash recorded by a run against a registry of
byte-exact archived source versions (``provenance/source_archive``).

Verdicts (one per recorded hash claim)
--------------------------------------
VERIFIED_EXACT            raw SHA-256 equals an archived file with the claimed name
VERIFIED_EXACT_RENAMED    raw SHA-256 equals an archived file stored under another
                          name (e.g. metadata key ``train_resnet.py`` but the bytes
                          are ``train_resnet_ex.py``)
VERIFIED_EOL_NORMALIZED   hash differs only because of line endings: the LF-normalized
                          content equals an archived file. Behaviourally the same
                          source, but NOT a byte match.
UNVERIFIED_NOT_ARCHIVED   no archived file reproduces the hash. Provenance of the run
                          cannot be tied to any preserved source. Never guessed.
COVERED_BY_FOLDER_HASHES  the run's own metadata has no hash, but its launch folder holds a
                          ``_code_hashes_*.txt`` whose every claim verifies (EXP07/08/09 pattern).
                          Weaker than a per-run hash: it proves what the launcher hashed, not
                          that this exact run used it.
NO_HASH_RECORDED          neither the run's metadata nor its folder records a code hash.

Usage
-----
    python tools/provenance.py build  [--archive DIR] [--out registry.json]
    python tools/provenance.py verify [--root DIR] [--registry registry.json]
                                      [--csv report.csv] [--strict]

``verify`` scans ``*_metadata.json`` and ``_code_hashes_*.txt`` in the directory
tree and inside (nested) ``.zip`` files, never extracting to disk.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = REPO / "provenance" / "source_archive"
DEFAULT_REGISTRY = REPO / "provenance" / "source_hash_registry.json"

VERIFIED = {"VERIFIED_EXACT", "VERIFIED_EXACT_RENAMED", "VERIFIED_EOL_NORMALIZED", "COVERED_BY_FOLDER_HASHES"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def lf_normalize(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def build_registry(archive: Path) -> dict:
    """Hash every archived source file (name@sha8 convention)."""
    entries = []
    for path in sorted(archive.iterdir()):
        if not path.is_file():
            continue
        data = path.read_bytes()
        raw = sha256_bytes(data)
        base, _, tag = path.name.partition("@")
        if tag and not raw.startswith(tag):
            raise SystemExit(f"{path.name}: label does not match actual SHA-256 {raw[:8]}")
        entries.append({
            "archived_as": str(path.relative_to(REPO)).replace(os.sep, "/"),
            "file_name": base,
            "sha256": raw,
            "sha256_lf_normalized": sha256_bytes(lf_normalize(data)),
            "bytes": len(data),
        })
    return {"schema": 1, "entries": entries}


def load_registry(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def classify(claimed_name: str, claimed_hash: str, registry: dict) -> tuple[str, str]:
    """Return (verdict, archived_as or '')."""
    claimed_hash = claimed_hash.strip().lower()
    raw_hits = [e for e in registry["entries"] if e["sha256"] == claimed_hash]
    if raw_hits:
        same_name = [e for e in raw_hits if e["file_name"] == claimed_name]
        if same_name:
            return "VERIFIED_EXACT", same_name[0]["archived_as"]
        return "VERIFIED_EXACT_RENAMED", raw_hits[0]["archived_as"]
    norm_hits = [e for e in registry["entries"] if e["sha256_lf_normalized"] == claimed_hash]
    if norm_hits:
        return "VERIFIED_EOL_NORMALIZED", norm_hits[0]["archived_as"]
    # A recorded hash might be the CRLF/LF variant of an archived file.
    for e in registry["entries"]:
        data = (REPO / e["archived_as"]).read_bytes()
        lf = lf_normalize(data)
        if sha256_bytes(lf.replace(b"\n", b"\r\n")) == claimed_hash or sha256_bytes(lf) == claimed_hash:
            return "VERIFIED_EOL_NORMALIZED", e["archived_as"]
    return "UNVERIFIED_NOT_ARCHIVED", ""


def iter_members(root: Path):
    """Yield (location, member_name, bytes) for metadata/hash files, recursing into zips."""
    def from_zip(zf: zipfile.ZipFile, where: str):
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            base = os.path.basename(name)
            if base.endswith("_metadata.json") or (base.startswith("_code_hashes") and base.endswith(".txt")):
                yield where, name, zf.read(name)
            elif name.lower().endswith(".zip"):
                try:
                    inner = zipfile.ZipFile(io.BytesIO(zf.read(name)))
                except zipfile.BadZipFile:
                    continue
                yield from from_zip(inner, f"{where}>{name}")

    for dirpath, _dirs, files in os.walk(root):
        if "/.git" in dirpath.replace(os.sep, "/") or "provenance/source_archive" in dirpath.replace(os.sep, "/"):
            continue
        for fn in files:
            full = Path(dirpath) / fn
            rel = str(full.relative_to(root)).replace(os.sep, "/")
            if fn.endswith("_metadata.json") or (fn.startswith("_code_hashes") and fn.endswith(".txt")):
                yield "dir", rel, full.read_bytes()
            elif fn.lower().endswith(".zip"):
                try:
                    yield from from_zip(zipfile.ZipFile(full), f"zip:{rel}")
                except zipfile.BadZipFile:
                    continue


def claims_from_member(name: str, data: bytes):
    """Yield (run_id, claimed_file_name, claimed_hash) or (run_id, None, None) when absent."""
    base = os.path.basename(name)
    if base.endswith("_metadata.json"):
        meta = json.loads(data.decode("utf-8"))
        run_id = base[: -len("_metadata.json")]
        hashes = meta.get("code_sha256")
        if not hashes:
            yield run_id, None, None
            return
        for fname, h in sorted(hashes.items()):
            yield run_id, fname, h
    else:  # _code_hashes_*.txt: "<sha256>  <file>"
        for line in data.decode("utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2 and len(parts[0]) == 64:
                yield base, parts[1], parts[0]


def verify(root: Path, registry: dict) -> list[dict]:
    rows = []
    for where, name, data in iter_members(root):
        try:
            for run_id, fname, h in claims_from_member(name, data):
                if fname is None:
                    rows.append(dict(container=where, member=name, run=run_id, file="", claimed_sha256="",
                                     verdict="NO_HASH_RECORDED", archived_as=""))
                else:
                    verdict, arch = classify(fname, h, registry)
                    rows.append(dict(container=where, member=name, run=run_id, file=fname,
                                     claimed_sha256=h, verdict=verdict, archived_as=arch))
        except (ValueError, UnicodeDecodeError) as exc:
            rows.append(dict(container=where, member=name, run="", file="", claimed_sha256="",
                             verdict=f"UNREADABLE:{exc}", archived_as=""))
    return rows


def apply_folder_coverage(rows: list[dict]) -> list[dict]:
    """Upgrade NO_HASH_RECORDED rows whose folder has a fully verified _code_hashes_*.txt."""
    folder = lambda r: (r["container"], os.path.dirname(r["member"]))
    ok: dict = {}
    for r in rows:
        if os.path.basename(r["member"]).startswith("_code_hashes"):
            ok.setdefault(folder(r), []).append(r["verdict"] in VERIFIED and r["verdict"] != "COVERED_BY_FOLDER_HASHES")
    for r in rows:
        if r["verdict"] == "NO_HASH_RECORDED" and ok.get(folder(r)) and all(ok[folder(r)]):
            r["verdict"] = "COVERED_BY_FOLDER_HASHES"
    return rows


def summarize(rows: list[dict]) -> dict:
    out: dict[str, int] = {}
    for r in rows:
        out[r["verdict"]] = out.get(r["verdict"], 0) + 1
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    b.add_argument("--out", type=Path, default=DEFAULT_REGISTRY)
    v = sub.add_parser("verify")
    v.add_argument("--root", type=Path, default=REPO)
    v.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    v.add_argument("--csv", type=Path, default=None)
    v.add_argument("--strict", action="store_true",
                   help="exit 1 if any claim is unverified or any run has no hash")
    args = ap.parse_args(argv)

    if args.cmd == "build":
        reg = build_registry(args.archive)
        args.out.write_text(json.dumps(reg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"registry: {len(reg['entries'])} archived source versions -> {args.out}")
        return 0

    reg = load_registry(args.registry)
    rows = apply_folder_coverage(verify(args.root, reg))
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["verdict"])
            w.writeheader()
            w.writerows(rows)
    summary = summarize(rows)
    print("claims scanned:", len(rows))
    for k in sorted(summary):
        print(f"  {k:28s} {summary[k]}")
    bad = sum(v for k, v in summary.items() if k not in VERIFIED)
    if args.strict and bad:
        print(f"STRICT: {bad} claim(s) not verified")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
