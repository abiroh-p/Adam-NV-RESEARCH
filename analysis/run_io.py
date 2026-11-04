"""Strict loading and explicit metric definitions for per-epoch run logs.

Two accuracy summaries are deliberately kept as separate, named functions so
they cannot be confused in tables or prose:

* ``final_epoch_metric``      value at the last epoch (epoch == expected_epochs)
* ``last_k_epoch_mean_metric`` mean over the last k epochs (default k = 5)

The loader refuses to return a run unless it has exactly ``expected_epochs``
rows numbered 1..expected_epochs with no duplicates and only finite numbers
in the metric columns. Nothing is ever interpolated.

Recovered-epoch supplements (see ``supplements/``) are only used when the
caller passes ``use_supplement=True``. A supplement can only fill a *missing*
epoch, only for the columns it actually recorded, and every use is reported
through ``RunData.supplement_used`` so downstream tables can flag it.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

EXPECTED_EPOCHS = 60
TAIL_EPOCHS = 5


class RunIntegrityError(ValueError):
    """The log violates a completeness or integrity requirement."""


@dataclass
class RunData:
    path: Path
    frame: pd.DataFrame
    supplement_used: dict = field(default_factory=dict)  # epoch -> {column: value}

    @property
    def used_supplement(self) -> bool:
        return bool(self.supplement_used)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find_supplement(csv_path: Path, supplements_dir: Path | None) -> Path | None:
    """Return the supplement JSON that refers to ``csv_path`` (matched by SHA-256)."""
    if supplements_dir is None or not Path(supplements_dir).is_dir():
        return None
    digest = sha256_file(csv_path)
    for cand in sorted(Path(supplements_dir).rglob("*.json")):
        try:
            doc = json.loads(cand.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if doc.get("run_csv_sha256") == digest:
            return cand
    return None


def validate_frame(frame: pd.DataFrame, path, expected_epochs: int, columns=("epoch", "val_acc")) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise RunIntegrityError(f"{path}: missing columns {missing}")
    epochs = [int(e) for e in frame["epoch"]]
    dupes = sorted({e for e in epochs if epochs.count(e) > 1})
    if dupes:
        raise RunIntegrityError(f"{path}: duplicated epochs {dupes}")
    gaps = sorted(set(range(1, expected_epochs + 1)) - set(epochs))
    extra = sorted(set(epochs) - set(range(1, expected_epochs + 1)))
    if gaps or extra:
        raise RunIntegrityError(
            f"{path}: incomplete run ({len(frame)} rows; missing epochs {gaps[:8]}"
            f"{'...' if len(gaps) > 8 else ''}; unexpected epochs {extra[:8]}; expected 1..{expected_epochs})"
        )
    if epochs != sorted(epochs):
        raise RunIntegrityError(f"{path}: epochs are not in increasing order")
    for col in columns:
        vals = pd.to_numeric(frame[col], errors="coerce")
        if vals.isna().any() or not all(math.isfinite(v) for v in vals):
            raise RunIntegrityError(f"{path}: non-finite or non-numeric values in '{col}'")
    if "val_acc" in frame.columns and not frame["val_acc"].between(0.0, 1.0).all():
        raise RunIntegrityError(f"{path}: val_acc outside [0, 1]")


def load_run(path, expected_epochs: int = EXPECTED_EPOCHS, *, use_supplement: bool = False,
             supplements_dir=None, columns=("epoch", "val_acc")) -> RunData:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    used: dict = {}
    if use_supplement:
        sup = find_supplement(path, supplements_dir)
        if sup is not None:
            doc = json.loads(sup.read_text(encoding="utf-8"))
            present = {int(e) for e in frame["epoch"]}
            for ep_str, values in doc["recovered"].items():
                ep = int(ep_str)
                if ep in present:
                    continue  # never overwrite an observed epoch
                row = {c: float("nan") for c in frame.columns}
                row["epoch"] = ep
                for k, v in values.items():
                    if k in row:
                        row[k] = v
                frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
                used[ep] = dict(values)
            frame = frame.sort_values("epoch").reset_index(drop=True)
    # Columns the caller needs must be finite; unrecovered columns of a supplemented
    # epoch stay NaN and are therefore only usable if the caller did not request them.
    validate_frame(frame, path, expected_epochs, columns)
    return RunData(path=path, frame=frame, supplement_used=used)


def final_epoch_metric(run: RunData, column: str = "val_acc") -> float:
    """Value at the LAST epoch only."""
    return float(run.frame.sort_values("epoch")[column].iloc[-1])


def last_k_epoch_mean_metric(run: RunData, column: str = "val_acc", k: int = TAIL_EPOCHS) -> float:
    """Mean of ``column`` over the last ``k`` epochs."""
    vals = run.frame.sort_values("epoch")[column].tail(k)
    if vals.isna().any():
        raise RunIntegrityError(f"{run.path}: '{column}' has unrecovered values in the last {k} epochs")
    return float(vals.mean())


def paired_seeds(left: dict, right: dict) -> list:
    """Seeds present in both arms; callers must report any seed outside this set."""
    return sorted(set(left) & set(right))
