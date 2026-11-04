#!/usr/bin/env python3
"""E1 (CIFAR-10 / ResNet) convergence analysis, regenerated from raw per-epoch logs.

This replaces the missing ``analyze_e1.py`` / ``analyze_e1_protocol_v2.py`` that README and
the EXP01/EXP03 runbooks refer to. It does NOT reproduce numbers from those lost scripts; it
recomputes everything from the logs, with explicit definitions:

* ``last5``  mean validation accuracy over epochs 56-60  (primary summary)
* ``final``  validation accuracy at epoch 60              (reported separately, never mixed)
* SD uses ddof=1 (the committed ``e1_results_table.csv`` used ddof=0; see DISCREPANCY_REGISTER S3)

Inputs (explicit, overridable)::

    --top-logs         e1_convergence/logs                         adam, adamw, adabelief, adam_nv, seeds 0-3
    --additional-logs  adam-nv/e1_convergence/AdditionalsLogs      adam, adamw, adam_nv, seeds 4-9
    --gono-logs        adam-nv/e1_convergence/logs                 gono, seeds 0-9

Integrity: every log must have 60 contiguous epochs. An (optimizer, seed) found in more than one
folder is an error. GONO seed 1 is incomplete in the original CSV. Resolution order per run:
(1) a complete original; (2) a complete author-supplied rerun (``--rerun-logs``, default
``gono_seed_1/``) whose overlapping epochs equal the original in every non-timing column
(otherwise exit 2, nothing merged); (3) only with ``--use-supplement``, the stdout-recovered epoch 60
(val_acc exact), flagged ``used_supplement=True``; (4) otherwise the run is excluded and listed.

Contrasts are paired by seed on the common seeds only; unpaired seeds are listed, never dropped
silently. Holm is applied within the families declared in ``FAMILIES``. These families are the
ones implied by the EXP01 runbook ("Adam-NV vs GONO vs Adam") plus baselines; the runbooks give
no multiplicity family or equivalence margin, so they are PROPOSED here and marked
``family_status = proposed_pending_author_confirmation``. No verdict is issued.
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import run_io  # noqa: E402

_spec = importlib.util.spec_from_file_location("analyze_e13_e15", HERE / "analyze_e13_e15.py")
_e13 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_e13)  # reuse paired_result / holm_adjust: one implementation of the statistics

OPTIMIZERS = ("adam", "adamw", "adabelief", "adam_nv", "gono")
def rel(p: Path) -> str:
    """Repo-relative path when inside the repo, otherwise the path as given (inputs may live anywhere)."""
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except ValueError:
        return str(p)


LOG_RE = re.compile(r"^(?P<opt>[a-z_]+?)_seed(?P<seed>\d+)\.csv$")

# (family, left, right, seeds the contrast is expected to cover)
FAMILIES = [
    ("primary", "adam_nv", "adam", list(range(10))),
    ("primary", "gono", "adam", list(range(10))),
    ("primary", "adam_nv", "gono", list(range(10))),
    ("secondary", "adam_nv", "adamw", list(range(10))),
    ("secondary", "adam_nv", "adabelief", list(range(4))),
]


def discover(dirs: list[Path]) -> tuple[dict, list[str]]:
    paths: dict[tuple[str, int], tuple[Path, str]] = {}
    problems: list[str] = []
    for d in dirs:
        if not d.is_dir():
            problems.append(f"missing directory {d}")
            continue
        for f in sorted(d.glob("*_seed*.csv")):
            m = LOG_RE.match(f.name)
            if not m or m["opt"] not in OPTIMIZERS:
                continue
            key = (m["opt"], int(m["seed"]))
            if key in paths:
                problems.append(f"duplicate observation for {key}: {paths[key][0]} and {f}")
            else:
                paths[key] = (f, d.name)
    return paths, problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top-logs", type=Path, default=REPO / "e1_convergence" / "logs")
    ap.add_argument("--additional-logs", type=Path, default=REPO / "adam-nv" / "e1_convergence" / "AdditionalsLogs")
    ap.add_argument("--gono-logs", type=Path, default=REPO / "adam-nv" / "e1_convergence" / "logs")
    ap.add_argument("--rerun-logs", type=Path, nargs="*", default=[REPO / "adam-nv" / "e1_convergence" / "logs" / "gono_seed_1"],
                    help="author-supplied reruns. Policy: a complete original always wins; a complete rerun replaces an "
                         "INCOMPLETE original only after its overlapping epochs are shown to match (non-timing columns)")
    ap.add_argument("--supplements", type=Path, default=REPO / "supplements")
    ap.add_argument("--use-supplement", action="store_true",
                    help="fill missing epochs from recorded supplements (flagged in every output row)")
    ap.add_argument("--metric", choices=("last5", "final"), default="last5")
    ap.add_argument("--out-dir", type=Path, default=HERE / "results_recomputed")
    a = ap.parse_args(argv)

    paths, problems = discover([a.top_logs, a.additional_logs, a.gono_logs])
    if any("duplicate" in p for p in problems):
        for p in problems:
            print("ERROR:", p)
        return 2

    reruns, _rp = discover([d for d in a.rerun_logs if d.is_dir()])
    consistency = []
    per_seed, excluded = [], []
    values: dict[str, dict[int, float]] = {o: {} for o in OPTIMIZERS}
    flagged: dict[str, set] = {o: set() for o in OPTIMIZERS}
    for (opt, seed), (path, source) in sorted(paths.items()):
        run, reason = None, ""
        try:  # 1) a complete original always wins
            run = run_io.load_run(path)
        except run_io.RunIntegrityError as exc:
            reason = str(exc).split(": ", 1)[-1]
        if run is None and (opt, seed) in reruns:  # 2) verified rerun replaces an INCOMPLETE original
            rpath = reruns[(opt, seed)][0]
            try:
                rrun = run_io.load_run(rpath)
                orig = pd.read_csv(path)
                cols = [c for c in orig.columns if c != "epoch_time_sec" and c in rrun.frame.columns]
                overlap = rrun.frame[rrun.frame.epoch.isin(orig.epoch)].reset_index(drop=True)
                maxdiff = max(float((overlap[c] - orig[c]).abs().max()) for c in cols)
                consistency.append(dict(optimizer=opt, seed=seed, original=rel(path), rerun=rel(rpath),
                                        original_rows=len(orig), overlapping_epochs=len(overlap), compared_columns=len(cols),
                                        max_abs_diff_non_timing=maxdiff, consistent=maxdiff == 0.0))
                if maxdiff != 0.0:
                    print(f"ERROR: rerun {rpath} disagrees with the incomplete original (max diff {maxdiff}); refusing to merge")
                    return 2
                run, path, source = rrun, rpath, f"{rpath.parent.name} (rerun replacing incomplete original)"
            except run_io.RunIntegrityError as exc2:
                reason += f"; rerun also unusable: {str(exc2).split(': ', 1)[-1]}"
        if run is None and a.use_supplement:  # 3) opt-in, flagged supplement
            try:
                run = run_io.load_run(path, use_supplement=True, supplements_dir=a.supplements)
            except run_io.RunIntegrityError:
                pass
        if run is None:  # 4) excluded, reported
            excluded.append(dict(optimizer=opt, seed=seed, reason=reason, file=rel(path)))
            continue
        last5 = run_io.last_k_epoch_mean_metric(run)
        final = run_io.final_epoch_metric(run)
        values[opt][seed] = last5 if a.metric == "last5" else final
        if run.used_supplement:
            flagged[opt].add(seed)
        per_seed.append(dict(optimizer=opt, seed=seed, source_folder=source, last5_val_acc=last5,
                             final_epoch_val_acc=final, used_supplement=run.used_supplement,
                             file=rel(path)))
    per = pd.DataFrame(per_seed)

    summary = []
    for opt in OPTIMIZERS:
        v = np.array([values[opt][s] for s in sorted(values[opt])])
        if len(v):
            summary.append(dict(optimizer=opt, metric=_e13.METRIC_LABELS["last5" if a.metric == "last5" else "final"],
                                n_seeds=len(v), seeds=" ".join(map(str, sorted(values[opt]))),
                                mean=v.mean(), sd_ddof1=v.std(ddof=1) if len(v) > 1 else np.nan,
                                min=v.min(), max=v.max(),
                                used_supplement=bool(flagged[opt])))
    summ = pd.DataFrame(summary)

    _e13.METRIC_MODE = a.metric
    rows = []
    for fam, left, right, expected in FAMILIES:
        if not values[left] or not values[right]:
            continue
        r = _e13.paired_result("E1", f"{left} minus {right}", left, right, values[left], values[right], expected)
        r["family"] = fam
        r["family_status"] = "proposed_pending_author_confirmation"
        r["unpaired_seeds_left"] = " ".join(str(s) for s in sorted(set(values[left]) - set(values[right])))
        r["unpaired_seeds_right"] = " ".join(str(s) for s in sorted(set(values[right]) - set(values[left])))
        r["used_supplement"] = bool((flagged[left] | flagged[right]) & set(map(int, r["paired_seeds"].split())))
        rows.append(r)
    res = pd.DataFrame(rows)
    res["p_paired_t_holm_within_family"] = np.nan
    for fam, idx in res.groupby("family").groups.items():
        res.loc[list(idx), "p_paired_t_holm_within_family"] = _e13.holm_adjust(res.loc[list(idx), "p_paired_t_raw"].tolist())

    tag = f"{a.metric}{'_with_supplement' if a.use_supplement else ''}"
    a.out_dir.mkdir(parents=True, exist_ok=True)
    per.to_csv(a.out_dir / "e1_per_seed.csv", index=False)  # carries both metrics
    summ.to_csv(a.out_dir / f"e1_summary_{tag}.csv", index=False)
    res.to_csv(a.out_dir / f"e1_paired_{tag}.csv", index=False)
    if consistency:
        pd.DataFrame(consistency).to_csv(a.out_dir / "e1_rerun_consistency.csv", index=False)
    if excluded:
        pd.DataFrame(excluded).to_csv(a.out_dir / f"e1_excluded_runs_{tag}.csv", index=False)

    print(f"metric={a.metric}  use_supplement={a.use_supplement}")
    print(summ[["optimizer", "n_seeds", "mean", "sd_ddof1", "used_supplement"]].round(5).to_string(index=False))
    for e in excluded:
        print(f"EXCLUDED {e['optimizer']} seed {e['seed']}: {e['reason']}")
    print(res[["comparison", "n_pairs", "complete", "mean_difference_percentage_points", "ci95_low", "ci95_high",
               "p_paired_t_raw", "p_sign_flip_exact", "p_paired_t_holm_within_family"]].round(4).to_string(index=False))
    print(f"outputs in {a.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
