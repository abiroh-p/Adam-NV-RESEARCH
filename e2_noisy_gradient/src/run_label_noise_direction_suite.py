"""SUPERSEDED by Experiment 9 (adam-nv/e2_noisy_gradient/src/run_exp09_label_noise_smooth.sh), which the
author has since run: 8 complete seeds are in adam-nv/e2_noisy_gradient/logs/exp09_logs_{A,B}.zip.
The file names predicted below match the real EXP09 outputs. Kept only as a record; use the EXP09 launcher.

Smooth-response vs distrust-response ablation under 20% label noise (originally written as NOT YET RUN).

The controller-direction suite (run_controller_direction_suite.py) covers clean, constant-noise
and intermittent gradient-noise conditions only. It contains no label-noise condition, so the
smooth-vs-distrust comparison under label noise did not exist in the repository.

This suite adds exactly the missing arm and nothing else:

  Adam-NV, --response-mode smooth, constant schedule, sigma 0, --label-noise 0.2,
  --label-diagnostic-interval 1, seeds 0-7.

It is built to be paired with arms that ALREADY exist, run with the same part_b_resnet_v8.py
settings (see analysis/analyze_exp07.py NAME and EXP07/EXP08 runbooks):

  adam                    logs/Label_Diagnostic seeds 0-3  +  EXP07 seeds 4-7
  adam_nv (distrust)      logs/Label_Diagnostic seeds 0-3  +  EXP07 seeds 4-7

Expected output files (one CSV + one *_metadata.json each):
  adam_nv_scheduleconstant_sigma0.0_responsesmooth_betamax0.999_ln0.2_seed{0..7}.csv

Status: this script was never executed (the author ran EXP09 instead; the repair environment has no GPU or CIFAR-10 data).
Use --dry-run to print the commands and the expected file names without training.
Record the code hashes printed by the run scripts (see EXP08_RUNBOOK) and verify them with
`python tools/provenance.py verify` afterwards.
"""
import argparse
import csv
import os
import subprocess
import sys
import time

SEEDS = list(range(8))
LABEL_NOISE = 0.2
BETA_MAX = 0.999


def expected_name(seed: int) -> str:
    # Mirrors the run_name construction in part_b_resnet_v8.py (response suffix, then label-noise suffix).
    return (f"adam_nv_scheduleconstant_sigma0.0_responsesmooth_betamax{BETA_MAX}"
            f"_ln{LABEL_NOISE}_seed{seed}")


def build_command(seed, args):
    cmd = [sys.executable, "part_b_resnet_v8.py", "--optimizer", "adam_nv",
           "--response-mode", "smooth", "--beta-max", str(BETA_MAX),
           "--schedule", "constant", "--sigma", "0.0",
           "--label-noise", str(LABEL_NOISE), "--label-diagnostic-interval", "1",
           "--seed", str(seed), "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
           "--data-dir", args.data_dir, "--log-dir", args.log_dir]
    if args.deterministic:
        cmd.append("--deterministic")
    return cmd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--data-dir", default="../data")
    p.add_argument("--log-dir", default="../logs/label_noise_direction")
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    os.makedirs(args.log_dir, exist_ok=True) if not args.dry_run else None
    rows = []
    for i, seed in enumerate(args.seeds, 1):
        name = expected_name(seed)
        path = os.path.join(args.log_dir, name + ".csv")
        cmd = build_command(seed, args)
        if args.dry_run:
            print(f"[LabelNoiseDirection] ({i}/{len(args.seeds)}) {name}.csv\n    " + " ".join(cmd))
            continue
        if args.skip_existing and os.path.exists(path):
            with open(path, newline="") as fh:
                if sum(1 for _ in fh) - 1 >= args.epochs:
                    print(f"[LabelNoiseDirection] skipping complete {name}")
                    continue
        t0 = time.time()
        try:
            subprocess.run(cmd, check=True); status, err = "success", ""
        except subprocess.CalledProcessError as exc:
            status, err = "FAILED", f"exit code {exc.returncode}"
        rows.append({"seed": seed, "status": status, "elapsed_sec": round(time.time() - t0, 1), "error": err})
        with open(os.path.join(args.log_dir, "_label_noise_direction_status.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["seed", "status", "elapsed_sec", "error"]); w.writeheader(); w.writerows(rows)
    if any(r["status"] == "FAILED" for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
