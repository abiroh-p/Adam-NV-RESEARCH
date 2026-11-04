#!/usr/bin/env bash
# Experiment 16: noise-proxy diagnostic (Adam-NV's score versus directly measured gradient noise).
# Runs noise_proxy_diagnostic.py, which trains a ResNet-18 with Adam-NV (60 epochs, as in the
# label-noise experiments) and measures minibatch gradient noise at fixed weights after epochs
# 1, 5, 10, 20, 30, 45 and 60.
#
# Jobs are "<label_noise>:<seed>" pairs, run sequentially per GPU:
#   JOBS_GPU0="0.0:0 0.2:0" JOBS_GPU1="0.0:1 0.2:1" \
#   DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp16 \
#   bash run_exp16_proxy_diagnostic.sh
#
# Environment variables: JOBS_GPU0, JOBS_GPU1, DATA_DIR (default ../data), LOG_DIR (default ../logs_exp16),
# EPOCHS (default 60), K_BATCHES (default 16), DRY_RUN=1 (print the commands only).
# A job is skipped if its proxy_train CSV already has EPOCHS data rows and its diagnostic file has
# one record per checkpoint.

set -u
cd "$(dirname "$0")"

DATA_DIR="${DATA_DIR:-../data}"
LOG_DIR="${LOG_DIR:-../logs_exp16}"
EPOCHS="${EPOCHS:-60}"
K_BATCHES="${K_BATCHES:-16}"
CHECKPOINTS="1 5 10 20 30 45 60"
JOBS_GPU0="${JOBS_GPU0:-}"
JOBS_GPU1="${JOBS_GPU1:-}"
DRY_RUN="${DRY_RUN:-0}"

if [ -z "$JOBS_GPU0$JOBS_GPU1" ]; then
  echo "[exp16] No jobs given. Set JOBS_GPU0 and/or JOBS_GPU1 (see the header of this script)."
  exit 2
fi
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

if [ "$DRY_RUN" != "1" ]; then
  python3 - <<'PY' || { echo "[exp16] Environment check failed; not starting runs."; exit 2; }
import sys
import torch, torchvision
want = {"python": "3.12.13", "torch": "2.10.0+cu128", "torchvision": "0.25.0+cu128"}
have = {"python": sys.version.split()[0], "torch": torch.__version__, "torchvision": torchvision.__version__}
ok = True
for k in want:
    flag = "OK " if have[k] == want[k] else "MISMATCH"
    ok &= have[k] == want[k]
    print(f"[env_check] {flag} {k}: have {have[k]}, want {want[k]}")
if not torch.cuda.is_available():
    print("[env_check] MISMATCH CUDA not available"); ok = False
print("[env_check]", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
  sha256sum part_b_resnet_v8.py adam_nv.py noise_proxy_diagnostic.py noise_schedule.py noise.py experiment_metadata.py \
    > "$LOG_DIR/_code_hashes_${TS}.txt"
fi

run_job() {  # usage: run_job <gpu> <label_noise>:<seed>
  local gpu=$1 ln=${2%%:*} seed=${2##*:}
  local tag="ln${ln}_seed${seed}"
  local train_csv="$LOG_DIR/proxy_train_${tag}.csv" diag="$LOG_DIR/proxy_diag_${tag}.jsonl"
  local n_ck; n_ck=$(echo $CHECKPOINTS | wc -w)
  if [ -f "$train_csv" ] && [ "$(wc -l < "$train_csv")" -ge $((EPOCHS + 1)) ] \
     && [ -f "$diag" ] && [ "$(wc -l < "$diag")" -ge "$n_ck" ]; then
    echo "[exp16] skip $tag (complete)"
    return 0
  fi
  local cmd=(python3 noise_proxy_diagnostic.py --seed "$seed" --label-noise "$ln" --epochs "$EPOCHS"
             --checkpoints $CHECKPOINTS --k-batches "$K_BATCHES" --batch-size 128
             --data-dir "$DATA_DIR" --log-dir "$LOG_DIR" --deterministic)
  if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run][GPU $gpu] CUDA_VISIBLE_DEVICES=$gpu ${cmd[*]}"
    return 0
  fi
  local start=$SECONDS status=success err=""
  CUDA_VISIBLE_DEVICES=$gpu "${cmd[@]}" > "$LOG_DIR/stdout_${tag}_${TS}.txt" 2>&1 \
    || { status=failed; err="exit code $?"; }
  echo "${tag},${status},$((SECONDS - start)),${err}" >> "$LOG_DIR/_status_gpu${gpu}_${TS}.csv"
  echo "[exp16] GPU $gpu: $tag -> $status ($((SECONDS - start)) s)"
}

run_list() {  # usage: run_list <gpu> <jobs...>
  local gpu=$1; shift
  echo "job,status,elapsed_sec,error" > "$LOG_DIR/_status_gpu${gpu}_${TS}.csv"
  for job in "$@"; do run_job "$gpu" "$job"; done
}

if [ -n "$JOBS_GPU0" ]; then run_list 0 $JOBS_GPU0 & fi
if [ -n "$JOBS_GPU1" ]; then run_list 1 $JOBS_GPU1 & fi
wait
if [ "$DRY_RUN" = "1" ]; then exit 0; fi
echo "[exp16] runs finished; checking logs"
python3 - "$LOG_DIR" "$EPOCHS" $JOBS_GPU0 $JOBS_GPU1 <<'PY'
import json, os, sys
log_dir, epochs, jobs = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
bad = 0
for job in jobs:
    ln, seed = job.split(":")
    tag = f"ln{ln}_seed{seed}"
    problems = []
    tp, dp = os.path.join(log_dir, f"proxy_train_{tag}.csv"), os.path.join(log_dir, f"proxy_diag_{tag}.jsonl")
    n_train = sum(1 for _ in open(tp)) - 1 if os.path.isfile(tp) else -1
    n_diag = sum(1 for l in open(dp) if l.strip()) if os.path.isfile(dp) else -1
    if n_train != epochs: problems.append(f"{n_train}/{epochs} training epochs")
    if n_diag != 7: problems.append(f"{n_diag}/7 diagnostic records")
    print(f"{tag:22s} {'OK' if not problems else 'FAIL: ' + '; '.join(problems)}")
    bad += bool(problems)
print(f"{len(jobs) - bad}/{len(jobs)} jobs verified")
sys.exit(1 if bad else 0)
PY
