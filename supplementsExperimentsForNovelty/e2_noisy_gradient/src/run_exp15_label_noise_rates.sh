#!/usr/bin/env bash
# Experiment 15: Adam versus Adam-NV at other label-noise rates (10% and 40%), seeds 0-5.
# Uses the EXISTING part_b_resnet_v8.py unchanged; only the noise rate differs from Exp. 7.
#
# Jobs are "<optimizer>:<seed>" pairs with optimizer one of adam, adam_nv. One call = one noise
# rate (LN), and each rate needs its own LOG_DIR.
#   LN=0.4 JOBS_GPU0="adam:0 adam:1 adam_nv:0 adam_nv:1" JOBS_GPU1="adam:2 adam:3 adam_nv:2 adam_nv:3" \
#   DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp15_ln04_A \
#   bash run_exp15_label_noise_rates.sh
#
# Environment variables: LN (required, 0.1 or 0.4), JOBS_GPU0, JOBS_GPU1, DATA_DIR (default ../data),
# LOG_DIR (default ../logs_exp15), EPOCHS (default 60), DRY_RUN=1 (print the commands only).
# Complete logs (EPOCHS data rows) are skipped, so re-running after an interrupted session is safe.

set -u
cd "$(dirname "$0")"

DATA_DIR="${DATA_DIR:-../data}"
LOG_DIR="${LOG_DIR:-../logs_exp15}"
EPOCHS="${EPOCHS:-60}"
LN="${LN:-}"
JOBS_GPU0="${JOBS_GPU0:-}"
JOBS_GPU1="${JOBS_GPU1:-}"
DRY_RUN="${DRY_RUN:-0}"

case "$LN" in 0.1|0.4) ;; *) echo "[exp15] Set LN to 0.1 or 0.4."; exit 2;; esac
if [ -z "$JOBS_GPU0$JOBS_GPU1" ]; then
  echo "[exp15] No jobs given. Set JOBS_GPU0 and/or JOBS_GPU1 (see the header of this script)."
  exit 2
fi

mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

if [ "$DRY_RUN" != "1" ]; then
  python3 - <<'PY' || { echo "[exp15] Environment check failed; not starting runs."; exit 2; }
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
else:
    print("[env_check] OK  GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("[env_check]", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
  sha256sum part_b_resnet_v8.py adam_nv.py noise_schedule.py noise.py experiment_metadata.py \
    > "$LOG_DIR/_code_hashes_${TS}.txt"
fi

run_job() {  # usage: run_job <gpu> <optimizer>:<seed>
  local gpu=$1 opt=${2%%:*} seed=${2##*:}
  local base="${opt}_scheduleconstant_sigma0.0_ln${LN}_seed${seed}"
  local csv="$LOG_DIR/${base}.csv"
  if [ -f "$csv" ] && [ "$(wc -l < "$csv")" -ge $((EPOCHS + 1)) ]; then
    echo "[exp15] skip $opt seed $seed (complete log exists)"
    return 0
  fi
  local cmd=(python3 part_b_resnet_v8.py --optimizer "$opt"
             --schedule constant --sigma 0 --label-noise "$LN" --label-diagnostic-interval 1
             --seed "$seed" --epochs "$EPOCHS" --batch-size 128
             --data-dir "$DATA_DIR" --log-dir "$LOG_DIR" --deterministic)
  if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run][GPU $gpu] CUDA_VISIBLE_DEVICES=$gpu ${cmd[*]}"
    return 0
  fi
  local start=$SECONDS status=success err=""
  CUDA_VISIBLE_DEVICES=$gpu "${cmd[@]}" > "$LOG_DIR/stdout_${base}_${TS}.txt" 2>&1 \
    || { status=failed; err="exit code $?"; }
  echo "${opt},${seed},${status},$((SECONDS - start)),${err}" >> "$LOG_DIR/_status_gpu${gpu}_${TS}.csv"
  echo "[exp15] GPU $gpu: $opt seed $seed -> $status ($((SECONDS - start)) s)"
}

run_list() {  # usage: run_list <gpu> <jobs...>
  local gpu=$1; shift
  echo "optimizer,seed,status,elapsed_sec,error" > "$LOG_DIR/_status_gpu${gpu}_${TS}.csv"
  for job in "$@"; do run_job "$gpu" "$job"; done
}

if [ -n "$JOBS_GPU0" ]; then run_list 0 $JOBS_GPU0 & fi
if [ -n "$JOBS_GPU1" ]; then run_list 1 $JOBS_GPU1 & fi
wait

if [ "$DRY_RUN" = "1" ]; then exit 0; fi
echo "[exp15] runs finished; checking logs"
python3 check_exp15_logs.py --log-dir "$LOG_DIR" --epochs "$EPOCHS" --ln "$LN" \
    --jobs $JOBS_GPU0 $JOBS_GPU1
