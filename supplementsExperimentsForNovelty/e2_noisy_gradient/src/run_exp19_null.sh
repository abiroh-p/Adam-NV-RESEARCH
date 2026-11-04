#!/usr/bin/env bash
# Experiment 19: null run. ResNet-18, 20% label noise, Adam-NV code with gamma = 0 and a constant beta_1 = 0.9
# (--optimizer adam_nv_fixed --fixed-beta1 0.9), seeds 0-7: the trajectory floor of the Adam-NV code path against torch Adam.
# Runs part_b_resnet_v8.py directly (unmodified). Every call needs its own LOG_DIR (file names equal those of the beta1 = 0.81 control).
#
#   SEEDS_GPU0="0 1" SEEDS_GPU1="2 3" DATA_DIR=/kaggle/working/Adam-NV/data \
#   LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp19_null_A bash run_exp19_null.sh
#
# EPOCHS defaults to 60, DRY_RUN=1 prints the commands only. Complete logs are skipped on re-run.
set -u
cd "$(dirname "$0")"
DATA_DIR="${DATA_DIR:-../data}"; LOG_DIR="${LOG_DIR:-../logs_exp19}"; EPOCHS="${EPOCHS:-60}"
SEEDS_GPU0="${SEEDS_GPU0:-}"; SEEDS_GPU1="${SEEDS_GPU1:-}"; DRY_RUN="${DRY_RUN:-0}"
if [ -z "$SEEDS_GPU0$SEEDS_GPU1" ]; then echo "[exp19] No seeds given. Set SEEDS_GPU0 and/or SEEDS_GPU1."; exit 2; fi
mkdir -p "$LOG_DIR"; TS=$(date +%Y%m%d_%H%M%S)
if [ "$DRY_RUN" != "1" ]; then
  python3 - <<'PY' || { echo "[exp19] Environment check failed; not starting runs."; exit 2; }
import sys
import torch, torchvision
want = {"python": "3.12.13", "torch": "2.10.0+cu128", "torchvision": "0.25.0+cu128"}
have = {"python": sys.version.split()[0], "torch": torch.__version__, "torchvision": torchvision.__version__}
ok = True
for k in want:
    flag = "OK " if have[k] == want[k] else "MISMATCH"; ok &= have[k] == want[k]
    print(f"[env_check] {flag} {k}: have {have[k]}, want {want[k]}")
if not torch.cuda.is_available(): print("[env_check] MISMATCH CUDA not available"); ok = False
else: print("[env_check] OK  GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("[env_check]", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
PY
  sha256sum part_b_resnet_v8.py adam_nv.py noise_schedule.py noise.py experiment_metadata.py > "$LOG_DIR/_code_hashes_${TS}.txt"
fi
run_job() {  # usage: run_job <gpu> <seed>
  local gpu=$1 opt=adam_nv_fixed seed=$2
  local base="${opt}_scheduleconstant_sigma0.0_ln0.2_seed${seed}" csv
  csv="$LOG_DIR/${base}.csv"
  if [ -f "$csv" ] && [ "$(wc -l < "$csv")" -ge $((EPOCHS + 1)) ]; then echo "[exp19] skip seed $seed (complete log exists)"; return 0; fi
  local cmd=(python3 part_b_resnet_v8.py --optimizer "$opt" --fixed-beta1 0.9
             --schedule constant --sigma 0 --label-noise 0.2 --label-diagnostic-interval 1
             --seed "$seed" --epochs "$EPOCHS" --batch-size 128 --data-dir "$DATA_DIR" --log-dir "$LOG_DIR" --deterministic)
  if [ "$DRY_RUN" = "1" ]; then echo "[dry-run][GPU $gpu] CUDA_VISIBLE_DEVICES=$gpu ${cmd[*]}"; return 0; fi
  local start=$SECONDS status=success err=""
  CUDA_VISIBLE_DEVICES=$gpu "${cmd[@]}" > "$LOG_DIR/stdout_${base}_${TS}.txt" 2>&1 || { status=failed; err="exit code $?"; }
  echo "${opt},${seed},${status},$((SECONDS - start)),${err}" >> "$LOG_DIR/_status_gpu${gpu}_${TS}.csv"
  echo "[exp19] GPU $gpu: seed $seed -> $status ($((SECONDS - start)) s)"
}
run_list() { local gpu=$1; shift; echo "optimizer,seed,status,elapsed_sec,error" > "$LOG_DIR/_status_gpu${gpu}_${TS}.csv"; for s in "$@"; do run_job "$gpu" "$s"; done; }
if [ -n "$SEEDS_GPU0" ]; then run_list 0 $SEEDS_GPU0 & fi
if [ -n "$SEEDS_GPU1" ]; then run_list 1 $SEEDS_GPU1 & fi
wait
if [ "$DRY_RUN" = "1" ]; then exit 0; fi
echo "[exp19] runs finished; checking logs"
JOBS=""; for s in $SEEDS_GPU0 $SEEDS_GPU1; do JOBS="$JOBS adam_nv_fixed:$s"; done
python3 check_exp07_logs.py --log-dir "$LOG_DIR" --epochs "$EPOCHS" --fixed-beta1 0.9 --jobs $JOBS
