#!/usr/bin/env bash
# Experiment 20: constant beta_1 on a subset of tensors (plain Adam elsewhere), 20% label noise, ResNet-18, seeds 0-7.
# Runs part_b_resnet_v8_constsubset.py (a wrapper; part_b_resnet_v8.py and adam_nv.py are unchanged).
# One call = one constant and one subset; every call needs its own LOG_DIR (file names are those of a normal Adam-NV run).
#
#   SUBSET=onedim CONST_BETA1=0.79 SEEDS_GPU0="0 1" SEEDS_GPU1="2 3" \
#   DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp20_onedim_c079_A \
#   bash run_exp20_constsubset.sh
#
# SUBSET is onedim (rank <= 1) | rest (rank >= 2). EPOCHS defaults to 60, DRY_RUN=1 prints the commands only.
# Complete logs (EPOCHS data rows plus the sidecar) are skipped on re-run.
set -u
cd "$(dirname "$0")"
DATA_DIR="${DATA_DIR:-../data}"; LOG_DIR="${LOG_DIR:-../logs_exp20}"; EPOCHS="${EPOCHS:-60}"
SUBSET="${SUBSET:-onedim}"; CONST_BETA1="${CONST_BETA1:-}"; SEEDS_GPU0="${SEEDS_GPU0:-}"; SEEDS_GPU1="${SEEDS_GPU1:-}"; DRY_RUN="${DRY_RUN:-0}"
case "$SUBSET" in onedim|rest) ;; *) echo "[exp20] Set SUBSET to onedim or rest."; exit 2;; esac
if [ -z "$CONST_BETA1" ]; then echo "[exp20] Set CONST_BETA1 (for example 0.79)."; exit 2; fi
if [ -z "$SEEDS_GPU0$SEEDS_GPU1" ]; then echo "[exp20] No seeds given. Set SEEDS_GPU0 and/or SEEDS_GPU1."; exit 2; fi
mkdir -p "$LOG_DIR"; TS=$(date +%Y%m%d_%H%M%S)
if [ "$DRY_RUN" != "1" ]; then
  python3 - <<'PY' || { echo "[exp20] Environment check failed; not starting runs."; exit 2; }
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
  sha256sum part_b_resnet_v8.py part_b_resnet_v8_constsubset.py adam_nv.py adam_const_subset.py noise_schedule.py noise.py experiment_metadata.py > "$LOG_DIR/_code_hashes_${TS}.txt"
fi
run_job() {  # usage: run_job <gpu> <seed>
  local gpu=$1 opt=adam_nv seed=$2
  local base="${opt}_scheduleconstant_sigma0.0_ln0.2_seed${seed}" csv
  csv="$LOG_DIR/${base}.csv"
  if [ -f "$csv" ] && [ "$(wc -l < "$csv")" -ge $((EPOCHS + 1)) ]; then echo "[exp20] skip seed $seed (complete log exists)"; return 0; fi
  local cmd=(python3 part_b_resnet_v8_constsubset.py --subset "$SUBSET" --const-beta1 "$CONST_BETA1" --optimizer "$opt"
             --schedule constant --sigma 0 --label-noise 0.2 --label-diagnostic-interval 1
             --seed "$seed" --epochs "$EPOCHS" --batch-size 128 --data-dir "$DATA_DIR" --log-dir "$LOG_DIR" --deterministic)
  if [ "$DRY_RUN" = "1" ]; then echo "[dry-run][GPU $gpu] CUDA_VISIBLE_DEVICES=$gpu ${cmd[*]}"; return 0; fi
  local start=$SECONDS status=success err=""
  CUDA_VISIBLE_DEVICES=$gpu "${cmd[@]}" > "$LOG_DIR/stdout_${base}_${TS}.txt" 2>&1 || { status=failed; err="exit code $?"; }
  echo "const_${SUBSET}_${CONST_BETA1},${seed},${status},$((SECONDS - start)),${err}" >> "$LOG_DIR/_status_gpu${gpu}_${TS}.csv"
  echo "[exp20] GPU $gpu: seed $seed -> $status ($((SECONDS - start)) s)"
}
run_list() { local gpu=$1; shift; echo "optimizer,seed,status,elapsed_sec,error" > "$LOG_DIR/_status_gpu${gpu}_${TS}.csv"; for s in "$@"; do run_job "$gpu" "$s"; done; }
if [ -n "$SEEDS_GPU0" ]; then run_list 0 $SEEDS_GPU0 & fi
if [ -n "$SEEDS_GPU1" ]; then run_list 1 $SEEDS_GPU1 & fi
wait
if [ "$DRY_RUN" = "1" ]; then exit 0; fi
echo "[exp20] runs finished; checking logs"
python3 check_exp20_logs.py --log-dir "$LOG_DIR" --epochs "$EPOCHS" --subset "$SUBSET" --const-beta1 "$CONST_BETA1" --seeds $SEEDS_GPU0 $SEEDS_GPU1
