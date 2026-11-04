#!/usr/bin/env bash
# Experiment 21: noise-proxy diagnostic v2 (Adam-NV's score versus directly measured gradient noise, within tensors).
# Runs noise_proxy_diagnostic_v2.py: trains a ResNet-18 with Adam-NV (60 epochs, as in the label-noise experiments) and measures
# minibatch gradient noise (K = 64 batches in groups, split halves, absolute scale, lambda = 0 score, last-64-step stream) after
# epochs 1, 5, 10, 20, 30, 45 and 60.
#
# Jobs are "<label_noise>:<seed>" pairs, run sequentially per GPU. Every call needs its OWN LOG_DIR.
#   JOBS_GPU0="0.0:0 0.2:0 0.0:1" JOBS_GPU1="0.2:1 0.0:2 0.2:2" \
#   DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp21_A \
#   bash run_exp21_proxy_v2.sh
#
# Environment variables: JOBS_GPU0, JOBS_GPU1, DATA_DIR (default ../data), LOG_DIR (default ../logs_exp21), EPOCHS (60), K_BATCHES (64),
# REF_DIRS (space-separated folders holding the earlier 20%-noise Adam-NV logs, for the training-equivalence check), DRY_RUN=1 (print the commands only). A complete job (60 train rows, 7 diagnostic records) is skipped on re-run.
set -u
cd "$(dirname "$0")"
DATA_DIR="${DATA_DIR:-../data}"; LOG_DIR="${LOG_DIR:-../logs_exp21}"; EPOCHS="${EPOCHS:-60}"; K_BATCHES="${K_BATCHES:-64}"
CHECKPOINTS="1 5 10 20 30 45 60"; JOBS_GPU0="${JOBS_GPU0:-}"; JOBS_GPU1="${JOBS_GPU1:-}"; DRY_RUN="${DRY_RUN:-0}"
if [ -z "$JOBS_GPU0$JOBS_GPU1" ]; then echo "[exp21] No jobs given. Set JOBS_GPU0 and/or JOBS_GPU1 (see the header of this script)."; exit 2; fi
mkdir -p "$LOG_DIR"; TS=$(date +%Y%m%d_%H%M%S)
if [ "$DRY_RUN" != "1" ]; then
  python3 - <<'PY' || { echo "[exp21] Environment check failed; not starting runs."; exit 2; }
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
  sha256sum part_b_resnet_v8.py adam_nv.py noise_proxy_diagnostic.py noise_proxy_diagnostic_v2.py noise_schedule.py noise.py experiment_metadata.py > "$LOG_DIR/_code_hashes_${TS}.txt"
fi
run_job() {  # usage: run_job <gpu> <label_noise>:<seed>
  local gpu=$1 ln=${2%%:*} seed=${2##*:}
  local tag="ln${ln}_seed${seed}" n_ck; n_ck=$(echo $CHECKPOINTS | wc -w)
  local train_csv="$LOG_DIR/proxy2_train_${tag}.csv" diag="$LOG_DIR/proxy2_diag_${tag}.jsonl"
  if [ -f "$train_csv" ] && [ "$(wc -l < "$train_csv")" -ge $((EPOCHS + 1)) ] && [ -f "$diag" ] && [ "$(wc -l < "$diag")" -ge "$n_ck" ]; then
    echo "[exp21] skip $tag (complete)"; return 0
  fi
  local cmd=(python3 noise_proxy_diagnostic_v2.py --seed "$seed" --label-noise "$ln" --epochs "$EPOCHS" --checkpoints $CHECKPOINTS
             --k-batches "$K_BATCHES" --stream-steps 64 --batch-size 128 --data-dir "$DATA_DIR" --log-dir "$LOG_DIR" --deterministic)
  if [ "$DRY_RUN" = "1" ]; then echo "[dry-run][GPU $gpu] CUDA_VISIBLE_DEVICES=$gpu ${cmd[*]}"; return 0; fi
  local start=$SECONDS status=success err=""
  CUDA_VISIBLE_DEVICES=$gpu "${cmd[@]}" > "$LOG_DIR/stdout_${tag}_${TS}.txt" 2>&1 || { status=failed; err="exit code $?"; }
  echo "${tag},${status},$((SECONDS - start)),${err}" >> "$LOG_DIR/_status_gpu${gpu}_${TS}.csv"
  echo "[exp21] GPU $gpu: $tag -> $status ($((SECONDS - start)) s)"
}
run_list() { local gpu=$1; shift; echo "job,status,elapsed_sec,error" > "$LOG_DIR/_status_gpu${gpu}_${TS}.csv"; for j in "$@"; do run_job "$gpu" "$j"; done; }
if [ -n "$JOBS_GPU0" ]; then run_list 0 $JOBS_GPU0 & fi
if [ -n "$JOBS_GPU1" ]; then run_list 1 $JOBS_GPU1 & fi
wait
if [ "$DRY_RUN" = "1" ]; then exit 0; fi
echo "[exp21] runs finished; checking logs"
python3 check_exp21_logs.py --log-dir "$LOG_DIR" --epochs "$EPOCHS" --k-batches "$K_BATCHES" --jobs $JOBS_GPU0 $JOBS_GPU1 \
  --reference-dirs ${REF_DIRS:-../../Label_Diagnostic ../../logs_exp07_A ../../logs_exp07_B ../../exp07_log_C/log_exp07_C ../Label_Diagnostic ../logs_exp07_A ../logs_exp07_B ../exp07_log_C/log_exp07_C}
