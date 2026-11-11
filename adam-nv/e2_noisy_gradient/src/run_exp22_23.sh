#!/usr/bin/env bash
# Sequential pilot calibration, then paired experiments across two GPUs.
# Place overlay into the existing Adam-NV tree first (see runbook).
set -euo pipefail
cd "$(dirname "$0")"
DATA_DIR="${DATA_DIR:-../data}"
LOG_DIR="${LOG_DIR:-../logs_exp22_23}"
EPOCHS="${EPOCHS:-60}"
PILOT_SEED="${PILOT_SEED:-100}"
EVAL_SEEDS="${EVAL_SEEDS:-0 1 2}"
METHODS="${METHODS:-adam_nv frozen adam}"
GPUS="${GPUS:-0 1}"
DRY_RUN="${DRY_RUN:-0}"
SAMPLE_EVERY="${SAMPLE_EVERY:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-0.001}"
LABEL_NOISE="${LABEL_NOISE:-0.2}"
SMOKE="${SMOKE:-0}"
mkdir -p "$LOG_DIR"
LOG_DIR=$(cd "$LOG_DIR" && pwd)
DATA_DIR=$(mkdir -p "$DATA_DIR" && cd "$DATA_DIR" && pwd)
read -r -a seeds <<< "$EVAL_SEEDS"
read -r -a methods <<< "$METHODS"
read -r -a gpus <<< "$GPUS"
if [ ${#gpus[@]} -lt 1 ]; then echo 'No GPU specified'; exit 2; fi
for s in "${seeds[@]}"; do if [ "$s" = "$PILOT_SEED" ]; then echo 'Pilot seed overlaps evaluation seed'; exit 2; fi; done
for m in "${methods[@]}"; do case "$m" in adam_nv|frozen|adam) ;; *) echo "Unsupported method $m"; exit 2;; esac; done
if [ "$DRY_RUN" != "1" ]; then
  python3 - <<'PY'
import sys, torch, torchvision
print('[environment]',sys.version.split()[0], torch.__version__, torchvision.__version__)
if not torch.cuda.is_available():
  raise SystemExit('CUDA unavailable. For CPU smoke tests use run_exp22_23.py directly.')
PY
  sha256sum part_b_resnet_v8.py adam_nv.py frozen_tensor_adam.py run_exp22_23.py run_exp22_23.sh > "$LOG_DIR/_code_hashes.txt"
fi
extra=()
if [ "$SMOKE" = 1 ]; then extra+=(--smoke-test); fi
CHECKPOINTS="${CHECKPOINTS:-1 5 10 20 30 45 60}"
read -r -a checkpoints <<< "$CHECKPOINTS"
for e in "${checkpoints[@]}"; do if [ "$e" -gt "$EPOCHS" ]; then echo "CHECKPOINTS contains epoch $e > EPOCHS=$EPOCHS"; exit 2; fi; done
pilot="$LOG_DIR/pilot_ln${LABEL_NOISE}_seed${PILOT_SEED}.json"
run() {
  local gpu="$1" mode="$2" seed="$3" tag="${2}_ln${LABEL_NOISE}_seed${3}"
  local csv="$LOG_DIR/trajectory_${tag}.csv"
  if [ -s "$csv" ] && [ "$(($(wc -l < "$csv") - 1))" -eq "$EPOCHS" ]; then
    if [ "$mode" != pilot ] || [ -s "$pilot" ]; then
      echo "[skip] $tag complete"; return 0
    fi
  fi
  local cmd=(python3 run_exp22_23.py --mode "$mode" --seed "$seed" --eval-seeds "${seeds[@]}"
    --epochs "$EPOCHS" --label-noise "$LABEL_NOISE" --batch-size "$BATCH_SIZE" --lr "$LR"
    --sample-every "$SAMPLE_EVERY" --checkpoints "${checkpoints[@]}" --data-dir "$DATA_DIR"
    --log-dir "$LOG_DIR" --deterministic "${extra[@]}")
  if [ "$mode" = frozen ]; then cmd+=(--pilot-map "$pilot"); fi
  if [ "$DRY_RUN" = 1 ]; then
    printf '[dry-run][GPU %s] CUDA_VISIBLE_DEVICES=%s ' "$gpu" "$gpu"
    printf '%q ' "${cmd[@]}"; echo; return 0
  fi
  echo "[start] GPU $gpu $tag"
  if CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$LOG_DIR/stdout_${tag}.txt" 2>&1; then
    echo "[success] $tag"; echo "$tag,success" >> "$LOG_DIR/_status.csv"
  else
    echo "[FAILED] $tag -- see $LOG_DIR/stdout_${tag}.txt" >&2
    echo "$tag,failed" >> "$LOG_DIR/_status.csv"
    return 1
  fi
}
# The independent pilot must finish before frozen jobs begin.
run "${gpus[0]}" pilot "$PILOT_SEED"
if [ "$DRY_RUN" != 1 ] && [ ! -s "$pilot" ]; then echo "Missing pilot map $pilot"; exit 1; fi
jobs=()
for s in "${seeds[@]}"; do for m in "${methods[@]}"; do jobs+=("$m:$s"); done; done
# Each GPU receives a deterministic subset, preserving a simple resume behavior.
pids=()
for i in "${!gpus[@]}"; do
  (
    for j in "${!jobs[@]}"; do
      if [ $((j % ${#gpus[@]})) -eq "$i" ]; then
        entry="${jobs[$j]}"; run "${gpus[$i]}" "${entry%%:*}" "${entry##*:}" || exit 1
      fi
    done
  ) &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [ "$DRY_RUN" = 1 ]; then exit 0; fi
python3 - "$LOG_DIR" "$EPOCHS" "$LABEL_NOISE" "$EVAL_SEEDS" "$METHODS" <<'PY'
import csv, os, sys
path, epochs, noise, seeds, methods = sys.argv[1], int(sys.argv[2]),sys.argv[3],sys.argv[4].split(),sys.argv[5].split()
bad = 0
for s in seeds:
 for m in methods:
  fname = os.path.join(path,f'trajectory_{m}_ln{noise}_seed{s}.csv')
  try:
   with open(fname,newline='') as f: rows=list(csv.DictReader(f))
   assert len(rows)==epochs and [int(r['epoch']) for r in rows]==list(range(1,epochs+1))
   assert all(r['train_loss'] and r['val_acc'] for r in rows)
   print('[OK]', m, s, epochs, 'epochs')
  except Exception as ex:
   bad += 1; print('[FAIL]', m, s,repr(ex))
if bad: sys.exit(1)
PY
if [ "$failed" -ne 0 ]; then echo "One or more runs failed"; exit 1; fi
echo "All evaluation runs verified. Run: python3 analysis/analyze_exp22_23.py --dir e2_noisy_gradient/logs_exp22_23"
