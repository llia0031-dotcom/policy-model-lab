#!/usr/bin/env bash
# Run the GPU stages one after another (one GPU, 8 GB: jobs must not overlap). Every step skips finished work,
# so the queue can be started again after an interruption. Core results first.
# Each step's full output goes to $POLICY_LAB_DATA/logs/<step>.log; a failed step is reported and the queue
# moves on.
# Usage: bash scripts/gpu_queue.sh
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
LOG="${POLICY_LAB_DATA:-data}/logs"
mkdir -p "$LOG"
step() {
  local name rc=0
  name=$(echo "$*" | tr -c 'A-Za-z0-9.-' '_')
  echo ">> $*  ($(date +%H:%M))"
  PYTHONUNBUFFERED=1 "$PY" "$@" > "$LOG/$name.log" 2>&1 || rc=$?
  grep -vE "Loading weights|^\s*$|INFO|FutureWarning|UserWarning|warnings.warn" "$LOG/$name.log" | tail -4
  [ "$rc" -eq 0 ] || echo "!! step failed (exit $rc): $*"
}
done_lora() { [ -f "${POLICY_LAB_DATA:-data}/lora/$1/adapter_config.json" ]; }
step -m lab.frames
step -m lab.local_vlm --model qwen35-4b
done_lora qwen3-1.7b-lora || step finetune/train_lora.py --base Qwen/Qwen3-1.7B --out qwen3-1.7b-lora
step finetune/predict.py --method qwen3-1.7b-zs
step finetune/predict.py --method qwen3-1.7b-lora
done_lora qwen3-0.6b-lora || step finetune/train_lora.py --base Qwen/Qwen3-0.6B --out qwen3-0.6b-lora
step finetune/predict.py --method qwen3-0.6b-zs
step finetune/predict.py --method qwen3-0.6b-lora
[ -f "${POLICY_LAB_DATA:-data}/encoder/mmbert-base/config.json" ] || step finetune/train_encoder.py
step finetune/predict.py --method mmbert
step -m lab.local_vlm --model qwen3vl-2b
step -m lab.siglip_frames --plan frames8
step -m lab.siglip_frames --plan frames16
step -m lab.kuaimod local
echo "queue done $(date +%H:%M)"
