#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$TASK_ROOT"
TASK_WORKERS="${WORKERS:-80}"
TASK_RUN_DIR="${RUN_DIR:-$TASK_ROOT/runs/noise_aware_top5_qwen_20261017}"
export PYTHONUNBUFFERED=1
echo "Stage 1: golden case optimization -> QA clusters -> crowd prompts -> top5 inference, workers=$TASK_WORKERS"
bash run_noise_aware.sh all --run-dir "$TASK_RUN_DIR" --context-budget 5 --workers "$TASK_WORKERS"
echo "Stage 2: original chapter3 mechanical metrics + Qwen-only 1-5 final scores"
"${EVAL_PYTHON:-/pub/yanglw/miniconda3/bin/python}" -m rag_core.final_qwen_evaluation \
    --input "$TASK_RUN_DIR/answers_top5.jsonl" \
    --output-dir "$TASK_RUN_DIR/evaluation_qwen" --workers "$TASK_WORKERS"
echo "FULL_PIPELINE_COMPLETE: $TASK_RUN_DIR/evaluation_qwen/summary.json"
