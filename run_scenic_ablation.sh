#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$TASK_ROOT"
export PYTHONPATH="$TASK_ROOT/runtime_deps:$TASK_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-8}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec "${RAG_PYTHON:-/pub/yanglw/miniconda3/envs/code_generation/bin/python}" -m rag_core.scenic_ablation "$@"
