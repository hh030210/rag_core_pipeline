#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$TASK_ROOT"
export PYTHONPATH="$TASK_ROOT/runtime_deps:$TASK_ROOT${PYTHONPATH:+:$PYTHONPATH}"
TASK_PYTHON="${RAG_PYTHON:-/pub/yanglw/miniconda3/envs/code_generation/bin/python}"
if [[ ! -x "$TASK_PYTHON" ]]; then TASK_PYTHON=python; fi
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-4}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec "$TASK_PYTHON" -m rag_core.case_collective_experiment "$@"
