#!/usr/bin/env bash

set -u

root="${QA_EXPERIMENT_ROOT:?QA_EXPERIMENT_ROOT is required}"
dataset="${QA_DATASET:?QA_DATASET is required}"
base_url="${LLM_BASE_URL:?LLM_BASE_URL is required}"
model="${LLM_MODEL:?LLM_MODEL is required}"
api_key="${LLM_API_KEY:?LLM_API_KEY is required}"

for k in 10 15 20 30 50 96; do
  (
    directory="$root/k_$k"
    while true; do
      if test -f "$directory/qa_summary.json" && grep -q '"completed": 435' "$directory/qa_summary.json"; then
        break
      fi
      sleep 30
    done
    mkdir -p "$directory/qa_evaluation"
    /home/humq/envs/denoise_qa/bin/python -u -m rag_core.qa_evaluation \
      --input "$directory/qa_results.jsonl" \
      --output-dir "$directory/qa_evaluation" \
      --base-url "$base_url" \
      --model "$model" \
      --api-key "$api_key" \
      --interval 1 \
      --concurrency 1 > "$directory/qa_evaluation/run.log" 2>&1
  ) &
done

wait
