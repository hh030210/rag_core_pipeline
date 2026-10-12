#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
rag_python="${RAG_PYTHON:-}"

if [[ -z "$rag_python" ]]; then
  for candidate in \
    /home/humq/envs/denoise_qa/bin/python \
    /home/humq/envs/sdu_ds/bin/python \
    "$(command -v python3 2>/dev/null || true)"; do
    [[ -n "$candidate" && -x "$candidate" ]] || continue
    if "$candidate" -c 'import importlib.util; assert importlib.util.find_spec("numpy") and importlib.util.find_spec("qdrant_client") and (importlib.util.find_spec("FlagEmbedding") or importlib.util.find_spec("sentence_transformers"))' >/dev/null 2>&1; then
      rag_python="$candidate"
      break
    fi
  done
fi

if [[ -z "$rag_python" || ! -x "$rag_python" ]]; then
  echo "找不到含 numpy、qdrant-client 和 BGE 编码依赖的 Python 环境。请安装项目 requirements.txt，或设置 RAG_PYTHON。" >&2
  exit 2
fi

exec "$rag_python" "$script_dir/interactive_rag.py" "$@"
