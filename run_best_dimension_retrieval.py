"""One-command reproducible evaluation for the best dimension retriever.

The command uses only local retrieval logic and cached query facts.  It never
calls an LLM.  If BGE-M3 is absent it downloads the public BAAI/bge-m3 model
once into ``models/bge-m3``; pass ``--model-path`` to reuse a shared model.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "result" / "hybrid_legacy_tags_sdu_entities_20261002_verified_facts"
DATASET = RUN_DIR / "merged_7_rag_test_set_filtered.json"
FACT_CACHE = RUN_DIR / "query_fact_cache_rewritten_v1.json"
DEFAULT_MODEL = ROOT / "models" / "bge-m3"


def ensure_model(model_path: Path) -> None:
    if model_path.exists():
        return
    print(f"BGE-M3 not found at {model_path}; downloading public BAAI/bge-m3 once...", flush=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: pip install -r requirements.txt") from exc
    snapshot_download(repo_id="BAAI/bge-m3", local_dir=str(model_path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=None,
                        help="Existing local BGE-M3 path. Defaults to models/bge-m3.")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "result" / "evaluation_best_dimension_repro",
                        help="Evaluation output directory.")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--skip-model-download", action="store_true",
                        help="Fail with a clear message if the model is missing.")
    args = parser.parse_args()

    for required in (RUN_DIR, DATASET, FACT_CACHE):
        if not required.exists():
            raise FileNotFoundError(
                f"Required reproducibility asset is missing: {required}. "
                "Run this script from a complete repository clone."
            )
    model_path = args.model_path or Path(os.getenv("BGE_MODEL_PATH", DEFAULT_MODEL))
    if not model_path.exists():
        if args.skip_model_download:
            raise FileNotFoundError(f"BGE-M3 model not found: {model_path}")
        ensure_model(model_path)

    command = [
        sys.executable, str(ROOT / "evaluate_dimension_only.py"),
        "--run-dir", str(RUN_DIR),
        "--dataset", str(DATASET),
        "--output", str(args.output),
        "--model-path", str(model_path),
        "--device", args.device,
        "--threshold-profile", "global_058",
        "--enable-no-candidate-lexical-fallback",
        "--enable-verified-fact-index",
        "--query-fact-cache", str(FACT_CACHE),
        "--verified-fact-tail-start", "10",
        "--enable-verified-fact-evidence-rerank",
        "--verified-fact-evidence-rerank-top-n", "20",
        "--enable-answer-evidence-rerank",
        "--answer-evidence-rerank-top-n", "20",
        "--enable-answerability-rerank",
        "--answerability-rerank-top-n", "20",
        "--enable-slot-coverage-rerank",
        "--slot-coverage-rerank-top-n", "20",
        "--use-rewritten-query",
    ]
    print("Running best dimension retrieval evaluation (LLM disabled)...", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"Done. Results: {args.output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
