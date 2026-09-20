"""Evaluate only the dimension route; no semantic model or vector scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_core.embedding import EmbeddingModel
from rag_core.evaluation import (
    aggregate_summary,
    evaluate_row,
    load_chunks,
    load_gold_records,
    map_gold_to_chunks,
)
from rag_core.retrieval import Retriever
from rag_core.settings import Settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--threshold-profile", default="global_058",
                        choices=("global_058", "typed_moderate"))
    parser.add_argument("--disable-poi-registry", action="store_true",
                        help="Run the direct-string subject filter baseline.")
    parser.add_argument("--disable-fact-anchor-rerank", action="store_true",
                        help="Disable query literal / POI fact tie-breaking bonus.")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    settings = Settings.from_env(
        args.run_dir, backend="local", mock=args.mock, vector_dim=1024,
        dimension_pool=100, model_path=args.model_path, embedding_device=args.device,
    )
    settings.tag_vector_threshold_profile = args.threshold_profile
    settings.poi_scope_enabled = not args.disable_poi_registry
    settings.fact_anchor_rerank_enabled = not args.disable_fact_anchor_rerank
    retriever = Retriever(
        run_dir=args.run_dir,
        settings=settings,
        embeddings=EmbeddingModel(
            args.model_path, args.device, dimension=1024, mock=args.mock,
        ),
        llm=None,
    )
    # Reuse the exact gold-to-chunk mapping from the baseline when available;
    # this makes the comparison independent of later fuzzy-mapping changes.
    baseline_results = args.run_dir / "evaluation" / "results.jsonl"
    if baseline_results.exists():
        records = []
        for line in baseline_results.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            records.append({
                "id": row["id"],
                "question": row.get("question", ""),
                "gold_chunk_ids": row.get("gold", {}).get("chunk_ids", []),
                "gold_evidence_texts": row.get("gold", {}).get("evidence_texts", []),
                "gold_resolution": row.get("gold", {}).get("gold_resolution", {}),
                "gold_relevance": {},
                "reference_answer": row.get("reference_answer", ""),
                "question_type": row.get("question_type", ""),
                "spot": row.get("spot", ""),
                "answerable": row.get("answerable", True),
            })
    else:
        chunks = load_chunks(args.run_dir)
        records = [
            map_gold_to_chunks(record, chunks)
            for record in load_gold_records(args.dataset)
        ]
    ks = (1, 3, 5, 10, 20)
    rows = []
    empty_candidates = 0
    for index, record in enumerate(records, 1):
        retrieval = retriever.search(
            record["question"], top_k=5, dimension_pool=100,
            dimension_only=True,
        )
        if not retrieval["dimension_candidates"]:
            empty_candidates += 1
        rows.append(evaluate_row(record, retrieval, ks, 20, 800))
        if index % 100 == 0:
            print(f"[维度评测] {index}/{len(records)}", flush=True)

    summary = aggregate_summary(rows, ks)
    summary["dimension_diagnostics"] = {
        "empty_candidate_queries": empty_candidates,
        "empty_candidate_rate": round(empty_candidates / len(rows), 8) if rows else 0.0,
        "policy": "wide recall; main-dimension rerank; canonical labels and parent coverage bonus",
        "scope": "parent_document",
        "embedding_mode": "mock" if args.mock else (args.model_path or "configured_default"),
        "tag_vector_threshold_profile": args.threshold_profile,
        "poi_registry_enabled": settings.poi_scope_enabled,
        "fact_anchor_rerank_enabled": settings.fact_anchor_rerank_enabled,
    }
    (args.output / "results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps({
        "total": summary["total"],
        "gold_mapped": summary["gold_mapped"],
        "dimension": summary["routes"]["dimension"],
        "dimension_diagnostics": summary["dimension_diagnostics"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
