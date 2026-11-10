#!/usr/bin/env python3
"""Evaluate scenic retrieval with the frozen local index and dimension candidates."""
from __future__ import annotations

from copy import deepcopy
from importlib import import_module
import json
import math
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPOSITORY))

from rag_core.retrieval_fusion.experiments.common.scenic_query_replay import (  # noqa: E402
    ScenicRetriever,
    ScenicRetrievalRuntime,
)

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = EXPERIMENT_DIR / "output" / "runs"
SNAPSHOT = Path("/home/humq/rag_core_runs/scenic_best_dimension_20261012_local_sync")
INDEX_DIR = SNAPSHOT / "result/hybrid_legacy_tags_sdu_entities_20261002_verified_facts"
DATASET = INDEX_DIR / "merged_7_rag_test_set_filtered.json"
CANDIDATES = SNAPSHOT / "dimension_retrieval_best_20261012/dimension_candidates_top20_for_fusion.jsonl"


def load_candidates():
    archive = {}
    with CANDIDATES.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = (str(row["id"]), row["question"])
            if key in archive:
                raise ValueError(f"Duplicate question in {CANDIDATES}: {key[0]}")
            hits = row["dimension_candidates"]
            ids = [str(hit["chunk_id"]) for hit in hits]
            if len(ids) != len(set(ids)) or any(
                hit["rank"] != rank or not math.isfinite(float(hit["score"]))
                for rank, hit in enumerate(hits, 1)
            ):
                raise ValueError(f"Invalid frozen ranks or scores for question {key[0]}")
            archive[key] = row
    return archive


def validate_index(archive, chunks_by_id):
    def compact_text(chunk):
        return "".join(str(chunk.get("chunk_text_full") or chunk.get("chunk_text") or "").split())

    texts = {chunk_id: compact_text(chunk) for chunk_id, chunk in chunks_by_id.items()}
    for row in archive.values():
        for hit in row["dimension_candidates"]:
            chunk_id = str(hit["chunk_id"])
            # The archive stores at most the first 800 characters of each chunk.
            if chunk_id not in texts or not texts[chunk_id].startswith(compact_text(hit)):
                raise ValueError(f"Frozen candidate does not match local_points.json: {chunk_id}")


class FrozenDimensionRetriever(ScenicRetriever):
    prepare_dimension_vectors = False

    def _retrieve_dimension_candidates(self, *, dimension_pool, **_):
        return deepcopy(self.frozen_candidates[:dimension_pool])


class FrozenDimensionRuntime(ScenicRetrievalRuntime):
    retriever_class = FrozenDimensionRetriever

    def __init__(self, args):
        self.archive = load_candidates()
        super().__init__(args)
        try:
            validate_index(self.archive, self.chunks_by_id)
            self.matched_questions = sum(
                (str(record["id"]), record["question"]) in self.archive for record in self.records
            )
            if not self.matched_questions:
                raise ValueError("No questions in the dataset match the frozen dimension archive")
        except Exception:
            self.close()
            raise

    def retrieve_record(self, record, parts, *, fusion_strategy_file=None):
        row = self.archive.get((str(record["id"]), record["question"]))
        self.retriever.frozen_candidates = row["dimension_candidates"] if row else []
        result = super().retrieve_record(record, parts, fusion_strategy_file=fusion_strategy_file)
        result["dimension_strategy"] = "frozen_candidates"
        result["dimension_retrieval_query"] = row["retrieval_query"] if row else None
        return result

    def metadata(self):
        data = super().metadata()
        data["frozen_dimension"] = {
            "source_commit": "9e5dd32",
            "local_points_path": str(INDEX_DIR / "local_points.json"),
            "candidate_path": str(CANDIDATES),
            "matched_questions": self.matched_questions,
            "missing_questions": len(self.records) - self.matched_questions,
        }
        return data

def main(argv=None):
    main_experiment = import_module(
        "rag_core.retrieval_fusion.experiments.00_main_experiment.code.export_retrieval_details"
    )
    parser = main_experiment.build_parser(output_root=OUTPUT_ROOT)
    parser.description = __doc__
    parser.set_defaults(run_dir=INDEX_DIR, dataset=DATASET, backend="local", collection="rag_core_chunks_v1")
    args = parser.parse_args(argv)
    if args.run_dir.expanduser().resolve() != INDEX_DIR.resolve() or args.backend != "local":
        parser.error("this experiment requires the frozen local index")
    if args.output_root.expanduser().resolve() == main_experiment.DEFAULT_OUTPUT_ROOT.resolve():
        parser.error("use a separate output root for the frozen-data experiment")
    return main_experiment.export(
        parser, args, runtime_class=FrozenDimensionRuntime, experiment=EXPERIMENT_DIR.name
    )


if __name__ == "__main__":
    raise SystemExit(main())
