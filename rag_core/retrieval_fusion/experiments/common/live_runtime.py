"""Shared runtime for end-to-end retrieval experiments.

This module loads raw QA records, maps their evidence against the current
collection, performs fresh query analysis, and searches the live vector store.
It intentionally never reads experiment output folders as retrieval inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from uuid import uuid4

from rag_core.embedding import EmbeddingModel
from rag_core.evaluation import load_gold_records, map_gold_to_chunks
from rag_core.llm import LLMClient
from rag_core.prompting import PromptOptimizer
from rag_core.retrieval import Retriever
from rag_core.settings import Settings
from rag_core.vector_retrieval import VectorRetriever
from rag_core.retrieval_fusion.fusion import FUSION_STRATEGY

REPOSITORY = Path(__file__).resolve().parents[4]
DEFAULT_RUN_DIR = Path(
    "/home/humq/rag_core_runs/real_merged7_sdu_v4pro_full_20260921_run1"
)
DEFAULT_DATASET = DEFAULT_RUN_DIR / "qa_dataset_435.json"


def add_live_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_top_k: int = 20,
    default_semantic_pool: int = 100,
    default_dimension_pool: int = 20,
) -> None:
    """Add common runtime options to an experiment CLI."""
    parser.add_argument("--dataset", type=Path, default=None, help="Raw QA JSON/JSONL dataset")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--backend", choices=("qdrant", "local"), default=None)
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--query-parser-mode", choices=("deterministic", "llm"), default="deterministic",
        help="Parse each raw query in this run; no prior query-analysis result is loaded",
    )
    parser.add_argument(
        "--query-expansion", action="store_true",
        help="Generate subqueries and entity terms with the configured LLM for this run",
    )
    parser.add_argument("--semantic-pool", type=int, default=default_semantic_pool)
    parser.add_argument("--dimension-pool", type=int, default=default_dimension_pool)
    parser.add_argument("--top-k", type=int, default=default_top_k)
    parser.add_argument("--ks", default="1,5,10,15,20", help="Metric cutoffs")
    parser.add_argument(
        "--vector-strategy", default=None,
        help="Optional vector strategy filename; defaults to vector_retrieval.search.py configuration",
    )


def parse_ks(value: str) -> tuple[int, ...]:
    ks = tuple(sorted({int(part.strip()) for part in str(value).split(",") if part.strip()}))
    if not ks or ks[0] <= 0:
        raise ValueError("ks must contain positive integers")
    return ks


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


class LiveRetrievalRuntime:
    """Initialize current settings, parser, embeddings, retriever, and raw Gold."""

    def __init__(self, args):
        self.args = args
        self.run_dir = Path(args.run_dir).expanduser().resolve()
        if not self.run_dir.is_dir():
            raise FileNotFoundError(f"run-dir does not exist: {self.run_dir}")

        store_manifest_path = self.run_dir / "store_manifest.json"
        store_manifest = (
            json.loads(store_manifest_path.read_text(encoding="utf-8"))
            if store_manifest_path.exists() else {}
        )
        overrides = {}
        for key, value in (
            ("backend", args.backend),
            ("qdrant_url", args.qdrant_url),
            ("collection", args.collection),
            ("model_path", args.model_path),
            ("embedding_device", args.device),
        ):
            if value is not None:
                overrides[key] = value
        self.settings = Settings.from_env(self.run_dir, **overrides)
        # Use the current index manifest when the environment/CLI did not
        # explicitly choose a store. Never infer the collection from old
        # experiment output.
        if args.backend is None and not os.getenv("RAG_BACKEND"):
            self.settings.backend = store_manifest.get("backend", self.settings.backend)
        if args.qdrant_url is None and not os.getenv("QDRANT_URL"):
            self.settings.qdrant_url = store_manifest.get("qdrant_url", self.settings.qdrant_url)
        if args.collection is None and not os.getenv("QDRANT_COLLECTION_NAME"):
            self.settings.collection = store_manifest.get("collection", self.settings.collection)
        if args.model_path is None:
            self.settings.model_path = os.getenv("BGE_MODEL_PATH", self.settings.model_path)
        self.settings.vector_dim = int(store_manifest.get("vector_dim", self.settings.vector_dim))
        self.settings.semantic_pool = int(args.semantic_pool)
        self.settings.dimension_pool = int(args.dimension_pool)
        self.settings.top_k = int(args.top_k)
        if self.settings.mock:
            raise ValueError("End-to-end experiments require real embeddings and retrieval; mock mode is disabled")
        if self.settings.backend == "qdrant" and not self.settings.collection:
            raise ValueError("No Qdrant collection configured; pass --collection or configure store_manifest.json")
        if self.settings.semantic_pool < self.settings.top_k:
            raise ValueError("--semantic-pool must be at least --top-k")
        if self.settings.dimension_pool < self.settings.top_k:
            raise ValueError("--dimension-pool must be at least --top-k")

        default_dataset = self.run_dir / "qa_dataset_435.json"
        if not default_dataset.is_file():
            default_dataset = DEFAULT_DATASET
        self.dataset_path = Path(args.dataset or os.getenv("RAG_EVAL_DATASET") or default_dataset)
        self.dataset_path = self.dataset_path.expanduser().resolve()
        if not self.dataset_path.is_file():
            raise FileNotFoundError(f"raw QA dataset does not exist: {self.dataset_path}")

        self.run_id = uuid4().hex
        self._temp = tempfile.TemporaryDirectory(prefix=f"rag-live-{self.run_id[:8]}-")
        self.cache_dir = Path(self._temp.name)

        self.settings.apply_runtime_environment()
        self.embeddings = EmbeddingModel(
            self.settings.model_path,
            self.settings.embedding_device,
            self.settings.vector_dim,
            self.settings.mock,
        )
        parser_llm = None
        if args.query_parser_mode == "llm":
            if not self.settings.llm_api_key:
                self.close()
                raise ValueError("--query-parser-mode llm requires LLM_API_KEY or DASHSCOPE_API_KEY")
            from code_jyx.llm_service import DimensionMiningWithQwen

            parser_llm = DimensionMiningWithQwen(
                api_key=self.settings.llm_api_key,
                model_name=self.settings.llm_model,
                base_url=self.settings.llm_base_url,
            )
        self.retriever = Retriever(
            run_dir=self.run_dir,
            settings=self.settings,
            embeddings=self.embeddings,
            llm=parser_llm,
        )
        self._reset_query_parser_cache()
        if args.vector_strategy:
            self.set_vector_strategy(args.vector_strategy)

        # Map Gold against payloads from the active collection, not a prior
        # experiment's snapshot or a stale copy of chunks.json.
        current_chunks = []
        for point in self.retriever.points:
            payload = dict(point.get("payload") or {})
            if payload.get("chunk_id") or payload.get("doc_id"):
                current_chunks.append(payload)
        if not current_chunks:
            self.close()
            raise RuntimeError(
                f"The current collection {self.settings.collection!r} returned no chunk payloads"
            )
        self.chunks_by_id = {
            str(chunk.get("chunk_id") or chunk.get("doc_id")): chunk
            for chunk in current_chunks
            if chunk.get("chunk_id") or chunk.get("doc_id")
        }
        current_chunk_ids = set(self.chunks_by_id)
        raw_records = load_gold_records(self.dataset_path)
        self.records = []
        for record in raw_records:
            mapped = map_gold_to_chunks(record, current_chunks)
            original_ids = list(map(str, mapped.get("gold_chunk_ids") or []))
            current_ids = [chunk_id for chunk_id in original_ids if chunk_id in current_chunk_ids]
            missing_ids = [chunk_id for chunk_id in original_ids if chunk_id not in current_chunk_ids]
            mapped["gold_chunk_ids"] = list(dict.fromkeys(current_ids))
            resolution = dict(mapped.get("gold_resolution") or {})
            resolution["unmatched_input_chunk_ids"] = missing_ids
            resolution["mapped_chunk_count"] = len(mapped["gold_chunk_ids"])
            resolution["gold_mapped"] = bool(mapped["gold_chunk_ids"])
            mapped["gold_resolution"] = resolution
            self.records.append(mapped)
        self.dataset_sha256 = file_sha256(self.dataset_path)
        self.point_count = len(current_chunks)
        self.query_expander = None
        if args.query_expansion:
            if not self.settings.llm_api_key:
                self.close()
                raise ValueError("--query-expansion requires LLM_API_KEY or DASHSCOPE_API_KEY")
            client = LLMClient(
                api_key=self.settings.llm_api_key,
                base_url=self.settings.llm_base_url,
                model=self.settings.llm_model,
                openai_compat=self.settings.llm_openai_compat,
                interval=self.settings.llm_interval,
            )
            # A run-local cache allows duplicate questions within this run to
            # share work, without reading previous run's query expansions.
            self.query_expander = PromptOptimizer(
                client=client, run_dir=self.cache_dir / "query_expansion"
            )

    def _reset_query_parser_cache(self) -> None:
        parser = getattr(self.retriever, "query_parser", None)
        if parser is None:
            return
        # Retriever's normal runtime cache is useful for production, but an
        # experiment must parse from current inputs. Discard anything loaded
        # during construction and write only to this run's temporary cache.
        parser.cache = {}
        parser.cache_file = str(self.cache_dir / "query_parser_cache.json")
        if hasattr(parser, "_cached_error_retry_attempted"):
            parser._cached_error_retry_attempted = set()

    def set_vector_strategy(self, strategy_file: str) -> None:
        self.retriever.vector_retriever = VectorRetriever(
            embeddings=self.embeddings,
            vector_store=self.retriever.store,
            vector_name=getattr(self.settings, "vector_name", "chunk_text_vec"),
            strategy_file=strategy_file,
        )

    def query_parts(self, question: str) -> list[str]:
        if self.query_expander is None:
            return [question]
        expanded = self.query_expander.expand(question)
        values = expanded.get("sub_queries") or [question]
        result = []
        for value in values:
            text = str(value or "").strip()
            if text and text not in result:
                result.append(text)
        return result or [question]

    def parse_query(self, query: str) -> dict:
        return self.retriever._parse_query(query)

    def retrieve(
        self,
        question: str,
        subqueries: list[str] | None = None,
        *,
        top_k: int | None = None,
        semantic_pool: int | None = None,
        dimension_pool: int | None = None,
    ) -> dict:
        parts = subqueries or [question]
        retrieval_query = " | ".join(parts)
        return self.retriever.search(
            retrieval_query,
            top_k=top_k or self.settings.top_k,
            semantic_pool=semantic_pool or self.settings.semantic_pool,
            dimension_pool=dimension_pool or self.settings.dimension_pool,
            original_query=question,
            subqueries=parts,
            save_snapshot=False,
        )

    def metadata(self) -> dict:
        return {
            "retrieval_mode": "live_end_to_end",
            "run_id": self.run_id,
            "created_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
            "source_commit": current_git_commit(),
            "dataset_path": str(self.dataset_path),
            "dataset_sha256": self.dataset_sha256,
            "dataset_count": len(self.records),
            "gold_mapped_count": sum(bool(row.get("gold_chunk_ids")) for row in self.records),
            "gold_unmapped_count": sum(not bool(row.get("gold_chunk_ids")) for row in self.records),
            "run_dir": str(self.run_dir),
            "backend": self.settings.backend,
            "qdrant_url": self.settings.qdrant_url if self.settings.backend == "qdrant" else "",
            "collection": self.settings.collection,
            "collection_points": self.point_count,
            "embedding_model": self.settings.model_path,
            "embedding_device": self.settings.embedding_device,
            "query_parser_mode": self.args.query_parser_mode,
            "query_parser_cache": "fresh temporary cache per run",
            "query_expansion": bool(self.query_expander),
            "semantic_pool": self.settings.semantic_pool,
            "dimension_pool": self.settings.dimension_pool,
            "top_k": self.settings.top_k,
            "vector_strategy": self.retriever.vector_retriever.strategy_file,
            "fusion_strategy": FUSION_STRATEGY,
        }

    def close(self) -> None:
        if getattr(self, "_temp", None) is not None:
            self._temp.cleanup()
            self._temp = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False
