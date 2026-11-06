"""Fall back to archived scenic query analysis when live LLM parsing fails."""
from __future__ import annotations

import gzip
import json
from collections import Counter
from copy import deepcopy
from functools import cached_property

from rag_core.retrieval import Retriever
from rag_core.retrieval_fusion.experiments.common.live_runtime import (
    LiveRetrievalRuntime,
    REPOSITORY,
)

ARCHIVE = REPOSITORY / "result/scenic_latest_gitmain_20261104/retrieval_results.jsonl.gz"


class ScenicRetriever(Retriever):
    replay_enabled = False
    replay_reason = None
    parser_source = "deterministic"

    @cached_property
    def archived_analyses(self) -> dict:
        analyses = {}
        with gzip.open(ARCHIVE, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                key = (str(row["id"]), row["question"], row["retrieval_query"])
                analysis = row["query_analysis"]
                diagnostics = analysis.get("parser_diagnostics", {})
                if diagnostics.get("parser_mode") != "llm" or diagnostics.get("llm_error"):
                    continue
                if key in analyses:
                    raise ValueError(f"Duplicate question identity in query replay archive: {key[0]}")
                analyses[key] = analysis
        return analyses

    def _parse_query(self, query: str) -> dict:
        if not self.replay_enabled:
            return super()._parse_query(query)

        if self.replay_reason is None:
            if self.llm is None:
                self.replay_reason = "missing_api_key"
            else:
                analysis = super()._parse_query(query)
                diagnostics = analysis.get("parser_diagnostics", {})
                if diagnostics.get("parser_mode") == "llm" and not diagnostics.get("llm_error"):
                    self.parser_source = "live_llm"
                    return analysis
                self.replay_reason = "llm_unavailable"

        record_id, question = self.replay_question
        archived = self.archived_analyses.get((record_id, question, query))
        if archived is None:
            raise ValueError(f"No matching LLM query replay for question ID {record_id} in {ARCHIVE}")
        self.parser_source = "archive_replay"
        return deepcopy(archived)


class ScenicRetrievalRuntime(LiveRetrievalRuntime):
    retriever_class = ScenicRetriever
    allow_query_replay = True

    def __init__(self, args):
        super().__init__(args)
        self.retriever.replay_enabled = args.query_parser_mode == "llm"
        if args.replay_query_analysis:
            self.retriever.replay_reason = "replay_requested"
        self.parser_sources = Counter()

    def retrieve_record(self, record, parts):
        question = record["question"]
        self.retriever.replay_question = (str(record["id"]), question)
        result = self.retrieve(question, parts)
        self.parser_sources[self.retriever.parser_source] += 1
        return result

    def metadata(self) -> dict:
        metadata = super().metadata()
        metadata["query_parser_sources"] = dict(self.parser_sources)
        metadata["query_parser_replay"] = {
            "used": self.parser_sources["archive_replay"] > 0,
            "data_path": str(ARCHIVE),
            "reason": self.retriever.replay_reason,
        }
        return metadata
