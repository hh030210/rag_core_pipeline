#!/usr/bin/env python3
"""Prepare the public DuRetrieval/MTEB dev split for this RAG pipeline.

The MTEB export stores DuReader passages as already-atomic retrieval units.
We still run the project's three-stage splitter in memory, while preserving
the original corpus id in ``parent_doc_id`` so qrels can be mapped after the
chunking version changes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_core.integrated_chunker import IntegratedChunker


def _read_table(path: Path):
    return pq.read_table(path).to_pylist()


def _chunk_records(corpus_rows, *, l_min: int, l_max: int, denoise_method: str):
    # Passage rows are atomic units in the retrieval benchmark. Disabling the
    # per-file LLM recommendation here is intentional: otherwise 100K short
    # passages would cause 100K recommendation calls. The actual sentence
    # denoise/boundary/final merge logic remains the project's implementation.
    args = SimpleNamespace(
        llm_api_key="",
        llm_base_url="http://127.0.0.1:8911/v1",
        llm_model="DeepSeek-V4-Pro",
        llm_timeout=180,
        ppl_model_name="",
        window_w=3,
        beta_small=0.8,
        beta=1.1,
        recommendation_config="",
        denoise=denoise_method != "none",
        denoise_method=denoise_method,
    )
    chunker = IntegratedChunker(args)
    records = []
    pid_to_chunks = {}
    for index, row in enumerate(corpus_rows, 1):
        pid = str(row.get("_id") or "").strip()
        text = str(row.get("text") or "").strip()
        title = str(row.get("title") or "").strip() or pid
        if not pid or not text:
            continue
        subfiles = chunker._round1_split(text, None, pid)
        produced = []
        for sub in subfiles or []:
            # Force the original retrieval passage id after round one. This
            # also handles a passage that contains Markdown-like headings.
            sub["doc_id"] = pid
            sub["file_name"] = pid
            sub["genre"] = "web_passage"
            sub["l_min"] = int(l_min)
            sub["l_max"] = int(l_max)
            r2_chunks, denoised = chunker._round2_process(sub)
            sub["denoised_content"] = denoised
            sub["r2_chunks"] = r2_chunks
            final_chunks = chunker._round3_process(sub)
            produced.extend(str(item.get("chunk_text") or "").strip() for item in final_chunks)
        produced = [value for value in produced if value]
        if not produced:
            produced = [text]
        ids = []
        for chunk_index, chunk_text in enumerate(produced):
            chunk_id = f"{pid}::chunk_{chunk_index:04d}"
            ids.append(chunk_id)
            records.append({
                "doc_id": chunk_id,
                "parent_doc_id": pid,
                "chunk_id": chunk_id,
                "doc_title": title,
                "chunk_gen_title": title,
                "source_file": pid,
                "doc_text": chunk_text,
                "chunk_text": chunk_text,
                "chunk_text_full": chunk_text,
                "chunk_len": len(chunk_text),
                "l_min": int(l_min),
                "l_max": int(l_max),
                "denoise_method": denoise_method,
                "dataset_corpus_id": pid,
            })
        pid_to_chunks[pid] = ids
        if index % 5000 == 0:
            print(f"[分片] {index}/{len(corpus_rows)} passages, {len(records)} chunks", flush=True)
    return records, pid_to_chunks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--l-min", type=int, default=400)
    parser.add_argument("--l-max", type=int, default=1000)
    parser.add_argument("--denoise-method", choices=("none", "mechanical", "ppl"), default="mechanical")
    parser.add_argument("--max-corpus", type=int, default=0, help="smoke test limit; 0 means all")
    parser.add_argument("--max-queries", type=int, default=0, help="smoke test limit; 0 means all")
    args = parser.parse_args()

    corpus_rows = _read_table(args.input_root / "corpus.parquet")
    query_rows = _read_table(args.input_root / "queries.parquet")
    qrels_rows = _read_table(args.input_root / "data.parquet")
    if args.max_corpus > 0:
        corpus_rows = corpus_rows[:args.max_corpus]
    if args.max_queries > 0:
        query_rows = query_rows[:args.max_queries]
    records, pid_to_chunks = _chunk_records(
        corpus_rows,
        l_min=args.l_min,
        l_max=args.l_max,
        denoise_method=args.denoise_method,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "chunks.json").write_text(
        json.dumps(records, ensure_ascii=False), encoding="utf-8"
    )

    text_by_pid = {str(row.get("_id")): str(row.get("text") or "") for row in corpus_rows}
    qrels_by_query = {}
    for row in qrels_rows:
        if int(row.get("score") or 0) <= 0:
            continue
        qid = str(row.get("query-id") or "")
        pid = str(row.get("corpus-id") or "")
        if qid and pid:
            qrels_by_query.setdefault(qid, []).append(pid)
    qa_rows = []
    for row in query_rows:
        qid = str(row.get("_id") or "")
        if not qid:
            continue
        pids = list(dict.fromkeys(qrels_by_query.get(qid, [])))
        gold_chunk_ids = [chunk_id for pid in pids for chunk_id in pid_to_chunks.get(pid, [])]
        evidence = [
            {"source_doc_id": pid, "evidence_text": text_by_pid.get(pid, "")}
            for pid in pids if text_by_pid.get(pid, "")
        ]
        qa_rows.append({
            "id": qid,
            "question": str(row.get("text") or "").strip(),
            "gold_chunk_ids": gold_chunk_ids,
            "gold_evidence": evidence,
            "reference_answer": "",
            "answerable": bool(evidence),
            "dataset": "DuRetrieval/MTEB dev",
        })
    (args.output_root / "qa_dataset.json").write_text(
        json.dumps(qa_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "dataset": "DuRetrieval/MTEB dev",
        "input_root": str(args.input_root),
        "corpus_passages": len(corpus_rows),
        "queries": len(query_rows),
        "positive_qrels": sum(len(value) for value in qrels_by_query.values()),
        "unique_positive_passages": len({pid for values in qrels_by_query.values() for pid in values}),
        "chunks": len(records),
        "chunking": {
            "implementation": "rag_core.integrated_chunker.IntegratedChunker",
            "stages": ["round1_structure_default_recommendation", "round2_denoise_boundary", "round3_length_merge"],
            "l_min": args.l_min,
            "l_max": args.l_max,
            "denoise_method": args.denoise_method,
            "llm_length_recommendation": False,
            "reason": "DuRetrieval rows are already atomic passages; per-row recommendation would create 100K LLM calls.",
        },
        "outputs": {
            "chunks": str(args.output_root / "chunks.json"),
            "qa_dataset": str(args.output_root / "qa_dataset.json"),
        },
    }
    (args.output_root / "prepare_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
