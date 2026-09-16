"""调用保留的三阶段分片实现，并把其输出统一为 chunk 记录。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

from .integrated_chunker import IntegratedChunker


def run_chunking(input_path: str | Path, output_dir: str | Path, settings) -> Dict[str, Any]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(
        llm_api_key=settings.llm_api_key,
        llm_base_url=settings.llm_base_url,
        llm_model=settings.llm_model,
        llm_timeout=180,
        ppl_model_name="",
        window_w=3,
        beta_small=0.8,
        beta=1.1,
        recommendation_config=str(output_dir / "chunk_recommendations.json"),
        denoise=settings.denoise_method != "none",
        denoise_method=settings.denoise_method,
    )
    IntegratedChunker(args).run(input_path, output_dir)
    aggregate = output_dir / "all_chunks_chunks.json"
    if not aggregate.exists():
        candidates = sorted(output_dir.glob("*_chunks.json"))
        if len(candidates) == 1:
            aggregate = candidates[0]
    if not aggregate.exists():
        raise RuntimeError(f"分片阶段没有生成结果: {output_dir}")

    raw = json.loads(aggregate.read_text(encoding="utf-8"))
    records: List[Dict[str, Any]] = []
    for sub_index, sub in enumerate(raw if isinstance(raw, list) else []):
        parent_id = str(sub.get("doc_id") or sub.get("source_file") or f"doc_{sub_index}")
        source_file = str(sub.get("source_file") or sub.get("file_name") or parent_id)
        title = str(sub.get("file_name") or source_file)
        for chunk_index, chunk in enumerate(sub.get("chunks") or []):
            text = str(chunk.get("chunk_text") or "").strip()
            if not text:
                continue
            chunk_id = f"{parent_id}::chunk_{chunk_index:04d}"
            records.append({
                "doc_id": chunk_id,
                "parent_doc_id": parent_id,
                "chunk_id": chunk_id,
                "doc_title": title,
                "chunk_gen_title": title,
                "source_file": source_file,
                "doc_text": text,
                "chunk_text": text,
                "chunk_text_full": text,
                "chunk_len": len(text),
                "l_min": sub.get("l_min", settings.l_min),
                "l_max": sub.get("l_max", settings.l_max),
                "denoise_method": sub.get("denoise_method", settings.denoise_method),
            })
    if not records:
        raise RuntimeError("分片结果为空")
    path = output_dir / "chunks.json"
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    lengths = [item["chunk_len"] for item in records]
    summary = {
        "total_chunks": len(records),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "average_length": round(sum(lengths) / len(lengths), 3),
        "shorter_than_l_min": sum(length < settings.l_min for length in lengths),
        "longer_than_l_max": sum(length > settings.l_max for length in lengths),
        "denoise_method": settings.denoise_method,
        "source_artifact": str(aggregate),
    }
    (output_dir / "chunk_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"records": records, "summary": summary, "path": str(path)}

