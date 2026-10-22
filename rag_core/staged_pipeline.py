"""可断点续跑的分阶段 RAG 数据流水线。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chunk_stage import run_chunking
from .dimension_stage import default_schema, induce_dataset_schema, run_dimensions
from .embedding import EmbeddingModel
from .settings import Settings
from .storage import VectorStore


STATE_FILE = "pipeline_state.json"
STAGE_ARTIFACTS = {
    "chunk": ["chunks.json", "chunks/chunk_summary.json"],
    "schema": ["V_cand_v2.json", "V_core_v2.json"],
    "tag": ["tags_output_v2.json", "inverted_index_v2.json", "dimension_diagnostics_v2.json"],
    "index": ["store_manifest.json", "run_manifest.json"],
    "retrieve": ["retrieval"],
    "evaluate": ["evaluation"],
    "qa": ["qa_summary.json"],
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _state_path(run_dir: Path) -> Path:
    return run_dir / STATE_FILE


def _load_state(run_dir: Path) -> dict[str, Any]:
    path = _state_path(run_dir)
    if not path.exists():
        return {"version": 1, "project": "rag_core_pipeline", "run_dir": str(run_dir), "stages": {}}
    value = _json(path)
    if not isinstance(value, dict) or not isinstance(value.get("stages", {}), dict):
        raise ValueError(f"流水线状态文件格式错误: {path}")
    return value


def _save_state(run_dir: Path, state: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = _now()
    _write_json(_state_path(run_dir), state)


def _artifact_paths(run_dir: Path, stage: str) -> list[Path]:
    return [run_dir / item for item in STAGE_ARTIFACTS[stage]]


def _stage_ready(run_dir: Path, stage: str) -> bool:
    state = _load_state(run_dir)
    recorded = state.get("stages", {}).get(stage, {}).get("artifacts") or []
    paths = [Path(path) for path in recorded] or _artifact_paths(run_dir, stage)
    return bool(paths) and all(path.exists() for path in paths)


def _begin(run_dir: Path, stage: str, config: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    run_dir.mkdir(parents=True, exist_ok=True)
    state = _load_state(run_dir)
    previous = state.setdefault("stages", {}).get(stage, {})
    if (previous.get("status") == "completed" and previous.get("config") == config
            and _stage_ready(run_dir, stage)):
        return state, True
    current = {
        "status": "running",
        "started_at": _now(),
        "config": config,
        "artifacts": [str(path) for path in _artifact_paths(run_dir, stage)],
    }
    state["stages"][stage] = current
    _save_state(run_dir, state)
    return state, False


def _finish(run_dir: Path, stage: str, state: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    entry = state["stages"][stage]
    paths = [Path(path) for path in entry.get("artifacts", [])] or _artifact_paths(run_dir, stage)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"{stage} 阶段声称完成但缺少产物: {missing}")
    entry.update({"status": "completed", "finished_at": _now(), "summary": summary})
    state["run_dir"] = str(run_dir)
    _save_state(run_dir, state)
    return {"stage": stage, **entry}


def _fail(run_dir: Path, stage: str, error: BaseException) -> None:
    state = _load_state(run_dir)
    entry = state.setdefault("stages", {}).setdefault(stage, {})
    entry.update({"status": "failed", "finished_at": _now(),
                  "error": f"{type(error).__name__}: {error}"})
    _save_state(run_dir, state)


def load_chunks(run_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(run_dir)
    path = root / "chunks.json"
    if not path.exists():
        legacy_path = root / "chunks" / "chunks.json"
        if legacy_path.exists():
            path = legacy_path
    if not path.exists():
        raise FileNotFoundError(f"未找到分片产物 chunks.json: {root}")
    records = _json(path)
    if not isinstance(records, list) or not records:
        raise ValueError(f"分片产物必须是非空 JSON 数组: {path}")
    invalid = [index for index, row in enumerate(records) if not isinstance(row, dict) or not row.get("doc_id")]
    if invalid:
        raise ValueError(f"分片记录缺少 doc_id，首个异常位置: {invalid[0]}")
    return records


def _dataset_fingerprint(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in records:
        digest.update(str(row.get("doc_id", "")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row.get("doc_text", row.get("chunk_text_full", ""))).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _source_signature(path: str | Path) -> str:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"输入路径不存在: {source}")
    files = ([source] if source.is_file() else sorted(
        item for item in source.rglob("*")
        if item.is_file() and item.suffix.lower() in {".txt", ".md", ".json"}
    ))
    digest = hashlib.sha256()
    for item in files:
        stat = item.stat()
        relative = item.name if source.is_file() else item.relative_to(source).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _stage_result(run_dir: Path, stage: str) -> dict[str, Any] | None:
    state = _load_state(run_dir)
    entry = state.get("stages", {}).get(stage)
    if entry and entry.get("status") == "completed" and _stage_ready(run_dir, stage):
        return {"stage": stage, "skipped": True, **entry}
    return None


def run_chunk_stage(settings: Settings) -> dict[str, Any]:
    run_dir = Path(settings.run_dir)
    if not settings.input_path:
        raise ValueError("chunk 阶段必须提供输入路径")
    state_before = _load_state(run_dir)
    previous = state_before.get("stages", {}).get("chunk", {})
    config = {"input": str(settings.input_path), "input_signature": _source_signature(settings.input_path),
              "denoise_method": settings.denoise_method,
              "l_min": settings.l_min, "l_max": settings.l_max}
    if (previous.get("status") == "completed" and previous.get("config") == config
            and _stage_ready(run_dir, "chunk")):
        return {"stage": "chunk", "skipped": True, **previous}
    if previous.get("status") == "completed" and _stage_ready(run_dir, "chunk"):
        raise ValueError("分片输入或参数已变化；为保留旧中间结果，请使用新的 --run-dir")
    # 允许从已有的 chunks.json 接续旧 run；不重新分片，也不覆盖旧产物。
    chunks_file = run_dir / "chunks.json"
    if chunks_file.exists():
        records = load_chunks(run_dir)
        state, _ = _begin(run_dir, "chunk", {**config, "adopted_existing": True})
        summary = {"total_chunks": len(records), "fingerprint": _dataset_fingerprint(records),
                   "adopted_existing": True}
        state["input"] = str(settings.input_path)
        state["dataset_fingerprint"] = summary["fingerprint"]
        return _finish(run_dir, "chunk", state, summary)
    state, _ = _begin(run_dir, "chunk", config)
    try:
        result = run_chunking(settings.input_path, run_dir / "chunks", settings)
        records = result["records"]
        chunks_file.parent.mkdir(parents=True, exist_ok=True)
        chunks_file.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        summary = {**result["summary"], "fingerprint": _dataset_fingerprint(records),
                   "input": str(settings.input_path)}
        _write_json(run_dir / "chunks" / "chunk_summary.json", summary)
        state["input"] = str(settings.input_path)
        state["dataset_fingerprint"] = summary["fingerprint"]
        return _finish(run_dir, "chunk", state, summary)
    except BaseException as exc:
        _fail(run_dir, "chunk", exc)
        raise


def run_schema_stage(settings: Settings) -> dict[str, Any]:
    run_dir = Path(settings.run_dir)
    records = load_chunks(run_dir)
    fingerprint = _dataset_fingerprint(records)
    state, skip = _begin(run_dir, "schema", {"mock": settings.mock, "model": settings.llm_model,
                                              "dataset_fingerprint": fingerprint})
    if skip:
        return {"stage": "schema", "skipped": True, **state["stages"]["schema"]}
    try:
        if settings.mock:
            schema = default_schema()
            schema.update({"generation_method": "code_jyx_mock_schema",
                           "schema_source_record_count": len(records),
                           "schema_source_fingerprint": fingerprint})
        else:
            settings.apply_runtime_environment()
            schema = induce_dataset_schema(records, settings)
        _write_json(run_dir / "V_cand_v2.json", schema)
        _write_json(run_dir / "V_core_v2.json", schema)
        summary = {"generation_method": schema.get("generation_method", ""),
                   "dimension_count": len(schema.get("dimensions", [])),
                   "source_records": len(records), "dataset_fingerprint": fingerprint}
        state["dataset_fingerprint"] = fingerprint
        return _finish(run_dir, "schema", state, summary)
    except BaseException as exc:
        _fail(run_dir, "schema", exc)
        raise


def run_tag_stage(settings: Settings, *, resume: bool = True) -> dict[str, Any]:
    run_dir = Path(settings.run_dir)
    records = load_chunks(run_dir)
    state, skip = _begin(run_dir, "tag", {"mock": settings.mock, "model": settings.llm_model,
                                          "resume_checkpoint": resume,
                                          "dataset_fingerprint": _dataset_fingerprint(records)})
    if skip:
        return {"stage": "tag", "skipped": True, **state["stages"]["tag"]}
    try:
        schema_path = run_dir / "V_core_v2.json"
        if not schema_path.exists():
            raise FileNotFoundError(f"缺少数据集专属维度 schema: {schema_path}；先运行 schema")
        schema = _json(schema_path)
        if schema.get("schema_source_fingerprint") != _dataset_fingerprint(records):
            raise ValueError("schema 与当前 chunks.json 不匹配；请从 schema 阶段重新生成")
        strict = schema.get("generation_method") == "llm_corpus_induced_strict"
        if strict and settings.mock:
            raise ValueError("真实数据集 schema 不能在 mock 模式下继续打标签")
        settings.apply_runtime_environment()
        result = run_dimensions(
            records, settings,
            schema_path=str(schema_path) if strict else "",
            strict_dataset_schema=strict,
            resume_tags=resume,
        )
        tag_docs = result["tags"].get("documents", {})
        summary = {"records": len(records), "tagged_documents": len(tag_docs),
                   "dimensions": len(result["schema"].get("dimensions", [])),
                   "generation_method": schema.get("generation_method", ""),
                   "resume_checkpoint": str(run_dir / "tags_checkpoint.jsonl")}
        return _finish(run_dir, "tag", state, summary)
    except BaseException as exc:
        _fail(run_dir, "tag", exc)
        raise


def run_index_stage(settings: Settings) -> dict[str, Any]:
    run_dir = Path(settings.run_dir)
    records = load_chunks(run_dir)
    fingerprint = _dataset_fingerprint(records)
    if not settings.mock and settings.backend == "qdrant" and not settings.collection:
        raise ValueError("Qdrant 入库必须显式指定专用 --collection，避免写入默认/现有集合")
    state, _ = _begin(run_dir, "index", {"backend": settings.backend,
                                          "collection": settings.collection,
                                          "qdrant_url": settings.qdrant_url,
                                          "embedding_model": settings.model_path or "configured_default",
                                          "dataset_fingerprint": fingerprint})
    if state["stages"]["index"].get("status") == "completed" and _stage_ready(run_dir, "index"):
        return {"stage": "index", "skipped": True, **state["stages"]["index"]}
    try:
        schema_path = run_dir / "V_core_v2.json"
        tags_path = run_dir / "tags_output_v2.json"
        if not schema_path.exists() or not tags_path.exists():
            raise FileNotFoundError("index 需要 V_core_v2.json 和 tags_output_v2.json；先运行 schema、tag")
        tags = _json(tags_path)
        expected_ids = {str(row["doc_id"]) for row in records}
        actual_ids = set((tags.get("documents") or {}).keys())
        missing = expected_ids - actual_ids
        if missing:
            raise ValueError(f"标签产物缺少 {len(missing)} 个 chunk，拒绝入库")
        schema = _json(schema_path)
        embeddings = EmbeddingModel(settings.model_path, settings.embedding_device,
                                    settings.vector_dim, settings.mock)
        store = VectorStore(run_dir=run_dir, backend="local" if settings.mock else settings.backend,
                            qdrant_url=settings.qdrant_url, collection=settings.collection,
                            vector_dim=settings.vector_dim)
        store_result = store.upsert(records, tags, schema, embeddings)
        chunk_summary_path = run_dir / "chunks" / "chunk_summary.json"
        chunk_summary = _json(chunk_summary_path) if chunk_summary_path.exists() else {"total_chunks": len(records)}
        manifest = {
            "project": "rag_core_pipeline", "schema_version": "2.0",
            "backend": store_result["backend"], "collection": store_result["collection"],
            "input": _load_state(run_dir).get("input", ""), "run_dir": str(run_dir),
            "schema_generation_method": schema.get("generation_method", ""),
            "schema_source_record_count": schema.get("schema_source_record_count", 0),
            "dimension_count": len(schema.get("dimensions", [])),
            "experiment_config": {"semantic_pool": settings.semantic_pool,
                                   "dimension_pool": settings.dimension_pool,
                                   "top_k": settings.top_k, "dim_alpha": settings.dim_alpha,
                                   "embedding_model": settings.model_path or "configured_default",
                                   "denoise_method": settings.denoise_method},
            "chunk_summary": chunk_summary, "store": store_result,
            "artifacts": {"chunks": str(run_dir / "chunks.json"),
                          "schema": str(schema_path), "tags": str(tags_path),
                          "inverted_index": str(run_dir / "inverted_index_v2.json"),
                          "entity_index": str(run_dir / "entity_index_v2.json")},
        }
        _write_json(run_dir / "run_manifest.json", manifest)
        state = _load_state(run_dir)
        state["input"] = chunk_summary.get("input", state.get("input", ""))
        _save_state(run_dir, state)
        summary = {"point_count": store_result["point_count"],
                   "backend": store_result["backend"], "collection": store_result["collection"],
                   "vector_dim": store_result["vector_dim"]}
        return _finish(run_dir, "index", state, summary)
    except BaseException as exc:
        _fail(run_dir, "index", exc)
        raise


def run_retrieval_stage(settings: Settings, query: str, *, top_k: int = 5,
                        query_expansion: bool = True, output: str | Path | None = None) -> dict[str, Any]:
    from .llm import LLMClient
    from .prompting import PromptOptimizer
    from .retrieval import Retriever

    run_dir = Path(settings.run_dir)
    if not query.strip():
        raise ValueError("retrieve 需要非空 --query")
    output_path = Path(output) if output else run_dir / "retrieval" / "latest.json"
    state, skip = _begin(run_dir, "retrieve", {"query": query, "top_k": top_k,
                                                "query_expansion": query_expansion,
                                                "output": str(output_path)})
    if skip and output_path.exists():
        return {"stage": "retrieve", "skipped": True, **state["stages"]["retrieve"]}
    try:
        settings.apply_runtime_environment()
        embeddings = EmbeddingModel(settings.model_path, settings.embedding_device,
                                    settings.vector_dim, settings.mock)
        query_llm = None
        if not settings.mock and settings.llm_api_key:
            from code_jyx.llm_service import DimensionMiningWithQwen
            query_llm = DimensionMiningWithQwen(api_key=settings.llm_api_key,
                                                model_name=settings.llm_model,
                                                base_url=settings.llm_base_url)
        retriever = Retriever(run_dir=run_dir, settings=settings, embeddings=embeddings, llm=query_llm)
        subqueries = [query]
        expansion = {"original_query": query, "sub_queries": [query], "cached": False}
        if query_expansion:
            client = LLMClient(api_key=settings.llm_api_key, base_url=settings.llm_base_url,
                               model=settings.llm_model, openai_compat=settings.llm_openai_compat,
                               interval=settings.llm_interval, mock=settings.mock)
            expansion = PromptOptimizer(client=client, run_dir=run_dir).expand(query)
            subqueries = expansion.get("sub_queries") or [query]
        retrieval = retriever.search(
            " | ".join(subqueries), top_k=top_k,
            semantic_pool=settings.semantic_pool, dimension_pool=settings.dimension_pool,
            original_query=query, subqueries=subqueries,
        )
        result = {"query": query, "retrieval_query": " | ".join(subqueries),
                  "query_expansion": expansion, "top_k": top_k, "retrieval": retrieval}
        _write_json(output_path, result)
        entry = state["stages"]["retrieve"]
        entry["artifacts"] = [str(output_path)]
        entry["summary"] = {"semantic": len(retrieval.get("semantic_results", [])),
                            "dimension": len(retrieval.get("dimension_results", [])),
                            "fusion": len(retrieval.get("fusion_results", [])),
                            "output": str(output_path)}
        entry.update({"status": "completed", "finished_at": _now()})
        state["updated_at"] = _now()
        _write_json(_state_path(run_dir), state)
        return {"stage": "retrieve", **entry}
    except BaseException as exc:
        _fail(run_dir, "retrieve", exc)
        raise


def run_evaluation_stage(settings: Settings, dataset: str | Path, *, output: str | Path | None = None,
                         top_k: int = 10, eval_depth: int = 20, ks: str = "1,3,5,10,20",
                         query_expansion: bool = True, query_parser_mode: str = "deterministic") -> dict[str, Any]:
    from .evaluation import run_evaluation

    run_dir = Path(settings.run_dir)
    output_dir = Path(output) if output else run_dir / "evaluation"
    state, skip = _begin(run_dir, "evaluate", {"dataset": str(dataset), "output": str(output_dir),
                                               "top_k": top_k, "eval_depth": eval_depth, "ks": ks,
                                               "query_expansion": query_expansion,
                                               "query_parser_mode": query_parser_mode})
    if skip and output_dir.exists():
        return {"stage": "evaluate", "skipped": True, **state["stages"]["evaluate"]}
    try:
        settings.apply_runtime_environment()
        output_dir.mkdir(parents=True, exist_ok=True)
        result = run_evaluation(run_dir=run_dir, dataset_path=dataset, settings=settings,
                                output_dir=output_dir, top_k=top_k, eval_depth=eval_depth,
                                ks=ks, query_expansion=query_expansion,
                                query_parser_mode=query_parser_mode,
                                experiments_dir=output_dir / "published")
        _write_json(output_dir / "summary.json", {
            "fusion_output": result.get("fusion_output", ""),
            "config": result.get("config", {}),
            "summary": result.get("summary", {}),
        })
        entry = state["stages"]["evaluate"]
        entry["artifacts"] = [str(output_dir)]
        entry["summary"] = {"output": result.get("fusion_output", ""),
                            "summary": result.get("summary", {})}
        entry.update({"status": "completed", "finished_at": _now()})
        _write_json(_state_path(run_dir), state)
        return {"stage": "evaluate", **entry}
    except BaseException as exc:
        _fail(run_dir, "evaluate", exc)
        raise


def run_qa_stage(settings: Settings, *, question: str = "", dataset: str = "",
                 output: str = "", top_k: int = 5, concurrency: int = 1,
                 context_chars: int = 0, resume: bool = True) -> dict[str, Any]:
    from .batch_qa import run_batch_qa
    from .qa import QAService

    run_dir = Path(settings.run_dir)
    if bool(question.strip()) == bool(dataset):
        raise ValueError("qa 必须且只能提供 --question 或 --dataset 之一")
    default_name = "qa_results.jsonl" if dataset else "qa_result.json"
    output_path = Path(output) if output else run_dir / default_name
    state, skip = _begin(run_dir, "qa", {"question": question, "dataset": dataset,
                                          "output": str(output_path), "top_k": top_k,
                                          "concurrency": concurrency, "resume": resume})
    if skip and output_path.exists():
        return {"stage": "qa", "skipped": True, **state["stages"]["qa"]}
    try:
        settings.apply_runtime_environment()
        if dataset:
            summary = run_batch_qa(run_dir=run_dir, dataset_path=dataset,
                                   output_path=output_path, settings=settings, top_k=top_k,
                                   context_chars=context_chars, concurrency=concurrency, resume=resume)
        else:
            answer = QAService(run_dir=run_dir, settings=settings).answer(
                question, top_k=top_k, context_chars=context_chars)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(answer, ensure_ascii=False, indent=2), encoding="utf-8")
            summary = {"question": question, "output": str(output_path), "completed": 1}
        entry = state["stages"]["qa"]
        entry["artifacts"] = [str(output_path)]
        entry["summary"] = summary
        entry.update({"status": "completed", "finished_at": _now()})
        _write_json(_state_path(run_dir), state)
        return {"stage": "qa", **entry}
    except BaseException as exc:
        _fail(run_dir, "qa", exc)
        raise


def pipeline_status(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    state = _load_state(root)
    output = {"run_dir": str(root), "state_file": str(_state_path(root)), "stages": {}}
    for stage, artifacts in STAGE_ARTIFACTS.items():
        entry = dict(state.get("stages", {}).get(stage, {}))
        registered_artifacts = entry.get("artifacts") or [str(root / item) for item in artifacts]
        present = [str(path) for path in registered_artifacts if Path(path).exists()]
        if not entry and present:
            status = "artifacts_found_not_registered"
        else:
            status = entry.get("status", "not_started")
        output["stages"][stage] = {"status": status,
                                   "artifacts_present": present,
                                   **({"summary": entry.get("summary"), "error": entry.get("error")}
                                      if entry else {})}
    return output


def inspect_run(run_dir: str | Path, stage: str = "all") -> dict[str, Any]:
    root = Path(run_dir)
    if stage not in {"all", *STAGE_ARTIFACTS}:
        raise ValueError(f"未知阶段 {stage!r}; 可选: all, {', '.join(STAGE_ARTIFACTS)}")
    selected = list(STAGE_ARTIFACTS) if stage == "all" else [stage]
    details: dict[str, Any] = {"run_dir": str(root), "stages": {}}
    registered = _load_state(root).get("stages", {})
    for name in selected:
        stage_state = registered.get(name, {})
        saved_artifacts = stage_state.get("artifacts") or [str(path) for path in _artifact_paths(root, name)]
        info: dict[str, Any] = {
            "status": stage_state.get("status", "not_started"),
            "artifacts": [str(path) for path in saved_artifacts if Path(path).exists()],
        }
        if stage_state.get("summary") is not None:
            info["stage_summary"] = stage_state["summary"]
        try:
            if name == "chunk" and (root / "chunks.json").exists():
                rows = load_chunks(root)
                lengths = [len(str(row.get("doc_text", row.get("chunk_text_full", "")))) for row in rows]
                info["summary"] = {"chunks": len(rows), "avg_chars": round(sum(lengths) / len(lengths), 2),
                                   "min_chars": min(lengths), "max_chars": max(lengths)}
            elif name == "schema" and (root / "V_core_v2.json").exists():
                schema = _json(root / "V_core_v2.json")
                dims = schema.get("dimensions", [])
                info["summary"] = {"generation_method": schema.get("generation_method", ""),
                                   "dimensions": len(dims),
                                   "parents": sum(item.get("level") == 1 for item in dims),
                                   "leaves": sum(item.get("level") == 2 and item.get("indexable", True)
                                                 for item in dims),
                                   "source_records": schema.get("schema_source_record_count", 0)}
            elif name == "tag" and (root / "tags_output_v2.json").exists():
                tags = _json(root / "tags_output_v2.json").get("documents", {})
                info["summary"] = {"documents": len(tags),
                                   "documents_with_tags": sum(bool(doc.get("tags")) for doc in tags.values()),
                                   "tag_count": sum(len(values if isinstance(values, list) else [values])
                                                    for doc in tags.values()
                                                    for values in (doc.get("tags") or {}).values())}
            elif name == "index" and (root / "store_manifest.json").exists():
                info["summary"] = _json(root / "store_manifest.json")
            elif name == "qa" and (root / "qa_summary.json").exists():
                info["summary"] = _json(root / "qa_summary.json")
            elif name == "evaluate" and (root / "evaluation" / "summary.json").exists():
                info["summary"] = _json(root / "evaluation" / "summary.json")
            elif name == "retrieve":
                files = sorted((root / "retrieval").glob("*.json")) if (root / "retrieval").exists() else []
                info["queries"] = [{"path": str(path), "query": _json(path).get("query", "")}
                                    for path in files]
        except Exception as exc:
            info["inspection_error"] = f"{type(exc).__name__}: {exc}"
        details["stages"][name] = info
    return details
