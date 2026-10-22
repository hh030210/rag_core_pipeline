"""Authenticated HTTP API for file processing, retrieval, and question answering."""

from __future__ import annotations

import hmac
import hashlib
import json
import os
import re
import threading
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Optional
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .pipeline import run_ingest
from .qa import QAService
from .settings import Settings


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _configured_path(env_name: str, default: str) -> Path:
    return Path(os.getenv(env_name, default)).expanduser().resolve()


STATE_DIR = _configured_path("RAG_API_STATE_DIR", "./api_state")
UPLOAD_ROOT = _configured_path("RAG_API_UPLOAD_DIR", "./api_data")
RUNS_ROOT = _configured_path("RAG_API_RUNS_DIR", "./api_runs")
MAX_FILE_BYTES = int(os.getenv("RAG_API_MAX_FILE_BYTES", str(50 * 1024 * 1024)))
MAX_TOTAL_BYTES = int(os.getenv("RAG_API_MAX_TOTAL_BYTES", str(200 * 1024 * 1024)))
MAX_UPLOAD_FILES = int(os.getenv("RAG_API_MAX_FILES", "100"))
ALLOWED_SUFFIXES = {".txt", ".md", ".json"}
JOB_ID_RE = re.compile(r"^[0-9a-f]{12}$")
COLLECTION_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

_store_lock = threading.RLock()
_service_lock = threading.RLock()
_service_cache: OrderedDict[tuple[str, str, str, str], QAService] = OrderedDict()
_job_llm_keys: dict[str, str] = {}
_executor = None
_bearer = HTTPBearer(auto_error=False)


class RetrievalRequest(BaseModel):
    run_id: str = Field(min_length=12, max_length=12)
    query: str = Field(min_length=1, max_length=4000)
    llm_api_key: str = Field(min_length=1, max_length=4096, repr=False)
    llm_base_url: str = Field(min_length=1, max_length=2048)
    llm_model: str = Field(min_length=1, max_length=256)
    top_k: int = Field(default=5, ge=1, le=100)
    semantic_pool: Optional[int] = Field(default=None, ge=1, le=500)
    dimension_pool: Optional[int] = Field(default=None, ge=1, le=500)
    include_text: bool = False


class QARequest(BaseModel):
    run_id: str = Field(min_length=12, max_length=12)
    query: str = Field(min_length=1, max_length=4000)
    llm_api_key: str = Field(min_length=1, max_length=4096, repr=False)
    llm_base_url: str = Field(min_length=1, max_length=2048)
    llm_model: str = Field(min_length=1, max_length=256)
    top_k: int = Field(default=5, ge=1, le=30)
    context_chars: int = Field(default=0, ge=0, le=100_000)
    include_context_text: bool = False


def _job_path(job_id: str) -> Path:
    if not JOB_ID_RE.fullmatch(job_id):
        raise HTTPException(status_code=404, detail="任务不存在")
    return STATE_DIR / "jobs" / f"{job_id}.json"


def _read_job(job_id: str) -> dict[str, Any]:
    path = _job_path(job_id)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="任务不存在") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="任务状态不可读取") from exc


def _write_job(job: dict[str, Any]) -> None:
    path = _job_path(str(job["job_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    job["updated_at"] = _utc_now()
    temp_path = path.with_suffix(".json.tmp")
    with _store_lock:
        temp_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(path)


def _update_job(job_id: str, **updates: Any) -> dict[str, Any]:
    with _store_lock:
        job = _read_job(job_id)
        job.update(updates)
        _write_job(job)
        return job


def _auth(credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)) -> None:
    expected = os.getenv("RAG_API_TOKEN", "")
    if not expected:
        raise HTTPException(status_code=503, detail="API 尚未配置访问令牌")
    actual = credentials.credentials if credentials else ""
    if not actual or not hmac.compare_digest(actual, expected):
        raise HTTPException(
            status_code=401,
            detail="无效的访问令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _validate_llm_config(api_key: str, base_url: str, model: str) -> tuple[str, str, str]:
    key = str(api_key or "").strip()
    endpoint = str(base_url or "").strip().rstrip("/")
    model_name = str(model or "").strip()
    parsed = urlsplit(endpoint)
    if not key:
        raise HTTPException(status_code=422, detail="必须提供 llm_api_key")
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise HTTPException(status_code=422, detail="llm_base_url 必须是有效的 HTTPS 地址")
    if not model_name:
        raise HTTPException(status_code=422, detail="必须提供 llm_model")
    return key, endpoint, model_name


def _configure_llm(settings: Settings, api_key: str, base_url: str, model: str) -> None:
    settings.llm_api_key = api_key
    settings.llm_base_url = base_url
    settings.llm_model = model
    settings.llm_openai_compat = True


def _progress(job_id: str, stage: str, percent: int, message: str) -> None:
    _update_job(job_id, status="running", stage=stage, progress=percent, message=message)


def _run_file_job(job_id: str) -> None:
    job = _read_job(job_id)
    with _store_lock:
        llm_api_key = _job_llm_keys.pop(job_id, "")
    _update_job(job_id, status="running", stage="starting", progress=1, message="任务已启动")
    run_dir = Path(job["run_dir"])
    input_dir = Path(job["input_dir"])
    try:
        settings = Settings.from_env(
            run_dir,
            input_path=input_dir,
            collection=job["collection"],
            mock=False,
        )
        if not llm_api_key:
            raise RuntimeError("此任务的 LLM API Key 未在内存中；请重新提交任务")
        _configure_llm(settings, llm_api_key, job["llm_base_url"], job["llm_model"])
        settings.denoise_method = job["denoise_method"]
        if not settings.model_path or not Path(settings.model_path).exists():
            raise RuntimeError("服务器未配置有效的 BGE_MODEL_PATH")
        manifest = run_ingest(
            settings,
            strict_dataset_schema=True,
            progress_callback=lambda stage, percent, message: _progress(
                job_id, stage, percent, message
            ),
        )
        _update_job(
            job_id,
            status="completed",
            stage="completed",
            progress=100,
            message="文件处理完成",
            result={
                "run_id": job["run_id"],
                "collection": manifest.get("collection", job["collection"]),
                "backend": manifest.get("backend", "qdrant"),
                "chunk_summary": manifest.get("chunk_summary", {}),
                "point_count": manifest.get("store", {}).get("point_count", 0),
                "schema_generation_method": manifest.get("schema_generation_method", ""),
                "schema_source_record_count": manifest.get("schema_source_record_count", 0),
                "dimension_count": manifest.get("dimension_count", 0),
                "run_manifest": str(run_dir / "run_manifest.json"),
                "observability": {
                    "chunks_url": f"/api/v1/file-processing/jobs/{job_id}/chunks",
                    "dimensions_url": f"/api/v1/file-processing/jobs/{job_id}/dimensions",
                    "ingestion_url": f"/api/v1/file-processing/jobs/{job_id}/ingestion",
                },
            },
        )
    except Exception as exc:  # Persist failure for clients polling the async job.
        error = str(exc)
        if llm_api_key:
            error = error.replace(llm_api_key, "[REDACTED]")
        _update_job(
            job_id,
            status="failed",
            stage="failed",
            message="文件处理失败",
            error={"type": type(exc).__name__, "detail": error[:2000]},
        )
    finally:
        with _store_lock:
            _job_llm_keys.pop(job_id, None)


def _submit_job(job_id: str) -> None:
    if _executor is None:
        raise RuntimeError("文件处理队列尚未启动")
    _executor.submit(_run_file_job, job_id)


def _recover_jobs() -> None:
    jobs_dir = STATE_DIR / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    for path in jobs_dir.glob("*.json"):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
            job_id = str(job.get("job_id", ""))
            if job.get("status") == "queued" and JOB_ID_RE.fullmatch(job_id):
                job.update(
                    status="failed",
                    stage="interrupted",
                    message="服务重启后不会持久化恢复 LLM API Key；请重新提交文件处理任务",
                    error={"type": "InterruptedError", "detail": "为保护密钥，任务凭据仅保存在进程内存"},
                )
                _write_job(job)
            elif job.get("status") == "running":
                job.update(
                    status="failed",
                    stage="interrupted",
                    message="服务重启导致任务中断；请重新提交文件处理任务",
                    error={"type": "InterruptedError", "detail": "任务未自动重跑，避免重复消耗模型调用"},
                )
                _write_job(job)
        except (OSError, json.JSONDecodeError, RuntimeError):
            continue


def _ready_run(run_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    job = _read_job(run_id)
    if job.get("status") != "completed":
        raise HTTPException(status_code=409, detail="该 run_id 的文件处理尚未完成")
    run_dir = Path(job.get("run_dir", "")).resolve()
    if not run_dir.is_relative_to(RUNS_ROOT):
        raise HTTPException(status_code=400, detail="运行目录不在 API 管理范围内")
    manifest_path = run_dir / "run_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="运行清单缺失或不可读取") from exc
    return job, manifest


def _read_run_artifact(run_dir: Path, name: str, default: Any = None) -> Any:
    """Read only a known, run-local JSON artifact (never a caller-supplied path)."""
    paths = {
        "chunks": run_dir / "chunks.json",
        "chunks_summary": run_dir / "chunks" / "chunk_summary.json",
        "source_chunks": run_dir / "chunks" / "source_chunks.json",
        "source_summary": run_dir / "chunks" / "source_summary.json",
        "chunk_recommendations": run_dir / "chunks" / "chunk_recommendations.json",
        "schema": run_dir / "V_core_v2.json",
        "tags": run_dir / "tags_output_v2.json",
        "dimension_metadata": run_dir / "dimension_metadata_v2.json",
        "diagnostics": run_dir / "dimension_diagnostics_v2.json",
        "store": run_dir / "store_manifest.json",
        "local_points": run_dir / "local_points.json",
    }
    path = paths[name]
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"运行产物 {name} 不可读取") from exc


def _page_params(offset: int, limit: int) -> tuple[int, int]:
    return max(0, offset), min(max(1, limit), 100)


def _ingestion_preview(point: dict[str, Any], include_text: bool) -> dict[str, Any]:
    payload = point.get("payload") or {}
    result = {
        "point_id": str(point.get("id", "")),
        "chunk_id": payload.get("chunk_id", ""),
        "doc_id": payload.get("doc_id", ""),
        "doc_title": payload.get("doc_title", ""),
        "chunk_len": payload.get("chunk_len", 0),
        "dimension_paths": payload.get("dimension_paths", []),
        "tags": {
            key: value for key, value in payload.items()
            if key.startswith("dim_") and isinstance(value, list)
        },
    }
    if include_text:
        result["chunk_text"] = payload.get("chunk_text_full") or payload.get("chunk_text") or ""
    return result


def _get_service(
    run_id: str, llm_api_key: str, llm_base_url: str, llm_model: str
) -> tuple[QAService, dict[str, Any]]:
    job, manifest = _ready_run(run_id)
    api_key, base_url, model = _validate_llm_config(llm_api_key, llm_base_url, llm_model)
    key_fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    cache_key = (run_id, key_fingerprint, base_url, model)
    with _service_lock:
        service = _service_cache.get(cache_key)
        if service is None:
            run_dir = Path(job["run_dir"]).resolve()
            settings = Settings.from_env(
                run_dir,
                backend=manifest.get("backend", "qdrant"),
                collection=manifest.get("collection", job["collection"]),
                mock=False,
            )
            _configure_llm(settings, api_key, base_url, model)
            service = QAService(run_dir=run_dir, settings=settings)
            _service_cache[cache_key] = service
            max_services = max(1, int(os.getenv("RAG_API_SERVICE_CACHE_SIZE", "2")))
            while len(_service_cache) > max_services:
                _service_cache.popitem(last=False)
        else:
            _service_cache.move_to_end(cache_key)
        return service, manifest


def _public_hit(item: dict[str, Any], rank: int, include_text: bool) -> dict[str, Any]:
    fields = (
        "chunk_id", "doc_id", "parent_doc_id", "doc_title", "chunk_gen_title",
        "source_file", "rank", "score", "normalized_semantic_score",
        "normalized_dimension_score", "fused_score", "final_score", "source",
        "matched_dimensions", "matches", "fact_anchor_matches",
        "precise_entity_matches", "slot_coverage_matches",
    )
    result = {key: item[key] for key in fields if key in item}
    result["rank"] = rank
    result["chunk_len"] = item.get("chunk_len")
    if include_text:
        result["chunk_text"] = item.get("chunk_text_full") or item.get("chunk_text") or ""
    return result


@asynccontextmanager
async def _lifespan(_: FastAPI):
    global _executor
    from concurrent.futures import ThreadPoolExecutor

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    _executor = ThreadPoolExecutor(
        max_workers=max(1, int(os.getenv("RAG_API_FILE_WORKERS", "1"))),
        thread_name_prefix="rag-file-job",
    )
    _recover_jobs()
    try:
        yield
    finally:
        _executor.shutdown(wait=False, cancel_futures=False)
        _executor = None


app = FastAPI(
    title="RAG Core Pipeline API",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    lifespan=_lifespan,
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "service": "rag-core-pipeline-api"}


@app.post("/api/v1/file-processing/jobs", status_code=202, dependencies=[Depends(_auth)])
async def create_file_processing_job(
    files: list[UploadFile] = File(...),
    collection: str = Form(default=""),
    denoise_method: str = Form(default="mechanical"),
    llm_api_key: str = Form(...),
    llm_base_url: str = Form(...),
    llm_model: str = Form(...),
) -> dict[str, Any]:
    llm_api_key, llm_base_url, llm_model = _validate_llm_config(
        llm_api_key, llm_base_url, llm_model
    )
    if not files:
        raise HTTPException(status_code=422, detail="至少上传一个 .txt、.md 或 .json 文件")
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(status_code=413, detail=f"单次最多上传 {MAX_UPLOAD_FILES} 个文件")
    if denoise_method not in {"none", "mechanical", "ppl"}:
        raise HTTPException(status_code=422, detail="denoise_method 仅支持 none、mechanical、ppl")

    job_id = uuid.uuid4().hex[:12]
    run_id = job_id
    if collection and not COLLECTION_RE.fullmatch(collection):
        raise HTTPException(status_code=422, detail="collection 只能包含字母、数字、下划线和连字符，长度 1–64")
    collection = collection or f"rag_api_{run_id}"
    input_dir = UPLOAD_ROOT / run_id
    run_dir = RUNS_ROOT / run_id
    input_dir.mkdir(parents=True, exist_ok=False)

    total_bytes = 0
    stored_files = []
    try:
        used_names: set[str] = set()
        for index, upload in enumerate(files, 1):
            original_name = str(upload.filename or "").replace("\\", "/")
            safe_name = PurePosixPath(original_name).name
            suffix = Path(safe_name).suffix.lower()
            if not safe_name or suffix not in ALLOWED_SUFFIXES:
                raise HTTPException(status_code=415, detail="仅支持 .txt、.md、.json 文件")
            base = Path(safe_name).stem or f"file_{index}"
            candidate = f"{base}{suffix}"
            serial = 2
            while candidate in used_names:
                candidate = f"{base}_{serial}{suffix}"
                serial += 1
            used_names.add(candidate)

            destination = input_dir / candidate
            file_bytes = 0
            with destination.open("wb") as output:
                while True:
                    block = await upload.read(1024 * 1024)
                    if not block:
                        break
                    file_bytes += len(block)
                    total_bytes += len(block)
                    if file_bytes > MAX_FILE_BYTES:
                        raise HTTPException(status_code=413, detail=f"{safe_name} 超过单文件大小限制")
                    if total_bytes > MAX_TOTAL_BYTES:
                        raise HTTPException(status_code=413, detail="上传总大小超过限制")
                    output.write(block)
            if not file_bytes:
                destination.unlink(missing_ok=True)
                continue
            stored_files.append({"name": candidate, "bytes": file_bytes})
        if not stored_files:
            raise HTTPException(status_code=422, detail="上传文件为空")
    except Exception:
        for child in input_dir.iterdir():
            child.unlink(missing_ok=True)
        input_dir.rmdir()
        raise
    finally:
        for upload in files:
            await upload.close()

    job = {
        "job_id": job_id,
        "run_id": run_id,
        "status": "queued",
        "stage": "queued",
        "progress": 0,
        "message": "任务已排队；将依次完成分片、语料维度归纳与标签抽取、向量入库",
        "collection": collection,
        "backend": "qdrant",
        "denoise_method": denoise_method,
        "llm_base_url": llm_base_url,
        "llm_model": llm_model,
        "strict_dataset_schema": True,
        "files": stored_files,
        "input_dir": str(input_dir),
        "run_dir": str(run_dir),
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
    }
    _write_job(job)
    try:
        with _store_lock:
            _job_llm_keys[job_id] = llm_api_key
        _submit_job(job_id)
    except Exception as exc:
        with _store_lock:
            _job_llm_keys.pop(job_id, None)
        _update_job(job_id, status="failed", stage="failed", message="任务无法入队", error={
            "type": type(exc).__name__, "detail": str(exc)[:1000],
        })
        raise HTTPException(status_code=503, detail="文件处理队列不可用") from exc
    return {
        "job_id": job_id,
        "run_id": run_id,
        "status": "queued",
        "status_url": f"/api/v1/file-processing/jobs/{job_id}",
        "summary_url": f"/api/v1/file-processing/jobs/{job_id}/summary",
        "collection": collection,
    }


@app.get("/api/v1/file-processing/jobs/{job_id}", dependencies=[Depends(_auth)])
def get_file_processing_job(job_id: str) -> dict[str, Any]:
    job = _read_job(job_id)
    return {key: value for key, value in job.items() if key not in {"input_dir", "run_dir"}}


@app.get("/api/v1/file-processing/jobs/{job_id}/summary", dependencies=[Depends(_auth)])
def get_file_processing_summary(job_id: str) -> dict[str, Any]:
    job = _read_job(job_id)
    if job.get("status") != "completed":
        raise HTTPException(status_code=409, detail="文件处理尚未完成")
    return {
        "job_id": job_id,
        "run_id": job["run_id"],
        "collection": job["collection"],
        "status": job["status"],
        "result": job.get("result", {}),
        "observability": {
            "chunks_url": f"/api/v1/file-processing/jobs/{job_id}/chunks",
            "dimensions_url": f"/api/v1/file-processing/jobs/{job_id}/dimensions",
            "ingestion_url": f"/api/v1/file-processing/jobs/{job_id}/ingestion",
        },
    }


@app.get("/api/v1/file-processing/jobs/{job_id}/chunks", dependencies=[Depends(_auth)])
def get_file_processing_chunks(
    job_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    include_text: bool = False,
    include_source_text: bool = False,
    include_intermediate: bool = True,
) -> dict[str, Any]:
    job, _ = _ready_run(job_id)
    run_dir = Path(job["run_dir"]).resolve()
    chunks = _read_run_artifact(run_dir, "chunks", default=[])
    if not isinstance(chunks, list):
        raise HTTPException(status_code=500, detail="分片产物格式无效")
    offset, limit = _page_params(offset, limit)
    items = []
    for chunk in chunks[offset:offset + limit]:
        item = {
            key: chunk.get(key) for key in (
                "chunk_id", "doc_id", "parent_doc_id", "doc_title", "chunk_gen_title",
                "source_file", "chunk_len", "l_min", "l_max", "denoise_method",
            ) if key in chunk
        }
        if include_text:
            item["chunk_text"] = chunk.get("chunk_text_full") or chunk.get("chunk_text") or ""
        items.append(item)
    intermediate = None
    if include_intermediate:
        source_chunks = _read_run_artifact(run_dir, "source_chunks", default=[]) or []
        source_summary = _read_run_artifact(run_dir, "source_summary", default={}) or {}
        recommendations_artifact = _read_run_artifact(run_dir, "chunk_recommendations", default={}) or {}
        recommendation_map = recommendations_artifact.get("recommendations", {})
        recommendations = list(recommendation_map.values()) if isinstance(recommendation_map, dict) else []
        source_items = []
        for source in source_chunks[offset:offset + limit]:
            source_item = {
                key: source.get(key) for key in ("doc_id", "file_name", "source_file", "file_type")
                if key in source
            }
            content = str(source.get("content", ""))
            source_item["content_chars"] = len(content)
            if include_source_text:
                source_item["content"] = content
            source_items.append(source_item)
        intermediate = {
            "source_summary": source_summary,
            "recommendation_metadata": recommendations_artifact.get("metadata", {}),
            "recommendations": recommendations,
            "source_documents": {
                "total": len(source_chunks),
                "offset": offset,
                "limit": limit,
                "items": source_items,
            },
        }
    return {
        "run_id": job["run_id"],
        "stage": "chunking",
        "summary": _read_run_artifact(run_dir, "chunks_summary", default={}),
        "total": len(chunks),
        "offset": offset,
        "limit": limit,
        "items": items,
        "intermediate": intermediate,
    }


@app.get("/api/v1/file-processing/jobs/{job_id}/dimensions", dependencies=[Depends(_auth)])
def get_file_processing_dimensions(
    job_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    include_evidence: bool = False,
) -> dict[str, Any]:
    job, manifest = _ready_run(job_id)
    run_dir = Path(job["run_dir"]).resolve()
    schema = _read_run_artifact(run_dir, "schema", default={}) or {}
    tags_artifact = _read_run_artifact(run_dir, "tags", default={}) or {}
    documents = tags_artifact.get("documents", {})
    if not isinstance(documents, dict):
        raise HTTPException(status_code=500, detail="维度标签产物格式无效")
    metadata = _read_run_artifact(run_dir, "dimension_metadata", default={}) or {}
    diagnostics = _read_run_artifact(run_dir, "diagnostics", default={}) or {}
    dimension_summaries = []
    for dimension in schema.get("dimensions", []):
        dimension_id = str(dimension.get("id", ""))
        labels_by_chunk = [
            (record.get("tags") or {}).get(dimension_id, [])
            for record in documents.values()
        ]
        labels = sorted({str(label) for values in labels_by_chunk if isinstance(values, list) for label in values})
        covered = sum(bool(values) for values in labels_by_chunk)
        dimension_summaries.append({
            **dimension,
            "covered_chunks": covered,
            "unique_label_count": len(labels),
            "labels": labels,
            "metadata": metadata.get(dimension_id, {}),
            "diagnostics": (diagnostics.get("dimensions") or {}).get(dimension_id, {}),
        })

    chunk_ids = list(documents)
    offset, limit = _page_params(offset, limit)
    tag_items = []
    for chunk_id in chunk_ids[offset:offset + limit]:
        record = documents[chunk_id] or {}
        item = {
            "chunk_id": chunk_id,
            "tags": record.get("tags", {}),
            "dimension_paths": record.get("dimension_paths", []),
        }
        if include_evidence:
            item["tag_details"] = record.get("tag_details", {})
        tag_items.append(item)
    return {
        "run_id": job["run_id"],
        "stage": "dimension_extraction",
        "schema_generation_method": manifest.get("schema_generation_method", schema.get("generation_method", "")),
        "schema_source_record_count": manifest.get("schema_source_record_count", schema.get("schema_source_record_count", 0)),
        "dimension_count": len(dimension_summaries),
        "dimensions": dimension_summaries,
        "tagged_chunks": {
            "total": len(chunk_ids),
            "offset": offset,
            "limit": limit,
            "items": tag_items,
        },
    }


@app.get("/api/v1/file-processing/jobs/{job_id}/ingestion", dependencies=[Depends(_auth)])
def get_file_processing_ingestion(
    job_id: str,
    sample_limit: int = Query(default=5, ge=1, le=100),
    include_text: bool = False,
) -> dict[str, Any]:
    job, manifest = _ready_run(job_id)
    run_dir = Path(job["run_dir"]).resolve()
    store = _read_run_artifact(run_dir, "store", default=None) or manifest.get("store", {})
    backend = str(store.get("backend", manifest.get("backend", "")))
    collection = str(store.get("collection", manifest.get("collection", "")))
    observed_count = None
    points: list[dict[str, Any]] = []
    verification_error = ""
    try:
        if backend == "local":
            points = _read_run_artifact(run_dir, "local_points", default=[]) or []
            observed_count = len(points)
        elif backend == "qdrant":
            from qdrant_client import QdrantClient

            client = QdrantClient(
                url=str(store.get("qdrant_url", "http://127.0.0.1:6333")),
                timeout=15,
                prefer_grpc=False,
                check_compatibility=False,
            )
            count_result = client.count(collection_name=collection, exact=True)
            observed_count = int(count_result.count)
            page, _ = client.scroll(
                collection_name=collection,
                limit=sample_limit,
                with_payload=True,
                with_vectors=False,
            )
            points = [
                {"id": point.id, "payload": point.payload or {}}
                for point in page
            ]
        else:
            verification_error = f"不支持的存储后端：{backend or 'unknown'}"
    except Exception as exc:
        verification_error = f"{type(exc).__name__}: {str(exc)[:300]}"

    expected = int(store.get("point_count", manifest.get("store", {}).get("point_count", 0)) or 0)
    return {
        "run_id": job["run_id"],
        "stage": "ingestion",
        "backend": backend,
        "collection": collection,
        "vector_dim": store.get("vector_dim"),
        "expected_run_points": expected,
        "observed_collection_points": observed_count,
        "verified": observed_count is not None and observed_count >= expected,
        "verification_scope": "collection_total; 若 collection 被多个任务共用，实际总数可能大于本 run 的入库数",
        "verification_error": verification_error,
        "sample_points": [_ingestion_preview(point, include_text) for point in points[:sample_limit]],
    }


@app.post("/api/v1/retrieval/search", dependencies=[Depends(_auth)])
def retrieval_search(request: RetrievalRequest) -> dict[str, Any]:
    service, manifest = _get_service(
        request.run_id, request.llm_api_key, request.llm_base_url, request.llm_model
    )
    results = service.retriever.search(
        request.query,
        top_k=request.top_k,
        semantic_pool=request.semantic_pool,
        dimension_pool=request.dimension_pool,
        original_query=request.query,
    )
    lanes = {}
    for lane, result_key in (
        ("semantic", "semantic_results"),
        ("dimension", "dimension_results"),
        ("fusion", "fusion_results"),
    ):
        items = results.get(result_key, [])[: request.top_k]
        lanes[lane] = [
            _public_hit(item, rank, request.include_text)
            for rank, item in enumerate(items, 1)
        ]
    analysis = results.get("query_analysis", {})
    return {
        "run_id": request.run_id,
        "collection": manifest.get("collection"),
        "query": request.query,
        "top_k": request.top_k,
        "fusion_strategy": results.get("fusion_strategy", {}),
        "dimension_policy": results.get("dimension_policy", {}),
        "constraints": results.get("constraints", {}),
        "query_parser_diagnostics": analysis.get("query_parser_diagnostics", {}),
        "candidate_counts": {
            "semantic": len(results.get("semantic_candidates", [])),
            "dimension": len(results.get("dimension_candidates", [])),
            "fusion": len(results.get("fusion_candidates", [])),
        },
        "results": lanes,
    }


@app.post("/api/v1/qa/ask", dependencies=[Depends(_auth)])
async def ask_question(request: QARequest) -> dict[str, Any]:
    service, manifest = _get_service(
        request.run_id, request.llm_api_key, request.llm_base_url, request.llm_model
    )
    result = await run_in_threadpool(
        service.answer,
        request.query,
        top_k=request.top_k,
        context_chars=request.context_chars,
    )
    if not str(result.get("answer") or "").strip():
        raise HTTPException(
            status_code=502,
            detail="上游模型返回空答案；请检查 llm_api_key、llm_base_url、llm_model 及服务商响应",
        )
    retrieval = result.get("retrieval", {})
    lanes = {}
    for lane, result_key in (
        ("semantic", "semantic_results"),
        ("dimension", "dimension_results"),
        ("fusion", "fusion_results"),
    ):
        items = retrieval.get(result_key, [])[: request.top_k]
        lanes[lane] = [
            _public_hit(item, rank, request.include_context_text)
            for rank, item in enumerate(items, 1)
        ]
    return {
        "run_id": request.run_id,
        "collection": manifest.get("collection"),
        "query": result.get("query"),
        "retrieval_query": result.get("retrieval_query"),
        "answer": result.get("answer"),
        "prompt_mode": result.get("prompt_mode"),
        "retrieval_top_k": request.top_k,
        "results": lanes,
        "context_stats": result.get("context_stats", {}),
    }
