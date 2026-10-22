import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

from rag_core import api


def test_file_processing_upload_is_authenticated_and_queued(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(api, "UPLOAD_ROOT", tmp_path / "uploads")
    monkeypatch.setattr(api, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setenv("RAG_API_TOKEN", "test-only-token")

    llm_api_key = "test-provider-secret"

    def complete_without_pipeline(job_id):
        with api._store_lock:
            api._job_llm_keys.pop(job_id, None)
        api._update_job(
            job_id,
            status="completed",
            stage="completed",
            progress=100,
            result={"run_id": job_id, "collection": "test_collection"},
        )

    monkeypatch.setattr(api, "_run_file_job", complete_without_pipeline)

    with TestClient(api.app) as client:
        unauthorized = client.post(
            "/api/v1/file-processing/jobs",
            files=[("files", ("corpus.md", b"valid text", "text/markdown"))],
            data={
                "llm_api_key": llm_api_key,
                "llm_base_url": "https://llm.example.test/v1",
                "llm_model": "test-model",
            },
        )
        assert unauthorized.status_code == 401

        response = client.post(
            "/api/v1/file-processing/jobs",
            headers={"Authorization": "Bearer test-only-token"},
            files=[("files", ("corpus.md", b"valid text", "text/markdown"))],
            data={
                "llm_api_key": llm_api_key,
                "llm_base_url": "https://llm.example.test/v1",
                "llm_model": "test-model",
            },
        )
        assert response.status_code == 202
        body = response.json()
        assert body["run_id"] == body["job_id"]

        deadline = time.time() + 2
        while time.time() < deadline:
            status = client.get(
                body["status_url"],
                headers={"Authorization": "Bearer test-only-token"},
            )
            if status.json()["status"] == "completed":
                break
            time.sleep(0.01)
        assert status.json()["status"] == "completed"
        saved_files = list((tmp_path / "uploads" / body["run_id"]).glob("*.md"))
        assert len(saved_files) == 1
        assert saved_files[0].read_text(encoding="utf-8") == "valid text"
        job_json = (tmp_path / "state" / "jobs" / f"{body['job_id']}.json").read_text()
        assert llm_api_key not in job_json
        assert llm_api_key not in status.text


def test_job_identifier_cannot_escape_state_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "STATE_DIR", tmp_path / "state")
    with TestClient(api.app) as client:
        response = client.get(
            "/api/v1/file-processing/jobs/../../etc/passwd",
            headers={"Authorization": "Bearer x"},
        )
        assert response.status_code in {404, 422}


def test_qa_returns_gateway_error_for_empty_model_answer(monkeypatch):
    monkeypatch.setenv("RAG_API_TOKEN", "test-only-token")

    class EmptyAnswerService:
        @staticmethod
        def answer(query, *, top_k, context_chars):
            return {"answer": "", "retrieval": {}}

    monkeypatch.setattr(
        api,
        "_get_service",
        lambda run_id, llm_api_key, llm_base_url, llm_model: (
            EmptyAnswerService(), {"collection": "test_collection"}
        ),
    )

    with TestClient(api.app) as client:
        response = client.post(
            "/api/v1/qa/ask",
            headers={"Authorization": "Bearer test-only-token"},
            json={
                "run_id": "012345abcdef",
                "query": "test",
                "llm_api_key": "test-secret",
                "llm_base_url": "https://llm.example.test/v1",
                "llm_model": "test-model",
            },
        )
    assert response.status_code == 502
    assert "上游模型返回空答案" in response.json()["detail"]


def test_llm_config_rejects_non_https_endpoint():
    try:
        api._validate_llm_config("test-secret", "http://llm.example.test/v1", "test-model")
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 422
    else:
        raise AssertionError("非 HTTPS 的大模型地址应被拒绝")


def test_service_uses_per_request_llm_credentials_and_separates_cache(monkeypatch, tmp_path):
    api._service_cache.clear()
    settings_seen = []

    def make_settings(run_dir, **kwargs):
        settings = SimpleNamespace(
            llm_api_key="server-must-not-be-used",
            llm_base_url="https://server-default.invalid/v1",
            llm_model="server-default",
            llm_openai_compat=False,
        )
        settings_seen.append(settings)
        return settings

    class FakeQAService:
        def __init__(self, *, run_dir, settings):
            self.settings = settings

    monkeypatch.setattr(
        api,
        "_ready_run",
        lambda run_id: (
            {"run_dir": str(tmp_path), "collection": "test_collection"},
            {"backend": "qdrant", "collection": "test_collection"},
        ),
    )
    monkeypatch.setattr(api.Settings, "from_env", make_settings)
    monkeypatch.setattr(api, "QAService", FakeQAService)

    first, _ = api._get_service(
        "012345abcdef", "caller-key-one", "https://provider-one.example/v1", "model-one"
    )
    second, _ = api._get_service(
        "012345abcdef", "caller-key-two", "https://provider-two.example/v1", "model-two"
    )

    assert first is not second
    assert first.settings.llm_api_key == "caller-key-one"
    assert first.settings.llm_base_url == "https://provider-one.example/v1"
    assert first.settings.llm_model == "model-one"
    assert first.settings.llm_openai_compat is True
    assert second.settings.llm_api_key == "caller-key-two"
    assert len(settings_seen) == 2
    api._service_cache.clear()


def test_file_processing_stage_artifacts_are_observable(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_API_TOKEN", "test-only-token")
    run_dir = tmp_path / "run"
    (run_dir / "chunks").mkdir(parents=True)
    (run_dir / "chunks.json").write_text(
        '[{"chunk_id":"c1","doc_title":"doc.md","chunk_len":25,"chunk_text_full":"颐和园上午开放。"}]',
        encoding="utf-8",
    )
    (run_dir / "chunks" / "chunk_summary.json").write_text(
        '{"total_chunks":1,"average_length":25}', encoding="utf-8"
    )
    (run_dir / "V_core_v2.json").write_text(
        '{"generation_method":"test-schema","schema_source_record_count":1,"dimensions":[{"id":"service.opening_hours","name":"开放时间","level":2,"indexable":true}]}',
        encoding="utf-8",
    )
    (run_dir / "tags_output_v2.json").write_text(
        '{"documents":{"c1":{"tags":{"service.opening_hours":["上午","开放"]},"dimension_paths":["服务属性 / 开放时间"],"tag_details":{"service.opening_hours":[{"label":"上午","evidence":"上午開放"}]}}}}',
        encoding="utf-8",
    )
    (run_dir / "dimension_metadata_v2.json").write_text(
        '{"service.opening_hours":{"path":"服务属性 / 开放时间"}}', encoding="utf-8"
    )
    (run_dir / "dimension_diagnostics_v2.json").write_text(
        '{"dimensions":{"service.opening_hours":{"coverage":1.0}}}', encoding="utf-8"
    )
    (run_dir / "store_manifest.json").write_text(
        '{"backend":"local","collection":"test_collection","vector_dim":8,"point_count":1}',
        encoding="utf-8",
    )
    (run_dir / "local_points.json").write_text(
        '[{"id":"point-1","payload":{"chunk_id":"c1","doc_title":"doc.md","chunk_len":25,"dimension_paths":["服务属性 / 开放时间"],"dim_service.opening_hours":["上午","开放"],"chunk_text_full":"颐和园上午开放。"}}]',
        encoding="utf-8",
    )
    job = {"job_id": "012345abcdef", "run_id": "012345abcdef", "run_dir": str(run_dir)}
    manifest = {
        "collection": "test_collection",
        "schema_generation_method": "test-schema",
        "schema_source_record_count": 1,
        "store": {"backend": "local", "collection": "test_collection", "point_count": 1},
    }
    monkeypatch.setattr(api, "_ready_run", lambda _: (job, manifest))

    headers = {"Authorization": "Bearer test-only-token"}
    with TestClient(api.app) as client:
        chunks = client.get(
            "/api/v1/file-processing/jobs/012345abcdef/chunks?include_text=true", headers=headers
        )
        dimensions = client.get(
            "/api/v1/file-processing/jobs/012345abcdef/dimensions?include_evidence=true", headers=headers
        )
        ingestion = client.get(
            "/api/v1/file-processing/jobs/012345abcdef/ingestion?include_text=true", headers=headers
        )

    assert chunks.status_code == dimensions.status_code == ingestion.status_code == 200
    assert chunks.json()["summary"]["total_chunks"] == 1
    assert chunks.json()["items"][0]["chunk_text"] == "颐和园上午开放。"
    assert dimensions.json()["dimensions"][0]["covered_chunks"] == 1
    assert dimensions.json()["tagged_chunks"]["items"][0]["tag_details"]
    assert ingestion.json()["verified"] is True
    assert ingestion.json()["sample_points"][0]["tags"]["dim_service.opening_hours"] == ["上午", "开放"]
