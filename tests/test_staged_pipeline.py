import json

from rag_core.settings import Settings
from rag_core.staged_pipeline import (
    inspect_run,
    pipeline_status,
    run_chunk_stage,
    run_evaluation_stage,
    run_index_stage,
    run_qa_stage,
    run_retrieval_stage,
    run_schema_stage,
    run_tag_stage,
)


def test_mock_pipeline_runs_each_stage_and_persists_intermediates(tmp_path):
    source = tmp_path / "source.md"
    source.write_text(
        "# 北京颐和园测试资料\n\n" + "\n".join(
            f"第{i}段：北京颐和园面向游客开放参观，上午开放，门票免费；这是用于分阶段流水线验收的不同内容{i}。"
            for i in range(1, 25)
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    settings = Settings.from_env(run_dir, mock=True, backend="local")
    settings.input_path = source
    settings.denoise_method = "none"
    settings.semantic_pool = 5
    settings.dimension_pool = 5
    settings.top_k = 3

    chunk_result = run_chunk_stage(settings)
    assert chunk_result["status"] == "completed"
    assert (run_dir / "chunks.json").exists()
    assert json.loads((run_dir / "chunks.json").read_text(encoding="utf-8"))

    schema_result = run_schema_stage(settings)
    assert schema_result["status"] == "completed"
    schema = json.loads((run_dir / "V_core_v2.json").read_text(encoding="utf-8"))
    assert schema["generation_method"] == "code_jyx_mock_schema"

    tag_result = run_tag_stage(settings)
    assert tag_result["status"] == "completed"
    assert (run_dir / "tags_checkpoint.jsonl").exists()
    assert (run_dir / "inverted_index_v2.json").exists()

    index_result = run_index_stage(settings)
    assert index_result["status"] == "completed"
    assert index_result["summary"]["backend"] == "local"
    assert index_result["summary"]["point_count"] > 0

    retrieval_result = run_retrieval_stage(
        settings, "北京颐和园什么时候开放？", top_k=3, query_expansion=False
    )
    retrieval_path = run_dir / "retrieval" / "latest.json"
    assert retrieval_result["status"] == "completed"
    retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
    assert retrieval["query"]
    assert retrieval["retrieval"]["semantic_results"]

    chunks = json.loads((run_dir / "chunks.json").read_text(encoding="utf-8"))
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps([{
        "id": "q1",
        "question": "北京颐和园什么时候开放？",
        "reference_answer": "上午开放",
        "gold_chunk_ids": [chunks[0]["doc_id"]],
    }], ensure_ascii=False), encoding="utf-8")
    evaluation_result = run_evaluation_stage(
        settings, golden, top_k=3, eval_depth=3, query_expansion=False
    )
    assert evaluation_result["status"] == "completed"
    assert (run_dir / "evaluation" / "summary.json").exists()

    qa_result = run_qa_stage(settings, question="北京颐和园什么时候开放？", top_k=3)
    assert qa_result["status"] == "completed"
    assert (run_dir / "qa_result.json").exists()

    statuses = pipeline_status(run_dir)
    assert all(statuses["stages"][stage]["status"] == "completed"
               for stage in ("chunk", "schema", "tag", "index", "retrieve", "evaluate", "qa"))
    inspected = inspect_run(run_dir, "all")
    assert inspected["stages"]["chunk"]["summary"]["chunks"] > 0


def test_completed_stage_is_idempotent_and_changed_input_is_not_overwritten(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("北京颐和园提供参观服务。\n" * 30, encoding="utf-8")
    run_dir = tmp_path / "run"
    settings = Settings.from_env(run_dir, mock=True, backend="local")
    settings.input_path = source
    settings.denoise_method = "none"
    first = run_chunk_stage(settings)
    second = run_chunk_stage(settings)
    assert first["status"] == "completed"
    assert second["skipped"] is True

    source.write_text(source.read_text(encoding="utf-8") + "输入有变化。", encoding="utf-8")
    try:
        run_chunk_stage(settings)
    except ValueError as exc:
        assert "新的 --run-dir" in str(exc)
    else:  # pragma: no cover - protects existing intermediate data
        raise AssertionError("changed source should not silently overwrite completed chunks")
