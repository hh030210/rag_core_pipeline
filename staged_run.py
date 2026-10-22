#!/usr/bin/env python3
"""分阶段运行 RAG 流水线；原有 run.py 命令保持不变。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

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


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", default="./runs/default", help="本次运行的独立目录/中间产物目录")
    parser.add_argument("--backend", choices=("qdrant", "local"), default=None)
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--collection", default=None, help="Qdrant 专用 collection；入库时建议显式设置")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-interval", type=float, default=None)
    parser.add_argument("--no-openai-compat", action="store_true")
    parser.add_argument("--mock", action="store_true", help="离线验收模式：mock schema/tag、哈希向量、本地存储")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAG 可分阶段执行入口：中间产物保存在 run-dir，可从已完成阶段继续"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    chunk = sub.add_parser("chunk", help="分片并保存标准化 chunks.json")
    _common(chunk)
    chunk.add_argument("--input", required=True, help="txt/md/json 文件或包含这些文件的目录")
    chunk.add_argument("--denoise-method", choices=("none", "ppl", "mechanical"), default="mechanical")
    chunk.add_argument("--chunk-min", type=int, default=None)
    chunk.add_argument("--chunk-max", type=int, default=None)

    for name, help_text in (
        ("schema", "基于当前 run-dir 的 chunks 重新归纳本数据集维度 schema"),
        ("tag", "按当前数据集 schema 对 chunks 提取维度标签；断点续跑"),
        ("index", "将 chunks、维度标签和向量写入本地后端或 Qdrant"),
    ):
        command = sub.add_parser(name, help=help_text)
        _common(command)
        if name == "tag":
            command.add_argument("--no-resume-tags", action="store_true",
                                 help="不使用已有 tags_checkpoint.jsonl；默认续跑")

    retrieve = sub.add_parser("retrieve", help="对已入库数据执行单条三路检索并保存完整 JSON")
    _common(retrieve)
    retrieve.add_argument("--query", required=True)
    retrieve.add_argument("--top-k", type=int, default=5)
    retrieve.add_argument("--no-query-expansion", action="store_true")
    retrieve.add_argument("--output", default="", help="默认保存到 run-dir/retrieval/latest.json")

    evaluate = sub.add_parser("evaluate", help="使用 Golden 数据集进行检索评测")
    _common(evaluate)
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--output", default="", help="默认保存到 run-dir/evaluation")
    evaluate.add_argument("--top-k", type=int, default=10)
    evaluate.add_argument("--eval-depth", type=int, default=20)
    evaluate.add_argument("--ks", default="1,3,5,10,20")
    evaluate.add_argument("--no-query-expansion", action="store_true")
    evaluate.add_argument("--query-parser-mode", choices=("deterministic", "llm"), default="deterministic")

    qa = sub.add_parser("qa", help="单条或批量问答；批量 JSONL 支持续跑")
    _common(qa)
    source = qa.add_mutually_exclusive_group(required=True)
    source.add_argument("--question", default="", help="单条问题")
    source.add_argument("--dataset", default="", help="问答 JSON/JSONL")
    qa.add_argument("--output", default="", help="默认单条 qa_result.json 或批量 qa_results.jsonl")
    qa.add_argument("--top-k", type=int, default=5)
    qa.add_argument("--context-chars", type=int, default=0, help="0 表示使用完整 chunk")
    qa.add_argument("--concurrency", type=int, default=1)
    qa.add_argument("--no-resume", action="store_true")

    status = sub.add_parser("status", help="查看每个阶段状态和已发现的中间产物")
    status.add_argument("--run-dir", default="./runs/default")

    inspect = sub.add_parser("inspect", help="查看指定阶段产物摘要")
    inspect.add_argument("--run-dir", default="./runs/default")
    inspect.add_argument("--stage", choices=("all", "chunk", "schema", "tag", "index",
                                               "retrieve", "evaluate", "qa"), default="all")
    return parser


def _settings(args) -> Settings:
    overrides = {key: value for key, value in {
        "backend": args.backend,
        "qdrant_url": args.qdrant_url,
        "collection": args.collection,
        "model_path": args.model_path,
        "llm_api_key": args.llm_api_key,
        "llm_base_url": args.llm_base_url,
        "llm_model": args.llm_model,
        "llm_interval": args.llm_interval,
        "mock": True if args.mock else None,
    }.items() if value is not None}
    if args.no_openai_compat:
        overrides["llm_openai_compat"] = False
    if args.mock:
        overrides["backend"] = "local"
    settings = Settings.from_env(args.run_dir, **overrides)
    if (args.command == "index" and not args.mock and settings.backend == "qdrant"
            and not args.collection and not os.getenv("QDRANT_COLLECTION_NAME")):
        settings.collection = ""
    if args.command == "chunk":
        settings.input_path = Path(args.input)
        settings.denoise_method = args.denoise_method
        if args.chunk_min is not None:
            settings.l_min = args.chunk_min
        if args.chunk_max is not None:
            settings.l_max = args.chunk_max
    return settings


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            result = pipeline_status(args.run_dir)
        elif args.command == "inspect":
            result = inspect_run(args.run_dir, args.stage)
        else:
            settings = _settings(args)
            if args.command == "chunk":
                result = run_chunk_stage(settings)
            elif args.command == "schema":
                result = run_schema_stage(settings)
            elif args.command == "tag":
                result = run_tag_stage(settings, resume=not args.no_resume_tags)
            elif args.command == "index":
                result = run_index_stage(settings)
            elif args.command == "retrieve":
                result = run_retrieval_stage(settings, args.query, top_k=args.top_k,
                                             query_expansion=not args.no_query_expansion,
                                             output=args.output or None)
            elif args.command == "evaluate":
                result = run_evaluation_stage(settings, args.dataset, output=args.output or None,
                                              top_k=args.top_k, eval_depth=args.eval_depth,
                                              ks=args.ks,
                                              query_expansion=not args.no_query_expansion,
                                              query_parser_mode=args.query_parser_mode)
            elif args.command == "qa":
                default_output = ("qa_results.jsonl" if args.dataset else "qa_result.json")
                result = run_qa_stage(settings, question=args.question or "", dataset=args.dataset or "",
                                      output=args.output or str(Path(args.run_dir) / default_output),
                                      top_k=args.top_k, concurrency=args.concurrency,
                                      context_chars=args.context_chars, resume=not args.no_resume)
            else:  # pragma: no cover - argparse constrains commands
                parser.error(f"不支持的命令: {args.command}")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
