"""命令行入口。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .pipeline import optimize_prompt, run_ingest
from .prompt_module import run_prompt_module
from .qa import QAService
from .settings import Settings


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", default="./runs/default", help="本次运行的独立目录")
    parser.add_argument("--backend", choices=("qdrant", "local"), default=None)
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--no-openai-compat", action="store_true")
    parser.add_argument("--llm-interval", type=float, default=None)
    parser.add_argument("--mock", action="store_true", help="本地无网络验收模式")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG 核心流程：分片、维度、入库、检索、Prompt、答案")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "ingest"):
        item = sub.add_parser(command, help="执行分片、维度抽取并入库")
        _common(item)
        item.add_argument("--input", required=True)
        item.add_argument("--schema", default="", help="可选 v2 Schema JSON")
        item.add_argument("--denoise-method", choices=("none", "ppl", "mechanical"), default="mechanical")
        item.add_argument("--prompt-examples", default="", help="可选问答样本 JSON，用于 Prompt 迭代")
        item.add_argument("--prompt-iterations", type=int, default=3)
        if command == "run":
            item.add_argument("--query", default="", help="入库后立即回答一条问题")
            item.add_argument("--top-k", type=int, default=5)
    ask = sub.add_parser("ask", help="对已有运行目录检索并生成答案")
    _common(ask)
    ask.add_argument("--query", required=True)
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--context-chars", type=int, default=0, help="每个 chunk 上限；0 表示传入完整 chunk")
    opt = sub.add_parser("optimize-prompt", help="用问答样本执行案例级 Prompt 迭代")
    _common(opt)
    opt.add_argument("--examples", required=True)
    opt.add_argument("--prompt-iterations", type=int, default=3)
    module = sub.add_parser("prompt-module", help="独立 Prompt 模块：JSON 输入到 JSON 输出")
    _common(module)
    module.add_argument("--input", required=True, help="问答样本 JSON")
    module.add_argument("--output", required=True, help="优化 Prompt JSON 输出路径")
    module.add_argument("--base-prompt", default="", help="可选基础 Prompt JSON")
    module.add_argument("--prompt-iterations", type=int, default=None)
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
    return Settings.from_env(args.run_dir, **overrides)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    settings = _settings(args)
    try:
        if args.command == "prompt-module":
            result = run_prompt_module(
                args.input,
                args.output,
                settings,
                base_prompt_path=args.base_prompt,
                iterations=args.prompt_iterations,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command in {"run", "ingest"}:
            settings.input_path = Path(args.input)
            settings.denoise_method = args.denoise_method
            settings.top_k = getattr(args, "top_k", 5)
            manifest = run_ingest(settings, schema_path=args.schema)
            if args.prompt_examples:
                optimize_prompt(settings, args.prompt_examples, args.prompt_iterations)
            if args.command == "run" and args.query:
                result = QAService(run_dir=settings.run_dir, settings=settings).answer(args.query, top_k=args.top_k)
                print(json.dumps({"answer": result["answer"], "top_chunks": result["retrieval"]["top_chunks"]},
                                 ensure_ascii=False, indent=2))
            else:
                print(json.dumps(manifest, ensure_ascii=False, indent=2))
        elif args.command == "optimize-prompt":
            result = optimize_prompt(settings, args.examples, args.prompt_iterations)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            manifest_path = Path(args.run_dir) / "run_manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if args.backend is None:
                    settings.backend = manifest.get("backend", settings.backend)
                if args.collection is None:
                    settings.collection = manifest.get("collection", settings.collection)
            result = QAService(run_dir=settings.run_dir, settings=settings).answer(
                args.query, top_k=args.top_k, context_chars=args.context_chars)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
