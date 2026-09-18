"""命令行入口。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .evaluation import compare_evaluations, run_evaluation
from .batch_qa import run_batch_qa
from .pipeline import optimize_prompt, run_cluster_prompting, run_ingest
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
    parser.add_argument("--cluster-prompts", action="store_true",
                        help="问答时启用 QA 聚类路由和聚类级 Prompt")
    parser.add_argument("--cluster-top-k", type=int, default=None,
                        help="新问题最多使用几个相近聚类 Prompt，默认 1")
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
        item.add_argument("--cluster-examples", default="",
                          help="可选 QA JSON；执行聚类级 Prompt 流程")
        item.add_argument("--cluster-count", type=int, default=None)
        item.add_argument("--cluster-iterations", type=int, default=None)
        if command == "run":
            item.add_argument("--query", default="", help="入库后立即回答一条问题")
            item.add_argument("--top-k", type=int, default=5)
    ask = sub.add_parser("ask", help="对已有运行目录检索并生成答案")
    _common(ask)
    ask.add_argument("--query", required=True)
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--context-chars", type=int, default=0, help="每个 chunk 上限；0 表示传入完整 chunk")
    batch = sub.add_parser("batch-ask", help="对问答数据集批量生成答案并保留三路 Top-5")
    _common(batch)
    batch.add_argument("--dataset", required=True, help="问答 JSON/JSONL 数据集")
    batch.add_argument("--output", required=True, help="批量问答 JSONL 输出路径")
    batch.add_argument("--top-k", type=int, default=5)
    batch.add_argument("--context-chars", type=int, default=0, help="每个 chunk 上限；0 表示完整 chunk")
    batch.add_argument("--concurrency", type=int, default=1, help="并发问答数")
    batch.add_argument("--no-resume", action="store_true", help="不读取已有输出，重新处理全部问题")
    evaluate = sub.add_parser("evaluate", help="使用 Golden 测试集批量评估三路检索")
    _common(evaluate)
    evaluate.add_argument("--dataset", required=True, help="Golden JSON/JSONL 测试集")
    evaluate.add_argument("--output", default="", help="评测输出目录，默认写入 run-dir/evaluation")
    evaluate.add_argument("--top-k", type=int, default=10, help="最终融合结果的 Top-K")
    evaluate.add_argument("--eval-depth", type=int, default=20, help="每一路用于评测的候选深度")
    evaluate.add_argument("--ks", default="1,3,5,10,20", help="评测 K，逗号分隔")
    evaluate.add_argument("--result-text-chars", type=int, default=800, help="JSONL 中每个结果保留的文本长度")
    evaluate.add_argument("--no-query-expansion", action="store_true", help="评测时不执行查询扩展")
    opt = sub.add_parser("optimize-prompt", help="用问答样本执行案例级 Prompt 迭代")
    _common(opt)
    opt.add_argument("--examples", required=True)
    opt.add_argument("--prompt-iterations", type=int, default=3)
    cluster = sub.add_parser("cluster-prompts", help="QA 聚类并生成聚类级群智 Prompt")
    _common(cluster)
    cluster.add_argument("--examples", required=True, help="问答样本 JSON")
    cluster.add_argument("--cluster-count", type=int, default=None)
    cluster.add_argument("--cluster-iterations", type=int, default=None)
    module = sub.add_parser("prompt-module", help="独立 Prompt 模块：JSON 输入到 JSON 输出")
    _common(module)
    module.add_argument("--input", required=True, help="问答样本 JSON")
    module.add_argument("--output", required=True, help="优化 Prompt JSON 输出路径")
    module.add_argument("--base-prompt", default="", help="可选基础 Prompt JSON")
    module.add_argument("--prompt-iterations", type=int, default=None)
    compare = sub.add_parser("compare", help="逐问题比较 baseline 与 candidate 评测结果")
    compare.add_argument("--baseline", required=True, help="baseline results.jsonl")
    compare.add_argument("--candidate", required=True, help="candidate results.jsonl")
    compare.add_argument("--output", required=True, help="对比输出目录")
    compare.add_argument("--ks", default="1,3,5,10,20", help="对比 K，逗号分隔")
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
        "cluster_top_k": args.cluster_top_k,
        "mock": True if args.mock else None,
    }.items() if value is not None}
    if args.no_openai_compat:
        overrides["llm_openai_compat"] = False
    if args.mock:
        overrides["backend"] = "local"
    if getattr(args, "cluster_prompts", False):
        overrides["cluster_prompt_enabled"] = True
    if getattr(args, "cluster_count", None) is not None:
        overrides["cluster_count"] = args.cluster_count
    return Settings.from_env(args.run_dir, **overrides)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "compare":
        result = compare_evaluations(
            baseline_path=args.baseline,
            candidate_path=args.candidate,
            output_dir=args.output,
            ks=args.ks,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

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
            if args.cluster_examples:
                settings.cluster_prompt_enabled = True
                cluster_result = run_cluster_prompting(
                    settings, args.cluster_examples,
                    cluster_count=args.cluster_count,
                    iterations=args.cluster_iterations,
                )
                manifest["cluster_prompting"] = {
                    "artifact": cluster_result.get("artifact_path", ""),
                    "sample_count": cluster_result.get("metrics", {}).get("sample_count", 0),
                    "cluster_count": cluster_result.get("metrics", {}).get("cluster_count_actual", 0),
                }
                (settings.run_dir / "run_manifest.json").write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            if args.command == "run" and args.query:
                result = QAService(run_dir=settings.run_dir, settings=settings).answer(args.query, top_k=args.top_k)
                print(json.dumps({
                    "query": result["query"],
                    "retrieval_query": result["retrieval_query"],
                    "answer": result["answer"],
                    "retrieval_top5": result["retrieval_top5"],
                }, ensure_ascii=False, indent=2))
            else:
                print(json.dumps(manifest, ensure_ascii=False, indent=2))
        elif args.command == "cluster-prompts":
            result = run_cluster_prompting(
                settings, args.examples,
                cluster_count=args.cluster_count,
                iterations=args.cluster_iterations,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "optimize-prompt":
            result = optimize_prompt(settings, args.examples, args.prompt_iterations)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "evaluate":
            manifest_path = Path(args.run_dir) / "run_manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if args.backend is None:
                    settings.backend = manifest.get("backend", settings.backend)
                if args.collection is None:
                    settings.collection = manifest.get("collection", settings.collection)
            output_dir = args.output or str(Path(args.run_dir) / "evaluation")
            result = run_evaluation(
                run_dir=args.run_dir,
                dataset_path=args.dataset,
                settings=settings,
                output_dir=output_dir,
                top_k=args.top_k,
                eval_depth=args.eval_depth,
                ks=args.ks,
                result_text_chars=args.result_text_chars,
                query_expansion=not args.no_query_expansion,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "batch-ask":
            result = run_batch_qa(
                run_dir=args.run_dir,
                dataset_path=args.dataset,
                output_path=args.output,
                settings=settings,
                top_k=args.top_k,
                context_chars=args.context_chars,
                concurrency=args.concurrency,
                resume=not args.no_resume,
            )
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
