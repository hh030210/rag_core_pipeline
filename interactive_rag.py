#!/usr/bin/env python3
"""Interactive retrieval and QA against an existing RAG Core Qdrant collection.

The script is read-only with respect to Qdrant. Runtime prompt/query/vector caches
are kept under ~/.cache/rag_core_interactive rather than inside a shared run dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _candidate_run_dirs(manifest_path: Path, manifest: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = [manifest_path.parent]
    raw_run_dir = manifest.get("run_dir")
    if raw_run_dir:
        path = Path(str(raw_run_dir)).expanduser()
        candidates.extend([path, PROJECT_ROOT / path])
    artifacts = manifest.get("artifacts") or {}
    if isinstance(artifacts, dict):
        for key in ("schema", "inverted_index"):
            raw = artifacts.get(key)
            if raw:
                path = Path(str(raw)).expanduser()
                candidates.extend([path.parent, PROJECT_ROOT / path.parent])
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        marker = str(resolved)
        if marker not in seen:
            seen.add(marker)
            unique.append(resolved)
    return unique


def _manifest_collections(manifest_path: Path, manifest: dict[str, Any]) -> list[str]:
    values = [manifest.get("collection")]
    store = manifest.get("store") or {}
    if isinstance(store, dict):
        values.append(store.get("collection"))
    store_manifest = _read_json(manifest_path.parent / "store_manifest.json")
    values.append(store_manifest.get("collection"))
    return list(dict.fromkeys(str(value).strip() for value in values if value))


def discover_run_dirs() -> dict[str, Path]:
    roots = [PROJECT_ROOT / "result", PROJECT_ROOT / "runs"]
    if Path("/home/humq/rag_core_runs").is_dir():
        roots.append(Path("/home/humq/rag_core_runs"))
    if os.getenv("RAG_RUNS_ROOT"):
        roots.append(Path(os.environ["RAG_RUNS_ROOT"]).expanduser())

    mapping: dict[str, tuple[float, Path]] = {}
    seen_manifests: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for manifest_path in root.rglob("run_manifest.json"):
            try:
                manifest_path = manifest_path.resolve()
            except OSError:
                continue
            if manifest_path in seen_manifests:
                continue
            seen_manifests.add(manifest_path)
            manifest = _read_json(manifest_path)
            for collection in _manifest_collections(manifest_path, manifest):
                for run_dir in _candidate_run_dirs(manifest_path, manifest):
                    if (run_dir / "V_core_v2.json").is_file() and (run_dir / "inverted_index_v2.json").is_file():
                        try:
                            modified = manifest_path.stat().st_mtime
                        except OSError:
                            modified = 0
                        prior = mapping.get(collection)
                        if prior is None or modified > prior[0]:
                            mapping[collection] = (modified, run_dir)
                        break
    return {name: item[1] for name, item in mapping.items()}


def qdrant_collections(qdrant_url: str) -> list[dict[str, Any]]:
    request = urllib.request.Request(qdrant_url.rstrip("/") + "/collections")
    with urllib.request.urlopen(request, timeout=8) as response:
        body = json.loads(response.read().decode("utf-8"))
    return list((body.get("result") or {}).get("collections") or [])


def qdrant_collection_info(qdrant_url: str, name: str) -> dict[str, Any]:
    request = urllib.request.Request(qdrant_url.rstrip("/") + "/collections/" + urllib.parse.quote(name, safe=""))
    with urllib.request.urlopen(request, timeout=8) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body.get("result") or {}


def choose_collection(qdrant_url: str, known_dirs: dict[str, Path]) -> tuple[str, Path]:
    collections = qdrant_collections(qdrant_url)
    if not collections:
        raise RuntimeError(f"Qdrant {qdrant_url} 中没有 collection")
    details: dict[str, dict[str, Any]] = {}
    for item in collections:
        name = str(item.get("name", ""))
        if name in known_dirs:
            try:
                details[name] = qdrant_collection_info(qdrant_url, name)
            except Exception:
                details[name] = {}

    # Prefer the most recently created corpus collection with matching local
    # schema artifacts, while allowing the user to select any mapped collection.
    choices = [str(item.get("name", "")) for item in collections if str(item.get("name", "")) in known_dirs]
    choices.sort(key=lambda name: (
        0 if details.get(name, {}).get("status") == "green" else 1,
        -int(details.get(name, {}).get("points_count") or 0),
        name,
    ))
    if not choices:
        raise RuntimeError(
            "Qdrant 有 collection，但没有找到匹配的 V_core_v2.json 和 inverted_index_v2.json。"
            "请使用 --collection 和 --run-dir 指定对应的 collection 与运行目录。"
        )

    print("可用 collection（仅列出本机存在对应维度 Schema/索引的项）：")
    for index, name in enumerate(choices, 1):
        info = details.get(name, {})
        count = info.get("points_count", "未知")
        status = info.get("status", "未知")
        print(f"  {index}. {name}  points={count}  status={status}")
        print(f"     run-dir: {known_dirs[name]}")
    while True:
        selected = input("选择序号或输入 collection 名称 [1]: ").strip() or "1"
        if selected.isdigit() and 1 <= int(selected) <= len(choices):
            name = choices[int(selected) - 1]
            return name, known_dirs[name]
        if selected in choices:
            return selected, known_dirs[selected]
        print("输入无效，请输入上面的序号或 collection 名称。")


def resolve_config(args: argparse.Namespace) -> tuple[str, Path, dict[str, Any]]:
    known_dirs = discover_run_dirs()
    manifest: dict[str, Any] = {}
    collection = args.collection or ""
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else None

    if run_dir is not None:
        manifest_path = run_dir / "run_manifest.json"
        manifest = _read_json(manifest_path)
        if not collection:
            collection = str(manifest.get("collection") or "")
            if not collection:
                collection = str((_read_json(run_dir / "store_manifest.json")).get("collection") or "")
        if not collection:
            raise RuntimeError("指定了 --run-dir，但目录的 run_manifest/store_manifest 中没有 collection。请再传 --collection。")
    elif collection:
        run_dir = known_dirs.get(collection)
        if run_dir is None:
            raise RuntimeError(f"没有为 collection {collection!r} 找到运行目录，请通过 --run-dir 指定。")
        manifest = _read_json(run_dir / "run_manifest.json")
    else:
        collection, run_dir = choose_collection(args.qdrant_url, known_dirs)
        manifest = _read_json(run_dir / "run_manifest.json")

    assert run_dir is not None
    if not (run_dir / "V_core_v2.json").is_file():
        raise FileNotFoundError(f"Schema 文件不存在：{run_dir / 'V_core_v2.json'}")
    if not (run_dir / "inverted_index_v2.json").is_file():
        raise FileNotFoundError(f"维度倒排索引不存在：{run_dir / 'inverted_index_v2.json'}")
    return collection, run_dir, manifest


def isolated_runtime_dir(source_run_dir: Path, collection: str) -> Path:
    """Copy small read artifacts so QA/cache writes cannot touch shared run data."""
    token = hashlib.sha256(collection.encode("utf-8")).hexdigest()[:16]
    target_dir = Path.home() / ".cache" / "rag_core_interactive" / token
    target_dir.mkdir(parents=True, exist_ok=True)
    copy_names = (
        "V_core_v2.json",
        "inverted_index_v2.json",
        "optimized_prompt.json",
        "cluster_prompting.json",
        "dimension_label_ontology.json",
        "dimension_tag_evidence_vectors_v2.npy",
        "dimension_tag_evidence_vectors_v2.json",
    )
    for name in copy_names:
        source = source_run_dir / name
        target = target_dir / name
        if not source.is_file():
            continue
        try:
            same = target.is_file() and source.stat().st_size == target.stat().st_size and source.stat().st_mtime_ns == target.stat().st_mtime_ns
        except OSError:
            same = False
        if not same:
            shutil.copy2(source, target)
    return target_dir


def _display_results(title: str, items: list[dict[str, Any]], limit: int) -> None:
    print(f"\n{title}（Top {min(limit, len(items))}）")
    if not items:
        print("  无结果")
        return
    for index, item in enumerate(items[:limit], 1):
        score = item.get("score", item.get("final_score", 0.0))
        title_text = str(item.get("doc_title") or item.get("source_file") or "未命名来源").replace("\n", " ")
        chunk_id = str(item.get("chunk_id") or item.get("id") or "")
        source = str(item.get("source_file") or "")
        dims = item.get("matched_dimensions") or []
        text = str(item.get("chunk_text_full") or item.get("chunk_text") or "").strip().replace("\n", " ")
        snippet = text[:260] + ("…" if len(text) > 260 else "")
        print(f"  {index}. score={float(score or 0):.4f}  {title_text}")
        print(f"     chunk={chunk_id}" + (f"  source={source}" if source else ""))
        if dims:
            print("     命中维度=" + "、".join(str(value) for value in dims))
        if snippet:
            print(f"     {snippet}")


def _show_retrieval(retrieval: dict[str, Any], top_k: int) -> None:
    _display_results("语义检索", retrieval.get("semantic_results") or [], top_k)
    _display_results("维度检索", retrieval.get("dimension_results") or [], top_k)
    _display_results("融合检索", retrieval.get("fusion_results") or [], top_k)
    strategy = retrieval.get("fusion_strategy") or {}
    if strategy:
        print(f"\n融合策略：{strategy.get('name', '')} / {strategy.get('branch', '')}")
    analysis = retrieval.get("query_analysis") or {}
    constraints = analysis.get("constraints") or {}
    if constraints:
        print("查询维度：" + "；".join(
            f"{key}={value.get('labels') or value.get('intent_terms') or []}"
            for key, value in constraints.items() if isinstance(value, dict)
        ))


def _answer_one(service: Any, query: str, top_k: int, context_chars: int) -> None:
    result = service.answer(query, top_k=top_k, context_chars=context_chars)
    _show_retrieval(result.get("retrieval") or {}, top_k)
    print("\n回答：\n" + (result.get("answer") or "（模型未返回答案）"))
    sources = [str(item.get("chunk_id", "")) for item in (result.get("retrieval", {}).get("top_chunks") or [])]
    if sources:
        print("\n答案证据 chunk：" + "、".join(value for value in sources if value))


def _search_one(service: Any, query: str, top_k: int) -> None:
    expansion = service.prompts.expand(query)
    retrieval_query = " | ".join(expansion.get("sub_queries") or [query])
    retrieval = service.retriever.search(retrieval_query, top_k=top_k)
    _show_retrieval(retrieval, top_k)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="基于现有 Qdrant collection 的交互式检索与问答")
    parser.add_argument("--collection", default="", help="Qdrant collection；省略时交互选择")
    parser.add_argument("--run-dir", default="", help="collection 对应且含 V_core_v2.json 的运行目录")
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"))
    parser.add_argument("--model-path", default=os.getenv("BGE_MODEL_PATH", ""))
    parser.add_argument("--device", default=os.getenv("EMBEDDING_DEVICE", "auto"))
    parser.add_argument(
        "--top-n", "--top-k", dest="top_n", type=int, default=5,
        help="语义、维度、融合三路统一返回 Top-N（默认 5；--top-k 为兼容别名）",
    )
    parser.add_argument("--context-chars", type=int, default=0, help="每条证据最大字符数；0 表示完整 chunk")
    parser.add_argument("--fusion-strategy", choices=("adaptive_report_v1", "dynamic_weights"), default="")
    parser.add_argument("--list-collections", action="store_true", help="列出 Qdrant collection 及其可用运行目录")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    known_dirs = discover_run_dirs()
    if args.list_collections:
        collections = qdrant_collections(args.qdrant_url)
        for item in collections:
            name = str(item.get("name", ""))
            try:
                info = qdrant_collection_info(args.qdrant_url, name)
            except Exception:
                info = {}
            run_dir = known_dirs.get(name)
            print(f"{name}\tpoints={info.get('points_count', '未知')}\tstatus={info.get('status', '未知')}\trun_dir={run_dir or '未映射'}")
        return 0

    collection, source_run_dir, manifest = resolve_config(args)
    runtime_dir = isolated_runtime_dir(source_run_dir, collection)

    if not args.model_path:
        config = manifest.get("experiment_config") or {}
        args.model_path = str(config.get("embedding_model") or "")
    if args.model_path:
        os.environ["BGE_MODEL_PATH"] = args.model_path
    elif not os.getenv("BGE_MODEL_PATH"):
        raise RuntimeError("找不到 BGE 模型路径；请设置 BGE_MODEL_PATH 或传入 --model-path。")

    llm_config = manifest.get("experiment_config") or {}
    if not os.getenv("LLM_API_KEY") and os.getenv("DASHSCOPE_API_KEY"):
        os.environ["LLM_API_KEY"] = os.environ["DASHSCOPE_API_KEY"]
    if not os.getenv("LLM_BASE_URL"):
        os.environ["LLM_BASE_URL"] = str(llm_config.get("llm_base_url") or "http://127.0.0.1:8911/v1")
    if not os.getenv("LLM_MODEL"):
        os.environ["LLM_MODEL"] = str(llm_config.get("llm_model") or "DeepSeek-V4-Pro")
    if args.fusion_strategy:
        os.environ["RAG_FUSION_STRATEGY"] = args.fusion_strategy

    from rag_core.qa import QAService
    from rag_core.settings import Settings
    # Some deployed snapshots call the fusion snapshot writer with an extra
    # positional Top-K list even though the helper accepts only three route
    # lists. Interactive retrieval does not need that persisted report, so
    # disable the writer in this process only; this also keeps shared project
    # output directories untouched.
    from rag_core import retrieval as retrieval_module
    if hasattr(retrieval_module, "_save_fusion_snapshot"):
        retrieval_module._save_fusion_snapshot = lambda *args, **kwargs: None

    settings = Settings.from_env(
        runtime_dir,
        backend="qdrant",
        qdrant_url=args.qdrant_url,
        collection=collection,
        model_path=args.model_path or os.getenv("BGE_MODEL_PATH", ""),
        embedding_device=args.device,
        top_k=max(1, args.top_n),
    )
    if not settings.llm_api_key:
        print("当前 shell 未配置 LLM_API_KEY；输入一次 API Key 以启用生成式问答。")
        import getpass
        settings.llm_api_key = getpass.getpass("LLM API Key: ").strip()
        if not settings.llm_api_key:
            raise RuntimeError("缺少 LLM API Key，无法启动问答。可在运行 shell 中设置 LLM_API_KEY。")

    print(f"collection：{collection}")
    print(f"运行目录：{source_run_dir}")
    print(f"嵌入模型：{settings.model_path}")
    print(f"LLM：{settings.llm_model} @ {settings.llm_base_url}")
    print("正在加载 collection 和检索模型；首次启动会建立独立的本地标签向量缓存……", flush=True)
    service = QAService(run_dir=runtime_dir, settings=settings)
    top_n = max(1, args.top_n)
    print(f"每路返回 Top-{top_n}。交互已就绪：直接输入问题执行检索和问答；:search 问题只检索；:topn 数字调整返回数量；:help 查看命令；:quit 退出。")

    while True:
        try:
            line = input("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return 0
        if not line:
            continue
        command, _, remainder = line.partition(" ")
        lowered = command.lower()
        if lowered in {":quit", ":exit", "/quit", "/exit", "quit", "exit"}:
            print("已退出。")
            return 0
        if lowered in {":help", "/help", "help"}:
            print("直接输入问题：三路检索 + 证据问答\n:search 问题：只执行三路检索\n:ask 问题：显式执行检索 + 问答\n:topn 数字：设置每路返回数量\n:quit：退出")
            continue
        if lowered in {":topn", "/topn"}:
            try:
                requested = int(remainder.strip())
                if requested < 1:
                    raise ValueError
            except ValueError:
                print("用法：:topn 10（请输入大于 0 的整数）")
                continue
            top_n = requested
            print(f"已设置语义、维度、融合三路均返回 Top-{top_n}。")
            continue
        query = remainder.strip() if lowered in {":search", "/search", ":ask", "/ask"} else line
        if not query:
            print("请在命令后输入问题。")
            continue
        try:
            if lowered in {":search", "/search"}:
                _search_one(service, query, top_n)
            else:
                _answer_one(service, query, top_n, max(0, args.context_chars))
        except Exception as exc:
            print(f"本轮处理失败：{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
