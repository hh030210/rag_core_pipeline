"""可独立输入输出的 Prompt 迭代优化模块。

输入一个 JSON 文件，输出一个可被问答流程直接读取的优化 Prompt JSON。
该模块不依赖 Qdrant、分片或问答检索，可以单独运行，也可以被主流程复用。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .llm import LLMClient
from .prompting import PromptOptimizer
from .settings import Settings


PROMPT_MODULE_VERSION = "1.0"


def _read_json(path: str | Path) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {file_path}")
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"不是合法 JSON: {file_path}: {exc}") from exc


def load_prompt_input(path: str | Path) -> Dict[str, Any]:
    """读取模块输入。

    支持三种形式：
    1. 直接传入问答样本数组；
    2. {"examples": [...], "base_prompt": {...}, "config": {...}}；
    3. 单条样本对象 {"question": ..., "context": ..., ...}。
    """
    raw = _read_json(path)
    if isinstance(raw, list):
        return {"examples": raw, "base_prompt": {}, "config": {}}
    if isinstance(raw, dict) and "question" in raw:
        return {"examples": [raw], "base_prompt": {}, "config": {}}
    if not isinstance(raw, dict):
        raise ValueError("Prompt 模块输入必须是数组、单条样本对象或包含 examples 的对象")

    examples = raw.get("examples", raw.get("cases", []))
    if not isinstance(examples, list):
        raise ValueError("输入字段 examples/cases 必须是数组")
    base_prompt = raw.get("base_prompt", raw.get("prompt", {}))
    if base_prompt is None:
        base_prompt = {}
    if not isinstance(base_prompt, dict):
        raise ValueError("输入字段 base_prompt/prompt 必须是对象")
    config = raw.get("config", {})
    if not isinstance(config, dict):
        config = {}
    return {"examples": examples, "base_prompt": base_prompt, "config": config}


def load_base_prompt(path: str | Path) -> Dict[str, Any]:
    """读取可选的基础 Prompt 文件，允许使用 module 包装字段。"""
    raw = _read_json(path)
    if isinstance(raw, dict) and isinstance(raw.get("module"), dict):
        return raw["module"]
    if not isinstance(raw, dict):
        raise ValueError("基础 Prompt 文件必须是 JSON 对象")
    return raw


class PromptOptimizationModule:
    """独立的 Prompt 输入输出模块。"""

    def __init__(self, *, client: LLMClient, work_dir: str | Path = "./runs/prompt_module"):
        self.client = client
        self.work_dir = Path(work_dir)

    def optimize(
        self,
        examples: List[Dict[str, Any]],
        *,
        base_prompt: Optional[Dict[str, Any]] = None,
        iterations: int = 3,
        output_path: str | Path,
        input_path: str | Path = "",
    ) -> Dict[str, Any]:
        output = Path(output_path)
        optimizer = PromptOptimizer(client=self.client, run_dir=self.work_dir)
        optimized = optimizer.optimize(
            examples,
            iterations=max(0, int(iterations)),
            initial_module=base_prompt if base_prompt else None,
            output_path=output,
        )
        result = {
            **optimized,
            "module_version": PROMPT_MODULE_VERSION,
            "io": {
                "input_path": str(input_path),
                "output_path": str(output),
                "example_count": len([item for item in examples if isinstance(item, dict) and item.get("question")]),
                "iterations": max(0, int(iterations)),
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def run_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        base_prompt_path: str | Path = "",
        iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload = load_prompt_input(input_path)
        base_prompt = payload.get("base_prompt") or {}
        if base_prompt_path:
            base_prompt = load_base_prompt(base_prompt_path)
        config = payload.get("config") or {}
        effective_iterations = int(iterations if iterations is not None else config.get("iterations", 3))
        return self.optimize(
            payload["examples"],
            base_prompt=base_prompt,
            iterations=effective_iterations,
            output_path=output_path,
            input_path=input_path,
        )


def build_client(settings: Settings) -> LLMClient:
    settings.apply_runtime_environment()
    return LLMClient(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        openai_compat=settings.llm_openai_compat,
        interval=settings.llm_interval,
        mock=settings.mock,
    )


def run_prompt_module(
    input_path: str | Path,
    output_path: str | Path,
    settings: Settings,
    *,
    base_prompt_path: str | Path = "",
    iterations: Optional[int] = None,
) -> Dict[str, Any]:
    """供主流程和外部脚本调用的显式输入输出函数。"""
    module = PromptOptimizationModule(client=build_client(settings), work_dir=Path(output_path).parent)
    return module.run_file(
        input_path,
        output_path,
        base_prompt_path=base_prompt_path,
        iterations=iterations,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="独立 Prompt 迭代优化：JSON 输入 → JSON 输出")
    parser.add_argument("--input", required=True, help="问答样本 JSON")
    parser.add_argument("--output", required=True, help="优化 Prompt JSON 输出路径")
    parser.add_argument("--base-prompt", default="", help="可选基础 Prompt JSON")
    parser.add_argument("--iterations", type=int, default=None, help="迭代轮数，默认读取输入 config 或 3")
    parser.add_argument("--run-dir", default="./runs/prompt_module", help="模块运行目录")
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-interval", type=float, default=None)
    parser.add_argument("--no-openai-compat", action="store_true")
    parser.add_argument("--mock", action="store_true", help="不访问网络，输出默认 Prompt 模块")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = {
        key: value
        for key, value in {
            "llm_api_key": args.llm_api_key,
            "llm_base_url": args.llm_base_url,
            "llm_model": args.llm_model,
            "llm_interval": args.llm_interval,
            "mock": True if args.mock else None,
        }.items()
        if value is not None
    }
    if args.no_openai_compat:
        overrides["llm_openai_compat"] = False
    settings = Settings.from_env(args.run_dir, **overrides)
    result = run_prompt_module(
        args.input,
        args.output,
        settings,
        base_prompt_path=args.base_prompt,
        iterations=args.iterations,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

