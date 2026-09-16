"""配置与路径约定。新项目只使用环境变量和命令行参数，不读取旧项目配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    run_dir: Path
    input_path: Optional[Path] = None
    backend: str = "qdrant"
    qdrant_url: str = "http://127.0.0.1:6333"
    collection: str = "rag_core_chunks_v1"
    vector_dim: int = 1024
    model_path: str = ""
    embedding_device: str = "auto"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.siliconflow.cn/v1"
    llm_model: str = "Qwen/Qwen3-8B"
    llm_openai_compat: bool = True
    llm_interval: float = 0.0
    denoise_method: str = "mechanical"
    l_min: int = 400
    l_max: int = 1000
    max_dimensions_per_request: int = 15
    tag_text_chars: int = 3000
    top_k: int = 5
    dim_alpha: float = 0.2
    prompt_iterations: int = 0
    mock: bool = False

    @classmethod
    def from_env(cls, run_dir: str | Path, **overrides) -> "Settings":
        values = {
            "run_dir": Path(run_dir),
            "backend": os.getenv("RAG_BACKEND", "qdrant"),
            "qdrant_url": os.getenv("QDRANT_URL", "http://127.0.0.1:6333"),
            "collection": os.getenv("QDRANT_COLLECTION_NAME", "rag_core_chunks_v1"),
            "vector_dim": int(os.getenv("EMBEDDING_DIM", "1024")),
            "model_path": os.getenv("BGE_MODEL_PATH", ""),
            "embedding_device": os.getenv("EMBEDDING_DEVICE", "auto"),
            "llm_api_key": os.getenv("LLM_API_KEY", "") or os.getenv("DASHSCOPE_API_KEY", ""),
            "llm_base_url": os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1"),
            "llm_model": os.getenv("LLM_MODEL", "Qwen/Qwen3-8B"),
            "llm_openai_compat": _env_bool("LLM_OPENAI_COMPAT", True),
            "llm_interval": float(os.getenv("LLM_API_INTERVAL", "0")),
            "denoise_method": os.getenv("DENOISE_METHOD", "mechanical"),
            "l_min": int(os.getenv("CHUNK_L_MIN", "400")),
            "l_max": int(os.getenv("CHUNK_L_MAX", "1000")),
            "max_dimensions_per_request": int(os.getenv("DIMENSIONS_PER_REQUEST", "15")),
            "tag_text_chars": int(os.getenv("TAG_TEXT_CHARS", "3000")),
            "top_k": int(os.getenv("TOP_K", "5")),
            "dim_alpha": float(os.getenv("DIM_ALPHA", "0.2")),
            "prompt_iterations": int(os.getenv("PROMPT_ITERATIONS", "0")),
            "mock": _env_bool("RAG_MOCK", False),
        }
        values.update(overrides)
        values["run_dir"] = Path(values["run_dir"])
        return cls(**values)

    def ensure_dirs(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def apply_runtime_environment(self) -> None:
        """让复制进来的 v2 LLM 服务与本项目配置保持一致。"""
        os.environ["LLM_OPENAI_COMPAT"] = "1" if self.llm_openai_compat else "0"
        if self.llm_api_key:
            os.environ["LLM_API_KEY"] = self.llm_api_key
        os.environ["LLM_BASE_URL"] = self.llm_base_url
        os.environ["LLM_MODEL"] = self.llm_model
        os.environ["LLM_API_INTERVAL"] = str(self.llm_interval)
