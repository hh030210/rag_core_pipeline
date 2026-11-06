"""BGE-M3 编码器，找不到模型时只在 mock 模式使用确定性哈希向量。"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Iterable, List


class EmbeddingModel:
    def __init__(self, model_path: str = "", device: str = "auto", dimension: int = 1024,
                 mock: bool = False):
        self.model_path = model_path or os.getenv("BGE_MODEL_PATH", "")
        self.device = device
        self.dimension = int(dimension)
        self.mock = mock
        self._model = None
        self.resolved_device = "cpu" if mock else self._resolve_device(device)
        if not mock:
            self._load()

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve ``auto`` to CUDA when available instead of forcing CPU."""
        if device and device != "auto":
            return device
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self) -> None:
        candidates = [self.model_path]
        if not self.model_path:
            candidates.extend(["./model/bge-m3", "../model/bge-m3"])
        resolved = next((p for p in candidates if p and os.path.exists(p)), None)
        if not resolved:
            raise FileNotFoundError("未找到 BGE 模型，请设置 BGE_MODEL_PATH；本地测试请使用 --mock")
        target_device = self.resolved_device
        print(f"[向量模型] 加载 BGE-M3，设备={target_device}", flush=True)
        try:
            from FlagEmbedding import BGEM3FlagModel
            self._model = BGEM3FlagModel(resolved, use_fp16=False, device=target_device)
            self._mode = "flag"
            print("[向量模型] 编码后端=FlagEmbedding", flush=True)
            return
        except ImportError:
            pass
        except Exception as exc:
            print(f"[提示] FlagEmbedding 加载失败，尝试 SentenceTransformer: {exc}")
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(resolved, device=target_device, local_files_only=True)
        self._mode = "sentence_transformers"
        print("[向量模型] 编码后端=SentenceTransformer", flush=True)

    def _hash_vector(self, text: str) -> List[float]:
        values = []
        seed = str(text or "").encode("utf-8")
        for index in range(self.dimension):
            digest = hashlib.blake2b(seed + index.to_bytes(4, "little"), digest_size=4).digest()
            values.append((int.from_bytes(digest, "little") / 2147483647.5) - 1.0)
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]

    def encode(self, texts: Iterable[str]) -> List[List[float]]:
        items = [str(text or "") for text in texts]
        if self.mock:
            return [self._hash_vector(text) for text in items]
        if not items:
            return []
        try:
            default_batch = "16" if self.resolved_device.startswith("cuda") else "8"
            batch_size = max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", default_batch)))
        except ValueError:
            batch_size = 8
        if self._mode == "flag":
            result = self._model.encode(items, return_dense=True, batch_size=batch_size)
            vectors = result["dense_vecs"]
        else:
            vectors = self._model.encode(
                items,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return [vector.tolist() if hasattr(vector, "tolist") else list(vector) for vector in vectors]

    def encode_compact(self, texts: Iterable[str]):
        """Encode to a float32 matrix without expanding vectors into Python lists."""
        try:
            import numpy as np
        except ImportError:
            return self.encode(texts)
        items = [str(text or "") for text in texts]
        if self.mock:
            return np.asarray([self._hash_vector(text) for text in items], dtype=np.float32)
        if not items:
            return np.empty((0, self.dimension), dtype=np.float32)
        try:
            default_batch = "16" if self.resolved_device.startswith("cuda") else "8"
            batch_size = max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", default_batch)))
        except ValueError:
            batch_size = 8
        if self._mode == "flag":
            result = self._model.encode(items, return_dense=True, batch_size=batch_size)
            vectors = result["dense_vecs"]
        else:
            vectors = self._model.encode(
                items,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        return np.asarray(vectors, dtype=np.float32)
