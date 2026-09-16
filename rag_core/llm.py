"""一个同时支持 DashScope 和 OpenAI 兼容接口的最小 LLM 客户端。"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Optional


class LLMClient:
    def __init__(self, *, api_key: str = "", base_url: str = "", model: str = "",
                 openai_compat: bool = True, interval: float = 0.0, mock: bool = False):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.openai_compat = openai_compat
        self.interval = max(0.0, float(interval))
        self.mock = mock
        self._last_request = 0.0
        self._lock = threading.Lock()

    def _wait(self) -> None:
        if self.interval <= 0:
            return
        with self._lock:
            wait = self.interval - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()

    @staticmethod
    def parse_json(text: str) -> Any:
        raw = str(text or "").strip()
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, re.I)
        if match:
            raw = match.group(1).strip()
        starts = [i for i in (raw.find("{"), raw.find("[")) if i >= 0]
        if starts:
            start = min(starts)
            end = max(raw.rfind("}"), raw.rfind("]"))
            if end > start:
                raw = raw[start:end + 1]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return json.loads(raw, strict=False)

    def complete(self, system: str, user: str, *, temperature: float = 0.2,
                 max_tokens: int = 800) -> str:
        if self.mock:
            return ""
        if not self.api_key:
            raise RuntimeError("未配置 LLM_API_KEY 或 DASHSCOPE_API_KEY")
        self._wait()
        if self.openai_compat:
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model,
                "messages": [{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if "qwen3" in self.model.lower():
                payload["extra_body"] = {"enable_thinking": False}
        else:
            url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
            payload = {
                "model": self.model or "qwen-plus",
                "input": {"messages": [{"role": "system", "content": system},
                                         {"role": "user", "content": user}]},
                "parameters": {"temperature": temperature, "max_tokens": max_tokens},
            }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail[:500]}") from exc
        if self.openai_compat:
            choices = body.get("choices") or []
            if not choices:
                raise RuntimeError(f"LLM 返回空 choices: {body}")
            return str((choices[0].get("message") or {}).get("content") or "").strip()
        return str((body.get("output") or {}).get("text") or "").strip()

    def complete_json(self, system: str, user: str, *, temperature: float = 0.1,
                      max_tokens: int = 1200, repair_prompt: Optional[str] = None) -> Any:
        text = self.complete(system, user, temperature=temperature, max_tokens=max_tokens)
        try:
            return self.parse_json(text)
        except Exception as first_error:
            if not repair_prompt:
                raise first_error
            repaired = self.complete(
                "你是严格的 JSON 修复器，只输出合法 JSON。",
                repair_prompt + "\n原始输出：\n" + text,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return self.parse_json(repaired)

