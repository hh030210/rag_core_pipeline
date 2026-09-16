#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import ast
import hashlib
import json
import math
import os
import re
import statistics
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from .mechanical_denoiser import MechanicalDenoiser
except ImportError:
    from mechanical_denoiser import MechanicalDenoiser

# --- 基础配置与导入检查 ---
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# --- 数据类定义 ---

@dataclass
class ChunkBlock:
    text: str
    source_doc_id: str
    genre: str
    l_min: int
    l_max: int
    from_large_split: bool = False
    frozen: bool = False
    source_chunk_id: str = ""
    stage: str = "initial"

    @property
    def length(self) -> int:
        return len(self.text)


@dataclass
class SubFileRecommendation:
    """子文件分片推荐配置"""
    doc_id: str
    file_name: str
    genre: str
    word_count: int
    l_min: int
    l_max: int
    model_reasoning: str = ""


# --- 核心辅助类 (LLM, PPL, IG, Splitter) ---

class OpenAICompatClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def chat_json(self, system_prompt: str, user_prompt: str) -> Dict:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.model.lower().startswith("qwen3"):
            payload["extra_body"] = {"enable_thinking": False}
        # 不同 OpenAI 兼容网关对 response_format 的支持不一致。默认不发送
        # json_object，避免自建网关返回 400；调用提示本身仍要求严格 JSON，
        # 响应随后由本类的 json.loads 校验。需要网关强制结构化输出时，可显式
        # 设置 LLM_RESPONSE_FORMAT=json_object 或 json_schema。
        response_format = os.getenv("LLM_RESPONSE_FORMAT", "").strip()
        if response_format:
            payload["response_format"] = {"type": response_format}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url=url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
                parsed = json.loads(raw)
                message = parsed["choices"][0]["message"]
                content = message.get("content") or message.get("reasoning_content") or ""
                if isinstance(content, list):
                    content = "".join(
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in content
                    )
                content = str(content).strip()
                fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
                candidate = fenced.group(1) if fenced else content
                candidate = self._normalize_jsonish(candidate)
                try:
                    value = json.loads(candidate)
                except json.JSONDecodeError:
                    decoder = json.JSONDecoder()
                    value = None
                    for match in re.finditer(r"[\{\[]", candidate):
                        try:
                            value, _ = decoder.raw_decode(candidate[match.start():])
                            if isinstance(value, (dict, list)):
                                break
                        except json.JSONDecodeError:
                            continue
                    if value is None:
                        raise ValueError(f"LLM 未返回可解析 JSON: {content[:300]}")
                if not isinstance(value, dict):
                    raise ValueError("LLM 分片推荐必须返回 JSON 对象")
                return value
        except Exception as e:
            raise RuntimeError(f"LLM API Error: {e}")

    @staticmethod
    def _normalize_jsonish(text: str) -> str:
        """兼容自建网关返回的代码块、行尾注释和 Math.floor 表达式。"""

        def safe_eval(node):
            if isinstance(node, ast.Expression):
                return safe_eval(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return float(node.value)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = safe_eval(node.operand)
                return value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                left, right = safe_eval(node.left), safe_eval(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                return left / right
            raise ValueError("不支持的数值表达式")

        def replace_floor(match):
            try:
                value = safe_eval(ast.parse(match.group(1), mode="eval"))
                return str(math.floor(value))
            except Exception:
                return match.group(0)

        text = re.sub(r"Math\.floor\(([^()]*)\)", replace_floor, text)
        text = re.sub(r"(?m)^\s*//.*$", "", text)
        text = re.sub(r",\s*//.*?(?=\n|$)", ",", text)
        text = re.sub(r",\s*([}\]])", r"\1", text)
        return text.strip()


class LocalHuggingFacePPLScorer:
    def __init__(self, model_name_or_path: str, device: str = None):
        if not HAS_TRANSFORMERS: raise ImportError("Need transformers/torch")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=self.device, torch_dtype="auto", trust_remote_code=True, local_files_only=True).eval()

    def score(self, context: str, target: str) -> float:
        full_text = (context + target) if context else target
        try:
            enc = self.tokenizer(full_text, return_tensors="pt")
            input_ids = enc.input_ids.to(self.device)
            target_start = self.tokenizer(context, return_tensors="pt").input_ids.shape[1] if context else 0
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                loss = torch.nn.CrossEntropyLoss(reduction="none")(outputs.logits[..., :-1, :].contiguous().view(-1, outputs.logits.size(-1)), input_ids[..., 1:].contiguous().view(-1))
                avg_nll = loss.view(input_ids[..., 1:].size())[0, max(0, target_start-1):].mean()
                return math.exp(avg_nll.item())
        except: return 100.0


class CharNgramPPLScorer:
    def score(self, context: str, target: str) -> float:
        if not target: return 1.0
        text = (context or "") + target
        vocab = set(text); v = max(len(vocab), 1)
        log_probs = []
        counts_prev = {}; counts_pair = {}
        for i in range(1, len(context or "")):
            p, c = context[i-1], context[i]
            counts_prev[p] = counts_prev.get(p, 0) + 1
            counts_pair[(p, c)] = counts_pair.get((p, c), 0) + 1
        prev = context[-1] if context else ""
        for ch in target:
            p = (counts_pair.get((prev, ch), 0) + 1.0) / (counts_prev.get(prev, 0) + v)
            log_probs.append(math.log(p))
            prev = ch
        return math.exp(-sum(log_probs) / len(log_probs))


class SentenceSplitter:
    SENT_PATTERN = re.compile(r"[^。！？；!?;\n]+[。！？；!?;]?")
    def split(self, text: str) -> List[str]:
        text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
        parts = [m.group(0).strip() for m in self.SENT_PATTERN.finditer(text) if m.group(0).strip()]
        return parts if parts else [text.strip()] if text.strip() else []


class IGCalculator:
    def embedding(self, text: str) -> Dict[str, float]:
        vec = {}
        for ch in text: vec[ch] = vec.get(ch, 0.0) + 1.0
        norm = math.sqrt(sum(v*v for v in vec.values()))
        if norm > 0:
            for k in vec: vec[k] /= norm
        return vec
    def ig(self, left: ChunkBlock, right: ChunkBlock) -> float:
        e1, e2 = self.embedding(left.text), self.embedding(right.text)
        keys = set(e1.keys()) & set(e2.keys())
        sim = max(-1.0, min(1.0, sum(e1[k] * e2[k] for k in keys)))
        return 1.0 / (1.0 + (1.0 - sim))


class LLMChunkRecommender:
    """使用 LLM 为每个子文件推荐分片大小"""

    # 只拦截明显不可能用于景区知识文档的返回值，不把 LLM 的合法推荐
    # 替换成固定长度。20/60 这类值会让一个自然句直接成为一个 chunk。
    _GENRE_ALIASES = {
        "小说/故事": "小说故事",
        "小说": "小说故事",
        "故事": "小说故事",
        "技术": "技术文档",
        "文档": "技术文档",
        "新闻": "新闻资讯",
        "论文": "学术论文",
        "对话": "对话记录",
    }
    _GENRE_DEFAULTS = {
        "技术文档": (400, 1000),
        "新闻资讯": (300, 600),
        "学术论文": (500, 1200),
        "小说故事": (400, 800),
        "对话记录": (200, 500),
    }
    
    SYSTEM_PROMPT = """你是一个专业的文档分片专家。根据文档的体裁和内容长度，为分片算法推荐合适的最小和最大分片长度。

请分析以下要素：
1. 体裁类型：新闻、论文、技术文档、小说、对话等
2. 内容长度：字数的多少
3. 语义完整性：确保每个分片包含完整的语义单元

输出格式要求为JSON，包含：
- genre: 体裁类型
- l_min: 最小分片长度（字符数）
- l_max: 最大分片长度（字符数）
- reasoning: 推荐理由（简短）

注意：
- 对于新闻/短资讯，l_max 应较小（500-800）
- 对于技术文档/论文，l_max 可以较大（800-1500）
- 对于小说/故事，l_max 适中（600-1200）
- l_min 通常为 l_max 的 40%-60%"""

    USER_PROMPT_TEMPLATE = """请分析以下文档并推荐分片大小：

文档标题/开头：{preview}

体裁特征：{genre_hint}

字数：约 {word_count} 字

请给出分片长度推荐（l_min 和 l_max）："""

    def __init__(self, llm_client: OpenAICompatClient = None):
        self.llm = llm_client
        # 仅对同一结构子段（或完全相同文本）复用推荐，不能按原始文件名
        # 缓存，否则一篇文档的第一个短子段会污染后续所有子段。
        self._cache: Dict[str, SubFileRecommendation] = {}
        self._default_recommendations = {
            "doc": {"genre": "技术文档", "l_min": 400, "l_max": 1000},
            "news": {"genre": "新闻资讯", "l_min": 300, "l_max": 600},
            "paper": {"genre": "学术论文", "l_min": 500, "l_max": 1200},
            "novel": {"genre": "小说故事", "l_min": 400, "l_max": 800},
            "chat": {"genre": "对话记录", "l_min": 200, "l_max": 500},
        }

    @classmethod
    def _normalize_genre(cls, value: object, fallback: str) -> str:
        genre = str(value or "").strip()
        genre = cls._GENRE_ALIASES.get(genre, genre)
        return genre if genre in cls._GENRE_DEFAULTS else fallback

    @classmethod
    def _sanitize_result(cls, result: Dict, default_rec: Dict[str, object]) -> Tuple[str, int, int, str]:
        """校验 LLM 推荐，拒绝明显异常的长度而保留合法推荐。"""
        default_genre = str(default_rec["genre"])
        genre = cls._normalize_genre(result.get("genre"), default_genre)
        default_min, default_max = cls._GENRE_DEFAULTS.get(genre, (400, 1000))
        try:
            l_min = int(float(result.get("l_min")))
            l_max = int(float(result.get("l_max")))
        except (TypeError, ValueError):
            return genre, default_min, default_max, "LLM 长度不是有效数字，使用该体裁默认范围"

        # 对景区知识文档，过小的区间会产生“一句一块”；过大的区间
        # 会让语义检索粒度失去意义。合法推荐仍由 LLM 决定。
        invalid = (
            l_min < 100
            or l_max < 200
            or l_min > l_max
            or l_max > 3000
            or l_min / max(l_max, 1) > 0.85
        )
        if invalid:
            return genre, default_min, default_max, f"LLM 推荐 {l_min}/{l_max} 超出安全范围，使用该体裁默认范围"
        return genre, l_min, l_max, ""

    def _detect_genre_hint(self, content: str) -> str:
        """根据内容特征初步判断体裁"""
        content_lower = content.lower()
        
        # 检测论文特征
        if any(kw in content_lower for kw in ["摘要", "abstract", "参考文献", "引言", "结论"]):
            return "paper"
        # 检测新闻特征
        if any(kw in content_lower for kw in ["日报", "快讯", "据报道", "新华社"]):
            return "news"
        # 检测小说特征
        if any(kw in content_lower for kw in ['"', '"', "他说着", "她说道", "心想"]):
            return "novel"
        # 检测对话特征
        if content.count("：") / max(len(content), 1) > 0.02:
            return "chat"
        
        return "doc"

    def get_recommendation(self, sub: Dict) -> SubFileRecommendation:
        """获取单个子文件的分片推荐（同一文件只调用一次 LLM）"""
        doc_id = sub.get("doc_id", "unknown")
        file_name = sub.get("file_name", "unknown")
        content = sub.get("content", "")
        word_count = len(content)

        # 缓存命中：只按子段身份和文本内容命中。不能使用 source_file，
        # 因为第一阶段会把同一原始文件拆成多个语义不同的结构子段。
        text_hash = hashlib.sha256(str(content).encode("utf-8")).hexdigest()[:16]
        cache_key = f"{file_name}:{text_hash}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return SubFileRecommendation(
                doc_id=doc_id,
                file_name=file_name,
                genre=cached.genre,
                word_count=word_count,
                l_min=cached.l_min,
                l_max=cached.l_max,
                model_reasoning=cached.model_reasoning,
            )

        # 初步判断体裁
        genre_hint = self._detect_genre_hint(content)
        default_rec = self._default_recommendations.get(genre_hint, self._default_recommendations["doc"])

        # 预览内容（前500字）
        preview = content[:500]
        if len(content) > 500:
            preview += "..."

        if self.llm:
            try:
                user_prompt = self.USER_PROMPT_TEMPLATE.format(
                    preview=preview,
                    genre_hint=default_rec["genre"],
                    word_count=word_count
                )
                result = self.llm.chat_json(self.SYSTEM_PROMPT, user_prompt)

                genre, l_min, l_max, validation_note = self._sanitize_result(result, default_rec)
                reasoning = str(result.get("reasoning", "") or "").strip()
                if validation_note:
                    reasoning = f"{reasoning}；{validation_note}" if reasoning else validation_note
                rec = SubFileRecommendation(
                    doc_id=doc_id,
                    file_name=file_name,
                    genre=genre,
                    word_count=word_count,
                    l_min=l_min,
                    l_max=l_max,
                    model_reasoning=reasoning,
                )
                self._cache[cache_key] = rec
                return rec
            except Exception as e:
                print(f"[Warning] LLM 调用失败，使用默认推荐: {e}", file=sys.stderr)

        # 使用默认推荐
        return SubFileRecommendation(
            doc_id=doc_id,
            file_name=file_name,
            genre=default_rec["genre"],
            word_count=word_count,
            l_min=default_rec["l_min"],
            l_max=default_rec["l_max"],
            model_reasoning="使用默认推荐"
        )


class ChunkRecommendationConfig:
    """分片推荐配置文件管理"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.recommendations: Dict[str, SubFileRecommendation] = {}
        self.metadata: Dict[str, Any] = {}
        
        if config_path and Path(config_path).exists():
            self.load()
    
    def add_recommendation(self, rec: SubFileRecommendation):
        """添加推荐配置"""
        self.recommendations[rec.doc_id] = rec
    
    def get_recommendation(self, doc_id: str) -> Optional[SubFileRecommendation]:
        """获取指定文档的推荐配置"""
        return self.recommendations.get(doc_id)
    
    def save(self, path: Optional[str] = None):
        """保存配置文件"""
        save_path = Path(path or self.config_path)
        if not save_path:
            raise ValueError("未指定配置文件路径")
        
        config_data = {
            "metadata": {
                "version": "1.0",
                "created_at": str(Path(__file__).stat().st_mtime if os.path.exists(__file__) else ""),
            },
            "recommendations": {}
        }
        
        for doc_id, rec in self.recommendations.items():
            config_data["recommendations"][doc_id] = {
                "doc_id": rec.doc_id,
                "file_name": rec.file_name,
                "genre": rec.genre,
                "word_count": rec.word_count,
                "l_min": rec.l_min,
                "l_max": rec.l_max,
                "model_reasoning": rec.model_reasoning,
            }
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
    
    def load(self, path: Optional[str] = None):
        """加载配置文件"""
        load_path = Path(path or self.config_path)
        if not load_path or not load_path.exists():
            return
        
        with open(load_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        
        self.metadata = config_data.get("metadata", {})
        
        for doc_id, rec_data in config_data.get("recommendations", {}).items():
            self.recommendations[doc_id] = SubFileRecommendation(**rec_data)


# --- 主流水线类 ---

class IntegratedChunker:
    def __init__(self, args):
        self.args = args
        self.splitter = SentenceSplitter()
        self.ig_calc = IGCalculator()
        self.llm = OpenAICompatClient(args.llm_base_url, args.llm_api_key, args.llm_model, timeout=getattr(args, 'llm_timeout', 120)) if args.llm_api_key else None
        self.ppl_scorer = LocalHuggingFacePPLScorer(args.ppl_model_name) if args.ppl_model_name else CharNgramPPLScorer()
        self.recommender = LLMChunkRecommender(self.llm) if args.llm_api_key else None

        # beta_small 兼容：args 可能缺少此参数
        self.beta_small = getattr(args, 'beta_small', 0.8)
        self.beta = getattr(args, 'beta', 1.1)
        self.window_w = getattr(args, 'window_w', 3)
        self.mechanical_denoiser = MechanicalDenoiser()
        self.denoise_method = getattr(args, "denoise_method", "") or (
            "ppl" if getattr(args, "denoise", True) else "none"
        )
        if self.denoise_method not in {"none", "ppl", "mechanical"}:
            raise ValueError(f"不支持的去噪方式: {self.denoise_method}")

        # 加载推荐配置（用于第三轮分片）
        rec_config_path = getattr(args, 'recommendation_config', '') or ''
        self.recommendation_config = ChunkRecommendationConfig(rec_config_path) if rec_config_path else None
    
    def run(self, input_path: Path, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)

        # 支持目录输入
        if input_path.is_dir():
            files = sorted([p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".md", ".json"}])
            if not files:
                print("[Error] 目录中没有找到 .txt/.md/.json 文件", file=sys.stderr)
                return
            print(f"[Info] 找到 {len(files)} 个文件", file=sys.stderr)

            all_results = []
            for idx, input_file in enumerate(files):
                print(f"[File {idx+1}/{len(files)}] {input_file.name}", file=sys.stderr)
                try:
                    result = self._process_single_file(input_file, output_dir)
                    if result:
                        all_results.extend(result)
                except Exception as e:
                    print(f"[Error] 处理失败: {input_file.name} - {e}", file=sys.stderr)

            # 汇总保存
            if all_results:
                self._save_directory_results(all_results, output_dir)
            return

        # 单文件处理
        self._process_single_file(input_path, output_dir)

    def _process_single_file(self, input_file: Path, output_dir: Path):
        text = input_file.read_text(encoding="utf-8")
        base_name = input_file.stem
        rec_config_path_str = getattr(self.args, 'recommendation_config', '') or ''
        recommendation_path = Path(rec_config_path_str) if rec_config_path_str else None

        print("[Step 1] 正在拆分子文件并调用大模型获取分片推荐...", file=sys.stderr)
        subfiles = self._round1_split(text, recommendation_path, base_name)

        all_results = []
        for i, sub in enumerate(subfiles):
            print(f"[Step 2 & 3] 正在处理子文件 {i+1}/{len(subfiles)}: {sub['file_name']}", file=sys.stderr)
            print(f"  分片推荐: l_min={sub['l_min']}, l_max={sub['l_max']} ({sub.get('genre', 'unknown')})", file=sys.stderr)

            r2_chunks, denoised_text = self._round2_process(sub)
            sub["denoised_content"] = denoised_text
            sub["r2_chunks"] = r2_chunks
            sub["denoise_method"] = self.denoise_method
            sub["denoise_report"] = getattr(self, "_last_denoise_report", {})

            final_chunks = self._round3_process(sub)
            sub["chunks"] = final_chunks
            all_results.append(sub)

        self._save_results(all_results, output_dir, base_name)
        return all_results

    def _save_directory_results(self, all_results: List[Dict], output_dir: Path):
        """保存目录处理汇总结果"""
        base_name = "all_chunks"

        # 保存完整结果
        (output_dir / f"{base_name}_chunks.json").write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        # 保存纯文本分片（兼容 auto_pipeline 格式）
        chunk_lines = []
        for sub in all_results:
            for c in sub.get("chunks", []):
                chunk_lines.append(f"=== {sub.get('file_name', 'unknown')} ===")
                chunk_lines.append(c["chunk_text"])
                chunk_lines.append("")
        (output_dir / f"{base_name}_chunks.txt").write_text("\n".join(chunk_lines), encoding="utf-8")

        # 保存汇总信息
        summary = {
            "total_files": len(set(sub.get("file_name", "") for sub in all_results)),
            "total_chunks": sum(len(sub.get("chunks", [])) for sub in all_results),
            "subfiles": [
                {
                    "doc_id": sub.get("doc_id", ""),
                    "file_name": sub.get("file_name", ""),
                    "genre": sub.get("genre", "unknown"),
                    "chunk_count": len(sub.get("chunks", [])),
                }
                for sub in all_results
            ]
        }
        (output_dir / f"{base_name}_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"[Done] 目录处理完成: {summary['total_chunks']} 个 chunk", file=sys.stderr)

    def _round1_split(self, text: str, recommendation_path: Path = None, file_name: str = "unknown") -> List[Dict]:
        """第一轮：按结构拆分 + LLM 推荐"""
        lines = text.splitlines()
        boundaries = [0]
        base_name = file_name or "doc"
        
        # 结构边界检测
        patterns = [
            re.compile(r"^\s*={3,}.+={0,}\s*$", re.I),  # ===== 标题 =====
            re.compile(r"^\s*#{1,6}\s+.+$"),            # Markdown 标题
            re.compile(r"^\s*-{20,}\s*$"),             # --------... 30个分隔线
            re.compile(r"^\s*-{3}\s*$"),               # --- 分隔线
        ]
        
        for i, line in enumerate(lines):
            if any(p.match(line) for p in patterns):
                boundaries.append(i)
        
        boundaries.append(len(lines))
        
        # 创建子文件并获取 LLM 推荐
        subfiles = []
        recommendations = []
        
        for i in range(len(boundaries) - 1):
            content = "\n".join(lines[boundaries[i]:boundaries[i+1]]).strip()
            if not content:
                continue
            
            doc_id = f"{base_name}_{i:03d}"
            file_name = f"{base_name}_sub_{i:03d}.txt"
            
            # 获取 LLM 推荐
            sub_data = {
                "doc_id": doc_id,
                "file_name": file_name,
                "source_file": base_name,
                "file_type": "doc",
                "content": content,
            }
            
            if self.recommender:
                rec = self.recommender.get_recommendation(sub_data)
                sub_data["genre"] = rec.genre
                sub_data["l_min"] = rec.l_min
                sub_data["l_max"] = rec.l_max
                sub_data["word_count"] = rec.word_count
                sub_data["model_reasoning"] = rec.model_reasoning
                recommendations.append(rec)
            else:
                # 默认值
                sub_data["genre"] = "doc"
                sub_data["l_min"] = 400
                sub_data["l_max"] = 1000
                sub_data["word_count"] = len(content)
                sub_data["model_reasoning"] = "默认推荐（未使用 LLM）"
            
            subfiles.append(sub_data)
        
        # 保存推荐配置（仅当提供了路径时才保存）
        if recommendation_path and (recommendations or subfiles):
            rec_config = ChunkRecommendationConfig(str(recommendation_path))
            if recommendations:
                for rec in recommendations:
                    rec_config.add_recommendation(rec)
            else:
                # 保存默认推荐
                for sub in subfiles:
                    rec_config.add_recommendation(SubFileRecommendation(
                        doc_id=sub["doc_id"],
                        file_name=sub["file_name"],
                        genre=sub["genre"],
                        word_count=sub["word_count"],
                        l_min=sub["l_min"],
                        l_max=sub["l_max"],
                        model_reasoning=sub["model_reasoning"]
                    ))
            rec_config.save()
            print(f"[Step 1] 分片推荐配置已保存至: {recommendation_path}", file=sys.stderr)
        
        return subfiles

    def _round2_process(self, sub: Dict) -> Tuple[List[str], str]:
        """第二轮：按选择的方式去噪，再沿用原 PPL 边界切分。"""
        sents = self.splitter.split(sub["content"])
        method = self.denoise_method
        self._last_denoise_report = {
            "method": method,
            "boundary_splitter": "ppl",
        }

        if method == "mechanical":
            mechanical_result = self.mechanical_denoiser.denoise(sents)
            denoised = mechanical_result["kept_sentences"]
            self._last_denoise_report.update(mechanical_result["summary"])
            self._last_denoise_report["removed"] = mechanical_result["removed"]
            self._last_denoise_report["review"] = mechanical_result["review"]
            ppls = [None] * len(denoised)
        elif method == "ppl":
            if len(sents) <= 1:
                self._last_denoise_report.update({
                    "input_sentence_count": len(sents),
                    "kept_sentence_count": len(sents),
                    "removed_sentence_count": 0,
                    "input_chars": sum(len(s) for s in sents),
                    "kept_chars": sum(len(s) for s in sents),
                    "removed_chars": 0,
                    "removed_char_ratio": 0.0,
                })
                return [sub["content"]], sub["content"]

            # 计算 PPL 并去噪
            ppls = [None] + [self.ppl_scorer.score("".join(sents[max(0, i-self.window_w):i]), sents[i]) for i in range(1, len(sents))]
            # 确保所有 PPL 值都是纯 Python float
            ppls = [float(p) if p is not None else None for p in ppls]
            valid_ppls = [float(p) for p in ppls if p is not None]

            # 手动计算标准差（避免 statistics.pstdev 的类型问题）
            def calc_mean_std(values):
                if not values:
                    return 100.0
                n = len(values)
                mean = sum(values) / n
                if n == 1:
                    return mean
                variance = sum((x - mean) ** 2 for x in values) / n
                std = variance ** 0.5
                return mean, std

            if len(valid_ppls) > 1:
                mean, std = calc_mean_std(valid_ppls)
                t1 = mean + 3 * std
            else:
                t1 = valid_ppls[0] if valid_ppls else 100.0

            denoised = [s for i, s in enumerate(sents) if ppls[i] is None or ppls[i] <= t1]
            if not denoised:
                denoised = sents
            removed = [s for i, s in enumerate(sents) if ppls[i] is not None and ppls[i] > t1]
            self._last_denoise_report.update({
                "input_sentence_count": len(sents),
                "kept_sentence_count": len(denoised),
                "removed_sentence_count": len(removed),
                "input_chars": sum(len(s) for s in sents),
                "kept_chars": sum(len(s) for s in denoised),
                "removed_chars": sum(len(s) for s in removed),
                "removed_char_ratio": round(sum(len(s) for s in removed) / max(sum(len(s) for s in sents), 1), 6),
            })
        else:
            denoised = sents
            ppls = [None] * len(sents)
            self._last_denoise_report.update({
                "input_sentence_count": len(sents),
                "kept_sentence_count": len(sents),
                "removed_sentence_count": 0,
                "input_chars": sum(len(s) for s in sents),
                "kept_chars": sum(len(s) for s in sents),
                "removed_chars": 0,
                "removed_char_ratio": 0.0,
            })

        if len(denoised) <= 1:
            return ["".join(denoised)], "".join(denoised)

        # 计算 PPL 寻找切分点
        ppls2 = [None] + [self.ppl_scorer.score("".join(denoised[max(0, i-self.window_w):i]), denoised[i]) for i in range(1, len(denoised))]
        ppls2 = [float(p) if p is not None else None for p in ppls2]
        valid_ppls2 = [float(p) for p in ppls2 if p is not None]

        def calc_mean_std(values):
            if not values:
                return 100.0
            n = len(values)
            mean = sum(values) / n
            if n == 1:
                return mean
            variance = sum((x - mean) ** 2 for x in values) / n
            std = variance ** 0.5
            return mean, std

        if len(valid_ppls2) > 1:
            mean2, std2 = calc_mean_std(valid_ppls2)
            t2 = mean2 + std2
        else:
            t2 = valid_ppls2[0] if valid_ppls2 else 100.0

        chunks = []
        curr = []
        for i, s in enumerate(denoised):
            if i > 0 and ppls2[i] is not None and ppls2[i] > t2:
                chunks.append("".join(curr))
                curr = [s]
            else:
                curr.append(s)
        if curr:
            chunks.append("".join(curr))

        return chunks, "".join(denoised)

    def _round3_process(self, sub: Dict) -> List[Dict]:
        """第三轮：策略优化与融合（使用推荐配置中的 l_min/l_max）"""
        doc_id = sub.get("doc_id", "")

        # 从推荐配置读取 l_min/l_max
        l_min = sub.get("l_min", 400)
        l_max = sub.get("l_max", 1000)

        # 尝试从配置文件获取（优先级最高）
        if self.recommendation_config and doc_id:
            config_rec = self.recommendation_config.get_recommendation(doc_id)
            if config_rec:
                l_min = config_rec.l_min
                l_max = config_rec.l_max

        # 获取第二轮的 chunks
        r2_chunks = sub.get("r2_chunks", [])
        if not r2_chunks:
            r2_chunks = [c["chunk_text"] for c in sub.get("chunks", [])] if sub.get("chunks") else [sub["content"]]

        blocks = [ChunkBlock(
            text=c,
            source_doc_id=doc_id,
            genre=sub.get("genre", "doc"),
            l_min=l_min,
            l_max=l_max
        ) for c in r2_chunks]
        
        # 处理超大分片 (使用 Strategy 2: 贪心 IG 融合)
        processed = []
        for b in blocks:
            if b.length <= b.l_max:
                processed.append(b)
            else:
                units = [ChunkBlock(
                    text=s,
                    source_doc_id=b.source_doc_id,
                    genre=b.genre,
                    l_min=b.l_min,
                    l_max=b.l_max
                ) for s in self.splitter.split(b.text)]
                
                # 贪心合并
                while len(units) > 1:
                    best_ig = -1
                    best_i = -1
                    for i in range(len(units) - 1):
                        if units[i].length + units[i+1].length <= b.l_max:
                            ig = self.ig_calc.ig(units[i], units[i+1])
                            if ig > best_ig:
                                best_ig = ig
                                best_i = i
                    if best_i == -1:
                        break
                    units[best_i].text += units[best_i+1].text
                    units.pop(best_i + 1)
                
                processed.extend(units)

        # 处理过小分片融合
        final = []
        for b in processed:
            if not final or b.length >= b.l_min * self.beta_small:
                final.append(b)
            else:
                if final[-1].length + b.length <= b.l_max * self.beta:
                    final[-1].text += b.text
                else:
                    final.append(b)
        
        return [{"chunk_text": b.text, "chunk_len": b.length} for b in final]

    def _save_results(self, results: List[Dict], output_dir: Path, base_name: str):
        """保存结果"""
        # 保存完整结果
        (output_dir / f"{base_name}_chunks.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        
        # 保存纯文本分片
        txt_content = "\n".join([
            c["chunk_text"].replace("\n", " ")
            for sub in results
            for c in sub.get("chunks", [])
        ])
        (output_dir / f"{base_name}_chunks.txt").write_text(txt_content, encoding="utf-8")
        
        # 保存汇总信息
        summary = {
            "total_subfiles": len(results),
            "total_chunks": sum(len(sub.get("chunks", [])) for sub in results),
            "subfiles": [
                {
                    "doc_id": sub["doc_id"],
                    "file_name": sub["file_name"],
                    "genre": sub.get("genre", "unknown"),
                    "l_min": sub.get("l_min", 0),
                    "l_max": sub.get("l_max", 0),
                    "chunk_count": len(sub.get("chunks", [])),
                    "word_count": sub.get("word_count", 0),
                }
                for sub in results
            ]
        }
        (output_dir / f"{base_name}_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        
        print(f"[Done] 结果已保存至 {output_dir}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="整合版三阶段分片脚本")
    parser.add_argument("--input", required=True, help="输入文件")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--llm_api_key", default="", help="LLM API Key")
    parser.add_argument("--llm_base_url", default="https://api.openai.com/v1")
    parser.add_argument("--llm_model", default="gpt-3.5-turbo")
    parser.add_argument("--llm_timeout", type=int, default=120, help="LLM 超时秒数（默认120s）")
    parser.add_argument("--ppl_model_name", default="", help="本地 PPL 模型路径")
    parser.add_argument("--window_w", type=int, default=3, help="PPL 窗口大小")
    parser.add_argument("--beta_small", type=float, default=0.8, help="小分片融合阈值")
    parser.add_argument("--beta", type=float, default=1.1, help="长度约束系数")
    parser.add_argument("--recommendation_config", default="", help="分片推荐配置文件路径")
    parser.add_argument(
        "--denoise-method",
        choices=["none", "ppl", "mechanical"],
        default="",
        help="第二阶段去噪方式：none、ppl（旧方案）或 mechanical（新增机械规则方案）",
    )
    parser.add_argument("--denoise", action="store_true", default=True, help="兼容旧参数：启用 PPL 去噪")
    parser.add_argument("--no-denoise", dest="denoise", action="store_false", help="禁用 PPL 去噪")

    args = parser.parse_args()
    IntegratedChunker(args).run(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
