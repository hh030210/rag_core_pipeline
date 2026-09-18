"""
dimension_search.py

基于本地 Milvus 的维度感知检索。

流程（与 evaluation_ours_base.py 逻辑一致，但适配本地 Milvus）：
    query_text
        │
        ▼
┌──────────────────────────────────────────────┐
│ Step 1: Milvus 向量检索                      │
│ BGE-M3 encode → Top-200 候选 doc_id          │
└────────────────────┬─────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                        ▼
┌────────────────────┐  ┌────────────────────────┐
│ Step 2: QueryParser  │  │ Step 3: 标签软匹配     │
│ 解析 query → 约束    │  │ Enum: 严格匹配 1.0分  │
│ {dim: [val1, val2]} │  │ Open:  向量相似度得分  │
└────────┬───────────┘  └────────────┬───────────┘
         │                           │
         └──────────┬────────────────┘
                    ▼
┌──────────────────────────────────────────────┐
│ Step 4: 融合排序                             │
│   - Score: vec_score + α * tag_score         │
│   - RRF:   1/(K+vec_rank) + α*1/(K+tag_rank) │
│ 输出最终 top_k 排序结果                       │
└──────────────────────────────────────────────┘

使用方式：
    python dimension_search.py -q "儿童发烧咳嗽怎么办"
    python dimension_search.py -q "老年人腰腿痛" --top_k 5 --fusion rrf
    python dimension_search.py -q "孕妇感冒用药" --fusion score --alpha 0.3
    python dimension_search.py -i          # 交互模式
    python dimension_search.py --debug     # 打印详细匹配日志
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from contextlib import contextmanager

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
try:
    import numpy as np
except ImportError:
    np = None
try:
    from pymilvus import MilvusClient
except ImportError:
    MilvusClient = None
try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    cosine_similarity = None

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
CODE_DIR = PROJECT_ROOT / "code"
EXPERIMENT_DATA = CODE_DIR / "experiment_data"
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from FlagEmbedding import BGEM3FlagModel
    _HAS_BGE = True
except ImportError:
    _HAS_BGE = False
    print("[警告] FlagEmbedding 未安装，将使用替代方案")

try:
    from .query_parser import QueryParser, QueryParserV2
    from .index_builder import IndexConfig
    from .schema_v2 import SCHEMA_VERSION, load_schema, normalize_label, normalize_text, schema_maps
except ImportError:
    from query_parser import QueryParser, QueryParserV2
    from index_builder import IndexConfig
    from schema_v2 import SCHEMA_VERSION, load_schema, normalize_label, normalize_text, schema_maps


# ===================== 路径配置 =====================

MILVUS_DB = str(EXPERIMENT_DATA / "experiment_data.db")
COLLECTION_NAME = "CmedqaRetrieval_Sampled"      # 评测用 collection
# COLLECTION_NAME = "test_collection"            # 换成你的实际 collection

PATH_INVERTED_INDEX = EXPERIMENT_DATA / "inverted_index_med_D.json"
PATH_DIM_META = EXPERIMENT_DATA / "dimension_metadata_med_D.json"
PATH_TAG_VECTORS = EXPERIMENT_DATA / "tag_vectors_med_D.pkl"
PATH_DOC_TAGS = EXPERIMENT_DATA / "tags_output.json"


# ===================== 配置 =====================

class SearchConfig:
    VECTOR_TOP_K = 200       # 向量检索召回数量
    RRF_K = 60               # RRF 平滑常数
    DEFAULT_ALPHA = 0.2      # 默认 α（标签权重）
    SOFT_MATCH_THRESHOLD = 0.65  # 向量软匹配阈值
    EXCLUDED_DIMS: Set[str] = {"适宜人群", "适用阶段"}  # 排除不参与匹配的维度


# ===================== 辅助工具 =====================

@contextmanager
def _suppress_stderr():
    with open(os.devnull, 'w') as devnull:
        old = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old


def _load_encoder():
    """加载 BGE-M3 编码器（懒加载，屏蔽 tqdm 刷屏）"""
    if not _HAS_BGE:
        return None
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, 'bge-m3')
    if not os.path.exists(model_path):
        model_path = str(PROJECT_ROOT / "model" / "bge-m3")
    try:
        return BGEM3FlagModel(model_path, use_fp16=True, device='cuda')
    except Exception as e:
        print(f"[警告] BGE-M3 加载失败: {e}")
        return None


# ===================== 核心检索类 =====================

class DimensionAwareSearch:
    """
    维度感知检索器。

    整合向量检索 + 标签软匹配 + 多路融合排序，
    输出带维度标签解释的最终排序结果。
    """

    def __init__(self, collection_name: str = COLLECTION_NAME, milvus_db: str = MILVUS_DB):
        print(">>> 初始化检索器...")

        self.collection_name = collection_name
        self.milvus_db = milvus_db

        # Milvus 客户端
        self.client = MilvusClient(uri=self.milvus_db)
        if not self.client._has_collection(self.collection_name):
            raise RuntimeError(
                f"Collection '{self.collection_name}' 不存在。"
                f"请检查 db_config.yaml 中的 vecdb.collection_name 配置。"
            )

        # Query 解析器（带 LLM 缓存）
        self.parser = QueryParser()

        # BGE-M3 编码器
        self.encoder = _load_encoder()
        if self.encoder is None:
            print("[警告] 编码器未加载，软匹配将不可用")

        # 加载索引数据
        self._load_indexes()

        # 加载正向索引（doc_id -> tags）
        self._build_forward_doc_tags()

        # 加载文档文本存储（SQLite）
        self._load_doc_texts()

        print(f"    Collection: {self.collection_name}")
        print(f"    维度数: {len(self.dim_meta)}")
        print(f"    倒排索引覆盖: {len(self.inverted_index)} 个维度")
        print(f"    文档标签: {len(self.doc_tags)} 篇")

    # -------------------- 索引加载 --------------------

    def _load_indexes(self):
        """加载倒排索引、维度元数据、标签向量"""
        self.inverted_index = {}
        if PATH_INVERTED_INDEX.exists():
            with open(PATH_INVERTED_INDEX, 'r', encoding='utf-8') as f:
                self.inverted_index = json.load(f)
            print(f"    倒排索引: {PATH_INVERTED_INDEX.name} ({len(self.inverted_index)} 个维度)")

        self.dim_meta = {}
        if PATH_DIM_META.exists():
            with open(PATH_DIM_META, 'r', encoding='utf-8') as f:
                self.dim_meta = json.load(f)
            print(f"    维度元数据: {PATH_DIM_META.name}")

        self.tag_vectors: Dict[str, Any] = {}
        if PATH_TAG_VECTORS.exists():
            with open(PATH_TAG_VECTORS, 'rb') as f:
                self.tag_vectors = pickle.load(f)
            print(f"    标签向量: {PATH_TAG_VECTORS.name} ({len(self.tag_vectors)} 个维度)")

        # 回退到 experiment_data 根目录
        if not self.inverted_index:
            alt = EXPERIMENT_DATA / "inverted_index.json"
            if alt.exists():
                with open(alt, 'r', encoding='utf-8') as f:
                    self.inverted_index = json.load(f)
                print(f"    [回退] 倒排索引: {alt.name}")

    def _build_forward_doc_tags(self):
        """从倒排索引反向构建：doc_id -> {dim: [val1, val2]}"""
        self.doc_tags: Dict[str, Dict[str, List[str]]] = {}
        for dim, tag_docs in self.inverted_index.items():
            for tag_val, doc_ids in tag_docs.items():
                for doc_id in doc_ids:
                    did = str(doc_id)
                    if did not in self.doc_tags:
                        self.doc_tags[did] = {}
                    if dim not in self.doc_tags[did]:
                        self.doc_tags[did][dim] = []
                    if tag_val not in self.doc_tags[did][dim]:
                        self.doc_tags[did][dim].append(tag_val)

    def _load_doc_texts(self):
        """加载文档文本（从 SQLite 或 tags_output.json）"""
        self.doc_texts: Dict[str, str] = {}

        # 优先从 tags_output.json 提取文本
        if PATH_DOC_TAGS.exists():
            try:
                # tags_output.json 格式: {doc_id: {dim: [vals], "doc_text": "..."}}
                with open(PATH_DOC_TAGS, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                for doc_id, data in raw.items():
                    if isinstance(data, dict):
                        self.doc_texts[str(doc_id)] = data.get("doc_text", "")
            except Exception as e:
                print(f"[警告] 加载 tags_output.json 失败: {e}")

        print(f"    文档文本: {len(self.doc_texts)} 篇")

    # -------------------- 向量检索 --------------------

    def _vector_search(self, query_text: str, top_k: int = None) -> List[Tuple[str, float]]:
        """
        Step 1: Milvus 向量检索。
        返回: [(doc_id, distance), ...]
        """
        top_k = top_k or SearchConfig.VECTOR_TOP_K

        if self.encoder is None:
            print("[错误] 编码器不可用，无法进行向量检索")
            return []

        # Encode
        with _suppress_stderr():
            raw_vec = self.encoder.encode(
                [query_text],
                return_dense=True
            )['dense_vecs'][0]
        query_vec = raw_vec.astype(np.float32).tolist()

        try:
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_vec],
                limit=top_k,
                output_fields=["id"]
            )
            return [(hit['id'], float(hit['distance'])) for hit in results[0]]
        except Exception as e:
            print(f"[错误] Milvus 检索失败: {e}")
            return []

    # -------------------- 标签匹配 --------------------

    def _soft_match_tags(
        self,
        dim: str,
        raw_val: str,
        threshold: float = None
    ) -> Dict[str, float]:
        """
        Step 2: 对单个维度的值做向量软匹配。
        返回: {matched_tag: similarity_score}
        例如: {"腹痛": 0.87, "胃痛": 0.72}
        """
        threshold = threshold or SearchConfig.SOFT_MATCH_THRESHOLD

        if dim not in self.tag_vectors or self.encoder is None:
            return {}

        target = self.tag_vectors[dim]
        with _suppress_stderr():
            query_vec = self.encoder.encode(
                [raw_val],
                return_dense=True
            )['dense_vecs']

        scores = cosine_similarity(query_vec, target['vectors'])[0]
        matched = {}
        for idx, score in enumerate(scores):
            if score >= threshold:
                matched[target['values'][idx]] = float(score)
        return matched

    def _get_tag_scores(
        self,
        constraints: Dict[str, List[str]]
    ) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        """
        Step 3: 将 Query 约束匹配到文档上，计算标签得分。
        返回: (doc_tag_scores, doc_tag_evidence)
            doc_tag_scores: {doc_id: total_score}
            doc_tag_evidence: {doc_id: [evidence_string, ...]}
        """
        if not constraints:
            return {}, {}

        doc_scores: Dict[str, float] = {}
        doc_evidence: Dict[str, List[str]] = {}

        for dim, vals in constraints.items():
            if dim in SearchConfig.EXCLUDED_DIMS:
                continue
            if dim not in self.inverted_index:
                continue

            # 记录当前维度下每个 doc 的最高得分（防止同维度近义词重复加分）
            dim_doc_scores: Dict[str, float] = {}
            dim_doc_ev: Dict[str, str] = {}

            for v in vals:
                is_enum = self.dim_meta.get(dim, {}).get("is_enum", False)

                if is_enum:
                    # Enum: 严格匹配，得 1.0 分
                    if v in self.inverted_index[dim]:
                        for doc_id in self.inverted_index[dim][v]:
                            dim_doc_scores[doc_id] = max(dim_doc_scores.get(doc_id, 0.0), 1.0)
                            dim_doc_ev[doc_id] = f"[{dim}] 枚举命中 '{v}'"
                else:
                    # Open: 向量软匹配
                    matched_with_scores = self._soft_match_tags(dim, v)
                    # 字面完全一致优先给 1.0
                    if v in self.inverted_index[dim] and v not in matched_with_scores:
                        matched_with_scores[v] = 1.0

                    for tag_val, sim_score in matched_with_scores.items():
                        for doc_id in self.inverted_index[dim][tag_val]:
                            dim_doc_scores[doc_id] = max(dim_doc_scores.get(doc_id, 0.0), sim_score)
                            dim_doc_ev[doc_id] = (
                                f"[{dim}] Query '{v}' → 软命中 '{tag_val}' "
                                f"(相似度 {sim_score:.2f})"
                            )

            # 累加到全局得分
            for doc_id, score in dim_doc_scores.items():
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score
                if doc_id not in doc_evidence:
                    doc_evidence[doc_id] = []
                doc_evidence[doc_id].append(dim_doc_ev.get(doc_id, ""))

        return doc_scores, doc_evidence

    # -------------------- 融合排序 --------------------

    def _fusion_score(
        self,
        vec_candidates: List[Tuple[str, float]],
        doc_tag_scores: Dict[str, float],
        alpha: float
    ) -> List[Tuple[str, float, dict]]:
        """
        策略 A: Score 分数融合。
        在向量召回的 Top-K 候选池内重排。
        返回: [(doc_id, final_score, meta), ...]
        """
        results = []
        for doc_id, vec_score in vec_candidates:
            t_score = doc_tag_scores.get(doc_id, 0.0)
            final = vec_score + alpha * t_score
            meta = {
                "vec_score": vec_score,
                "tag_score": t_score,
                "vec_contrib": vec_score,
                "tag_contrib": alpha * t_score
            }
            results.append((doc_id, final, meta))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _fusion_rrf(
        self,
        vec_candidates: List[Tuple[str, float]],
        doc_tag_scores: Dict[str, float],
        alpha: float,
        k: int = None
    ) -> List[Tuple[str, float, dict]]:
        """
        策略 B: RRF 倒数排名融合。
        可跨两路（向量 + 标签）合并所有候选。
        返回: [(doc_id, rrf_score, meta), ...]
        """
        k = k or SearchConfig.RRF_K

        # 构建排名映射
        vec_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(vec_candidates)}
        sorted_tag_docs = sorted(doc_tag_scores.items(), key=lambda x: x[1], reverse=True)
        tag_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(sorted_tag_docs)}

        # 合并所有候选
        all_ids = set(vec_ranks.keys()) | set(tag_ranks.keys())
        results = []
        for doc_id in all_ids:
            v_rank = vec_ranks.get(doc_id, float('inf'))
            t_rank = tag_ranks.get(doc_id, float('inf'))

            v_rrf = 1.0 / (k + v_rank) if v_rank != float('inf') else 0.0
            t_rrf = 1.0 / (k + t_rank) if t_rank != float('inf') else 0.0

            final = v_rrf + alpha * t_rrf
            vec_score = next((s for d, s in vec_candidates if d == doc_id), 0.0)
            t_score = doc_tag_scores.get(doc_id, 0.0)
            meta = {
                "vec_score": vec_score,
                "tag_score": t_score,
                "vec_rank": v_rank if v_rank != float('inf') else None,
                "tag_rank": t_rank if t_rank != float('inf') else None,
                "vec_contrib": v_rrf,
                "tag_contrib": alpha * t_rrf
            }
            results.append((doc_id, final, meta))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # -------------------- 公开 API --------------------

    def search_base(
        self,
        query_text: str,
        top_k: int = 10,
        anns_field: str = None
    ) -> Dict[str, Any]:
        """
        基础检索：纯向量 ANN 搜索，不带维度标签融合。

        Args:
            query_text: 查询文本
            top_k: 返回结果数
            anns_field: 向量字段名，默认 chunk_text_vec

        Returns:
            {
                "query": str,
                "results": [{"rank", "doc_id", "vec_score", "doc_text", "tags"}, ...]
            }
        """
        anns_field = anns_field or "chunk_text_vec"

        if self.encoder is None:
            print("[错误] 编码器不可用")
            return {"query": query_text, "results": []}

        with _suppress_stderr():
            raw_vec = self.encoder.encode(
                [query_text],
                return_dense=True
            )['dense_vecs'][0]
        query_vec = raw_vec.astype(np.float32).tolist()

        try:
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_vec],
                limit=top_k,
                output_fields=["id"]
            )
        except Exception as e:
            print(f"[错误] Milvus 检索失败: {e}")
            return {"query": query_text, "results": []}

        output = []
        for rank, hit in enumerate(results[0], 1):
            doc_id = hit['id']
            output.append({
                "rank": rank,
                "doc_id": doc_id,
                "vec_score": round(float(hit['distance']), 6),
                "tags": self.doc_tags.get(str(doc_id), {}),
                "doc_text": self.doc_texts.get(str(doc_id), self.doc_store_text(doc_id))[:200].replace("\n", " "),
            })

        return {
            "query": query_text,
            "top_k": top_k,
            "anns_field": anns_field,
            "results": output
        }

    def search(
        self,
        query_text: str,
        top_k: int = 10,
        fusion: str = "rrf",
        alpha: float = None,
        return_raw_vec: bool = False,
        debug: bool = False
    ) -> Dict[str, Any]:
        """
        端到端检索入口。

        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            fusion: 融合策略，"score" 或 "rrf"
            alpha: 标签权重（默认 0.2）
            return_raw_vec: 是否包含纯向量基线结果
            debug: 是否打印详细日志

        Returns:
            {
                "query": str,
                "constraints": {dim: [vals]},
                "vec_candidates": [(doc_id, vec_score), ...],
                "tag_scores": {doc_id: score},
                "results": [
                    {
                        "doc_id": str,
                        "final_score": float,
                        "vec_score": float,
                        "tag_score": float,
                        "tags": {dim: [vals]},
                        "doc_text": str,
                        "evidence": [str, ...]
                    },
                    ...
                ]
            }
        """
        alpha = alpha if alpha is not None else SearchConfig.DEFAULT_ALPHA

        # ---------- Step 1: 向量检索 ----------
        vec_candidates = self._vector_search(query_text, SearchConfig.VECTOR_TOP_K)
        vec_scores_dict = {d: s for d, s in vec_candidates}

        # ---------- Step 2: Query 解析 ----------
        constraints = self.parser.parse(
            qid="search",
            query_text=query_text,
            dim_meta=self.dim_meta
        )

        # ---------- Step 3: 标签匹配 ----------
        doc_tag_scores, doc_tag_evidence = self._get_tag_scores(constraints)

        # ---------- Step 4: 融合排序 ----------
        if fusion.lower() == "score":
            fused = self._fusion_score(vec_candidates, doc_tag_scores, alpha)
        else:
            fused = self._fusion_rrf(vec_candidates, doc_tag_scores, alpha)

        # ---------- Step 5: 构造输出 ----------
        output_results = []
        for rank, (doc_id, final_score, meta) in enumerate(fused[:top_k], 1):
            # 文档维度标签
            tags = self.doc_tags.get(str(doc_id), {})
            # 文档文本
            doc_text = self.doc_texts.get(str(doc_id), "")
            if not doc_text:
                doc_text = self.doc_store_text(doc_id)
            # 证据链
            evidence = doc_tag_evidence.get(str(doc_id), [])

            output_results.append({
                "rank": rank,
                "doc_id": str(doc_id),
                "final_score": round(float(final_score), 6),
                "vec_score": round(float(meta["vec_score"]), 6),
                "tag_score": round(float(meta["tag_score"]), 6),
                "vec_rank": meta.get("vec_rank"),
                "tag_rank": meta.get("tag_rank"),
                "vec_contrib": round(float(meta.get("vec_contrib", 0)), 6),
                "tag_contrib": round(float(meta.get("tag_contrib", 0)), 6),
                "tags": tags,
                "doc_text": doc_text[:200].replace("\n", " ") if doc_text else "",
                "evidence": evidence
            })

        # ---------- Debug 日志 ----------
        if debug:
            print(f"\n{'='*60}")
            print(f"[DEBUG] Query: {query_text}")
            print(f"[DEBUG] 解析约束: {json.dumps(constraints, ensure_ascii=False)}")
            print(f"[DEBUG] 标签命中: {len(doc_tag_scores)} 篇文档")
            for doc_id, score in sorted(doc_tag_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
                ev = doc_tag_evidence.get(doc_id, [])
                print(f"  doc_id={doc_id}, score={score:.2f}, evidence={ev[:2]}")
            print(f"{'='*60}")

        return {
            "query": query_text,
            "constraints": constraints,
            "fusion": fusion,
            "alpha": alpha,
            "vec_candidates_count": len(vec_candidates),
            "tag_hit_count": len(doc_tag_scores),
            "results": output_results,
            "raw_vec_results": vec_candidates[:top_k] if return_raw_vec else []
        }

    def doc_store_text(self, doc_id: str) -> str:
        """尝试从本地 tags_output.json 获取文档文本"""
        did = str(doc_id)
        if did in self.doc_texts:
            return self.doc_texts[did]
        # 尝试从 tags_output.json 动态加载
        if PATH_DOC_TAGS.exists():
            try:
                with open(PATH_DOC_TAGS, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                if did in raw and isinstance(raw[did], dict):
                    return raw[did].get("doc_text", "")
            except:
                pass
        return ""


class HierarchicalDimensionSearchV2:
    """Hierarchical postings scorer with array-aware multi-label semantics.

    ``semantic_scores`` is supplied by the existing vector retriever.  This
    class owns only the dimension branch, so parent expansion and score
    aggregation can be tested independently of Milvus and an encoder.
    """

    def __init__(self, schema, index, similarity_fn=None):
        self.schema = load_schema(schema, allow_legacy=False, strict=False)
        self.maps = schema_maps(self.schema)
        self.index = index or {}
        self.postings = self.index.get("postings", self.index.get("inverted_index", {}))
        self.similarity_fn = similarity_fn
        self.tag_details = self.index.get("tag_details", {})

    def _leaf_ids_for(self, dimension_id):
        if dimension_id in self.maps["leaves"]:
            return [dimension_id]
        return list(self.maps["children"].get(dimension_id, []))

    def _path(self, dimension_id):
        return " / ".join(
            self.maps["nodes"][node_id]["name"]
            for node_id in self.maps["paths"].get(dimension_id, [dimension_id])
            if node_id in self.maps["nodes"]
        )

    def _matches_for_label(self, leaf_id, label):
        normalized = normalize_label(label)
        exact = self.postings.get(leaf_id, {}).get(normalized, [])
        matches = {str(doc_id): (1.0, normalized, "exact") for doc_id in exact}
        if matches or self.similarity_fn is None:
            return matches
        for matched_label, score in self.similarity_fn(leaf_id, label) or []:
            score = float(score)
            if score <= 0:
                continue
            for doc_id in self.postings.get(leaf_id, {}).get(normalize_label(matched_label), []):
                doc_id = str(doc_id)
                previous = matches.get(doc_id)
                if previous is None or score > previous[0]:
                    matches[doc_id] = (score, normalize_label(matched_label), "vector")
        return matches

    def score_constraints(self, constraints, semantic_scores=None, semantic_top_n=50):
        """Return scores/evidence using OR-within-leaf and sum-across-leaves.

        When a query has a dimension intent but no canonical tag matches, the
        optional ``semantic_scores`` argument provides a conservative fallback
        over already-recalled chunk candidates.  This is explicitly marked as
        ``semantic_intent_fallback`` in evidence and receives less weight than
        an exact/vector tag match; it is not presented as a tag hit.
        """

        scores: Dict[str, float] = {}
        evidence: Dict[str, List[Dict[str, Any]]] = {}
        grouped_constraints: Dict[str, List[str]] = {}
        grouped_intents: Dict[str, List[str]] = {}
        for constraint in constraints or []:
            if not isinstance(constraint, dict):
                continue
            dimension_id = str(constraint.get("dimension_id", ""))
            if dimension_id not in self.maps["nodes"]:
                continue
            labels = constraint.get("labels", [])
            if isinstance(labels, str):
                labels = [labels]
            bucket = grouped_constraints.setdefault(dimension_id, [])
            intent_bucket = grouped_intents.setdefault(dimension_id, [])
            seen = {normalize_label(value) for value in bucket}
            for label in labels or []:
                normalized = normalize_label(label)
                if normalized and normalized not in seen:
                    bucket.append(label)
                    seen.add(normalized)
            for term in constraint.get("intent_terms", []) or []:
                term = normalize_text(term)
                if term and normalize_label(term) not in {normalize_label(value) for value in intent_bucket}:
                    intent_bucket.append(term)

        for dimension_id, labels in grouped_constraints.items():
            leaf_ids = self._leaf_ids_for(dimension_id)
            if not leaf_ids:
                continue

            # Each leaf gets one score: max over all query labels.  If the
            # query addressed a parent, all children are then collapsed to one
            # parent score (max over children), avoiding child-count inflation.
            per_leaf: Dict[str, Dict[str, Tuple[float, str, str]]] = {}
            for leaf_id in leaf_ids:
                best_by_doc = {}
                for label in labels:
                    for doc_id, match in self._matches_for_label(leaf_id, label).items():
                        if doc_id not in best_by_doc or match[0] > best_by_doc[doc_id][0]:
                            best_by_doc[doc_id] = match
                per_leaf[leaf_id] = best_by_doc

            if dimension_id in self.maps["children"]:
                collapsed = {}
                for leaf_id, doc_matches in per_leaf.items():
                    for doc_id, match in doc_matches.items():
                        if doc_id not in collapsed or match[0] > collapsed[doc_id][0]:
                            collapsed[doc_id] = (match[0], leaf_id, match[1], match[2])
                for doc_id, (score, leaf_id, matched_label, match_type) in collapsed.items():
                    scores[doc_id] = scores.get(doc_id, 0.0) + score
                    evidence.setdefault(doc_id, []).append({
                        "dimension_id": leaf_id,
                        "dimension_path": self._path(leaf_id),
                        "label": matched_label,
                        "evidence": self.tag_details.get(doc_id, {}).get(leaf_id, []),
                        "similarity": score,
                        "match": match_type,
                        "expanded_from": dimension_id,
                    })
                matched_doc_ids = set(collapsed)
            else:
                matched_doc_ids = set()
                for doc_matches in per_leaf.values():
                    for doc_id, (score, matched_label, match_type) in doc_matches.items():
                        matched_doc_ids.add(doc_id)
                        scores[doc_id] = scores.get(doc_id, 0.0) + score
                        evidence.setdefault(doc_id, []).append({
                            "dimension_id": dimension_id,
                            "dimension_path": self._path(dimension_id),
                            "label": matched_label,
                            "evidence": self.tag_details.get(doc_id, {}).get(dimension_id, []),
                            "similarity": score,
                            "match": match_type,
                        })

            # A query can express a valid dimension intent without using a
            # stored label literally, for example “几点开门” or “修建了多久”.
            # Do not turn every semantic candidate into a tag hit.  Only use
            # this fallback when the entire dimension has no exact/vector
            # match, and apply a bounded score so it cannot dominate tags.
            if not matched_doc_ids and semantic_scores:
                fallback_items = sorted(
                    ((str(doc_id), float(score)) for doc_id, score in semantic_scores.items()),
                    key=lambda item: item[1],
                    reverse=True,
                )[: max(1, int(semantic_top_n))]
                fallback_dimension_id = dimension_id if dimension_id in self.maps["leaves"] else dimension_id
                for doc_id, semantic_score in fallback_items:
                    if semantic_score <= 0:
                        continue
                    fallback_score = min(0.45, max(0.08, semantic_score * 0.45))
                    scores[doc_id] = scores.get(doc_id, 0.0) + fallback_score
                    evidence.setdefault(doc_id, []).append({
                        "dimension_id": fallback_dimension_id,
                        "dimension_path": self._path(fallback_dimension_id),
                        "label": "",
                        "evidence": [],
                        "similarity": round(fallback_score, 6),
                        "match": "semantic_intent_fallback",
                        "intent_terms": grouped_intents.get(dimension_id, []),
                    })
        return {"scores": scores, "evidence": evidence}

    def fuse(self, semantic_scores, constraints, *, alpha=0.5, top_k=10):
        """Fuse semantic scores with dimension scores without duplicate boosts."""

        tag_result = self.score_constraints(constraints, semantic_scores=semantic_scores)
        all_ids = set(str(doc_id) for doc_id in (semantic_scores or {})) | set(tag_result["scores"])
        ranked = []
        for doc_id in all_ids:
            semantic = float((semantic_scores or {}).get(doc_id, 0.0))
            tag_score = float(tag_result["scores"].get(doc_id, 0.0))
            ranked.append({
                "doc_id": doc_id,
                "semantic_score": semantic,
                "dimension_score": tag_score,
                "final_score": semantic + float(alpha) * tag_score,
                "dimension_matches": tag_result["evidence"].get(doc_id, []),
            })
        ranked.sort(key=lambda item: item["final_score"], reverse=True)
        for rank, item in enumerate(ranked[:top_k], 1):
            item["rank"] = rank
        return ranked[:top_k]


# ===================== 格式化输出 =====================

def _format_result(result: Dict[str, Any], top_k: int = 10) -> str:
    """将检索结果格式化为可读字符串"""
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"查询: {result['query']}")
    lines.append(f"融合策略: {result['fusion'].upper()}  |  α={result['alpha']}")
    lines.append(f"维度约束: {json.dumps(result['constraints'], ensure_ascii=False)}")
    lines.append(f"向量召回: {result['vec_candidates_count']} 篇  |  标签命中: {result['tag_hit_count']} 篇")
    lines.append(f"{'='*70}")

    for r in result['results'][:top_k]:
        lines.append(f"\n─── 第 {r['rank']} 名 ───")
        lines.append(f"  doc_id : {r['doc_id']}")
        lines.append(f"  综合分 : {r['final_score']:.4f}")
        lines.append(f"  ├─ 向量分: {r['vec_score']:.4f} (贡献 {r['vec_contrib']:.4f})")
        lines.append(f"  └─ 标签分: {r['tag_score']:.4f} (贡献 {r['tag_contrib']:.4f})")

        if r['tags']:
            tag_str = " | ".join(f"[{d}] {', '.join(vs)}" for d, vs in r['tags'].items())
            lines.append(f"  标签   : {tag_str}")

        if r['evidence']:
            lines.append(f"  命中依据:")
            for ev in r['evidence'][:3]:
                lines.append(f"    ✓ {ev}")

        if r['doc_text']:
            preview = r['doc_text'][:120] + ("..." if len(r['doc_text']) > 120 else "")
            lines.append(f"  文本   : {preview}")

    return "\n".join(lines)


def _print_table(results: List[Dict[str, Any]]):
    """打印多策略对比表"""
    print(f"\n{'='*70}")
    print(f"{'Rank':<5} | {'DocID':<20} | {'VecScore':<10} | {'TagScore':<10} | {'Final':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['rank']:<5} | {r['doc_id']:<20} | {r['vec_score']:<10.4f} | "
              f"{r['tag_score']:<10.4f} | {r['final_score']:<10.4f}")
    print("=" * 70)


# ===================== 主入口 =====================

def main():
    parser_cli = argparse.ArgumentParser(
        description="维度感知检索（本地 Milvus）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python dimension_search.py -q "儿童发烧咳嗽"
  python dimension_search.py -q "儿童发烧咳嗽" --fusion score --alpha 0.3
  python dimension_search.py -q "儿童发烧咳嗽" --top_k 5 --debug
  python dimension_search.py -i
        """
    )
    parser_cli.add_argument("-q", "--query", type=str, help="查询文本")
    parser_cli.add_argument("--top_k", type=int, default=10, help="返回结果数（默认 10）")
    parser_cli.add_argument("--fusion", choices=["base", "score", "rrf"], default="rrf",
                           help="融合策略: base=纯向量检索, score=分数相加, rrf=倒数排名融合（默认 rrf）")
    parser_cli.add_argument("--alpha", type=float, default=None,
                           help=f"标签权重 α（默认 {SearchConfig.DEFAULT_ALPHA}）")
    parser_cli.add_argument("--collection", type=str, default=COLLECTION_NAME,
                           help=f"Milvus Collection 名称（默认 {COLLECTION_NAME}）")
    parser_cli.add_argument("-i", "--interactive", action="store_true", help="交互模式")
    parser_cli.add_argument("--debug", action="store_true", help="打印详细匹配日志")
    parser_cli.add_argument("--compare", action="store_true",
                           help="对比 Base / Score / RRF 三种策略")

    args = parser_cli.parse_args()

    # 初始化检索器
    try:
        searcher = DimensionAwareSearch(collection_name=args.collection)
    except RuntimeError as e:
        print(f"[错误] {e}")
        print("\n请检查以下配置：")
        print(f"  1. Milvus collection 是否存在: {args.collection}")
        print(f"  2. experiment_data/ 下是否有索引文件:")
        print(f"     - {PATH_INVERTED_INDEX.name}")
        print(f"     - {PATH_DIM_META.name}")
        print(f"     - {PATH_TAG_VECTORS.name}")
        return

    print()

    def handle_query(q: str):
        alpha = args.alpha or SearchConfig.DEFAULT_ALPHA
        top_k = args.top_k

        if args.fusion == "base":
            # 纯基础检索
            r_base = searcher.search_base(q, top_k=top_k)
            print(f"\n{'='*70}")
            print(f"[Base] 纯向量检索  |  Top-{top_k}")
            print(f"{'='*70}\n")
            for r in r_base['results']:
                print(f"  #{r['rank']:2d}  {r['doc_id']:<20}  距离={r['vec_score']:.4f}")
                if r['tags']:
                    tag_str = " | ".join(f"[{d}] {','.join(v[:3])}" for d, v in r['tags'].items())
                    print(f"      标签: {tag_str}")
                if r['doc_text']:
                    print(f"      文本: {r['doc_text'][:80]}...")
            print()
            return

        if args.compare:
            # 对比三种策略
            print(f"\n{'='*70}")
            print(f"对比三种策略  |  α={alpha}  |  Top-{top_k}")
            print(f"{'='*70}\n")

            # 1) Base
            r_base = searcher.search_base(q, top_k=top_k)
            print(f"[1/3] Base（纯向量检索）")
            print(f"       查询向量命中: {top_k} 篇\n")
            for r in r_base['results'][:top_k]:
                print(f"       #{r['rank']:2d}  {r['doc_id']:<20}  距离={r['vec_score']:.4f}")
                if r['tags']:
                    tag_str = " | ".join(f"[{d}]" for d in r['tags'])
                    print(f"              标签: {tag_str}")
            print()

            # 2) Score
            r_score = searcher.search(q, top_k=top_k, fusion="score", alpha=alpha, debug=args.debug)
            print(f"[2/3] Score融合  (vec_score + {alpha}×tag_score)")
            for r in r_score['results'][:top_k]:
                vr = f"#{r.get('vec_rank', '?'):>3}" if r.get('vec_rank') else " N/A"
                tag_str = f"tag分={r['tag_score']:.2f}" if r['tag_score'] > 0 else "无标签命中"
                print(f"       #{r['rank']:2d}  {r['doc_id']:<20}  "
                      f"总分={r['final_score']:.4f}  (向量#{vr} {tag_str})")
            print()

            # 3) RRF
            r_rrf = searcher.search(q, top_k=top_k, fusion="rrf", alpha=alpha, debug=args.debug)
            print(f"[3/3] RRF融合    (1/(60+vec_rank) + {alpha}×1/(60+tag_rank))")
            for r in r_rrf['results'][:top_k]:
                vr = f"vec#{r.get('vec_rank', '?'):>3}" if r.get('vec_rank') else " N/A"
                tr = f"tag#{r.get('tag_rank', '?'):>3}" if r.get('tag_rank') else " N/A"
                tag_str = f"tag分={r['tag_score']:.2f}" if r['tag_score'] > 0 else "无标签命中"
                print(f"       #{r['rank']:2d}  {r['doc_id']:<20}  "
                      f"总分={r['final_score']:.4f}  ({vr} {tr} {tag_str})")
            print()

            # 分析 RRF 独有召回
            base_ids = {r['doc_id'] for r in r_base['results']}
            score_ids = {r['doc_id'] for r in r_score['results']}
            rrf_ids = {r['doc_id'] for r in r_rrf['results']}
            rrf_only = rrf_ids - score_ids - base_ids
            score_only = score_ids - rrf_ids - base_ids

            if rrf_only:
                print(f"   → RRF 独有召回 {len(rrf_only)} 篇（Score/Base 未进 Top-{top_k}）:")
                for r in r_rrf['results']:
                    if r['doc_id'] in rrf_only:
                        print(f"     #{r['rank']:2d} {r['doc_id']}  "
                              f"(vec_rank={r.get('vec_rank','?')}  tag_score={r['tag_score']:.2f})")

            if score_only:
                print(f"   → Score 独有召回 {len(score_only)} 篇:")
                for r in r_score['results']:
                    if r['doc_id'] in score_only:
                        print(f"     #{r['rank']:2d} {r['doc_id']}  "
                              f"(vec_rank={r.get('vec_rank','?')}  tag_score={r['tag_score']:.2f})")

            print(f"\n{'='*70}\n")
        else:
            r = searcher.search(
                q,
                top_k=top_k,
                fusion=args.fusion,
                alpha=alpha,
                debug=args.debug
            )
            print(_format_result(r, top_k))

    # 交互模式
    if args.interactive or not args.query:
        print("交互模式（输入 quit 退出）:\n")
        while True:
            q = input("查询: ").strip()
            if q.lower() in ("quit", "exit", "q"):
                break
            if not q:
                continue
            handle_query(q)
            print()
    else:
        handle_query(args.query)


if __name__ == "__main__":
    main()
