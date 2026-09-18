"""
dimension_retriever.py

基于维度约束的检索模块。
打通 QueryParser → 维度约束 → RDB SQL 过滤 → 最终文档结果。

核心流程：
    query_text
        │
        ▼
┌─────────────────────────────┐
│  1. QueryParser.parse()     │  从 query 解析出维度约束
│     → {dim: [val1, val2]}  │  {适宜人群: [儿童], 疾病类别: [发烧]}
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  2. DimensionConstraint     │  将约束转换为 RDB SQL WHERE 子句
│     → SQL filter expr       │  dim_suitable_population IN ('儿童')
│                              │  AND dim_disease_category IN ('发烧')
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  3. SQL 执行 + Chunk 召回   │  从 MySQL RDB 找到匹配的 doc_id
│     → [doc_id1, doc_id2]   │  再用 Milvus 向量检索精确匹配 chunks
└─────────────────────────────┘
        │
        ▼
    返回带维度标签的检索结果

使用方式：
    python dimension_retriever.py -q "儿童发烧咳嗽"
    python dimension_retriever.py -q "老年人腰腿痛吃什么药" --top_k 5
    python dimension_retriever.py -i  # 交互模式
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from tqdm import tqdm

import mysql.connector

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
CODE_DIR = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from db_config import get_config
from code.query_parser import QueryParser
from code.index_builder import IndexConfig, IndexBuilder


# ===================== 常量 =====================

DIM_META_PATH = CODE_DIR / "experiment_data" / "dimension_metadata.json"
DIM_CORE_PATH = CODE_DIR / "experiment_data" / "V_core.json"
TAGS_PATH = CODE_DIR / "experiment_data" / "tags_output.json"


# ===================== 核心类 =====================

class DimensionConstraintBuilder:
    """
    将维度约束字典转换为 SQL 过滤表达式和参数。
    """

    def __init__(self):
        self._dim_meta: Dict[str, Dict[str, Any]] = {}
        self._dim_cols: Dict[str, str] = {}  # {dim_name: col_name}
        self._load_metadata()

    def _load_metadata(self):
        """加载维度元数据（列名映射、是否枚举等）"""
        if DIM_META_PATH.exists():
            with open(DIM_META_PATH, "r", encoding="utf-8") as f:
                self._dim_meta = json.load(f)

    def build_sql_filter(
        self,
        constraints: Dict[str, List[str]],
        strategy: str = "AND"
    ) -> tuple[Optional[str], list]:
        """
        将维度约束构建为 SQL WHERE 子句。

        Args:
            constraints: {dim: [val1, val2]} 格式的约束
            strategy: 多维度之间用 AND 还是 OR 连接

        Returns:
            (sql_where_clause, params_list)
            例如: ("WHERE dim_disease_category IN (%s,%s) AND ...", ["发烧", "咳嗽", ...])
        """
        if not constraints:
            return None, []

        config = get_config()
        table = config.rdb.table

        clauses = []
        params = []

        for dim, vals in constraints.items():
            if not vals:
                continue

            # 获取维度对应的所有列名
            cols = self._get_dim_cols(dim)
            if not cols:
                continue

            # 为每个列构建 OR 子句（检查任意一列匹配）
            dim_clauses = []
            for col in cols:
                placeholders = ", ".join(["%s"] * len(vals))
                dim_clauses.append(f"`{col}` IN ({placeholders})")
                params.extend(vals)
            
            # 多个列之间用 OR 连接（任意一列匹配即可）
            if dim_clauses:
                clauses.append(f"({' OR '.join(dim_clauses)})")

        if not clauses:
            return None, []

        joiner = f" {strategy} "
        where = joiner.join(clauses)
        return where, params

    def build_milvus_filter(
        self,
        constraints: Dict[str, List[str]]
    ) -> Optional[str]:
        """
        将维度约束构建为 Milvus 过滤表达式。

        Milvus filter syntax:
            dim in ["val1", "val2"] && dim2 in ["val3"]
        """
        if not constraints:
            return None

        parts = []
        for dim, vals in constraints.items():
            if not vals:
                continue
            col = self._dim_to_col(dim)
            if not col:
                continue

            vals_str = ", ".join(f'"{v}"' for v in vals)
            parts.append(f'{col} in [{vals_str}]')

        return " && ".join(parts) if parts else None

    def _dim_to_col(self, dim: str) -> Optional[str]:
        """将维度名映射为 MySQL 列名（带缓存）- 返回第一个列名"""
        first_col = self._get_dim_cols(dim)
        return first_col[0] if first_col else None

    def _get_dim_cols(self, dim: str) -> List[str]:
        """获取维度对应的所有列名（dim_xxx_1, dim_xxx_2, ...）"""
        cache_key = f"_dim_cols_list_{dim}"
        if hasattr(self, cache_key):
            return getattr(self, cache_key)

        base_col = _normalize_col_name(dim)
        cols = []
        for i in range(1, 11):  # 最多10个值
            col = f"{base_col}_{i}"
            cols.append(col)
        
        # 缓存
        setattr(self, cache_key, cols)
        return cols

    def col_to_dim(self, col: str) -> Optional[str]:
        """MySQL 列名反向映射为维度名"""
        if col.startswith("dim_"):
            # 处理新格式: dim_xxx_1, dim_xxx_2, ...
            # 提取 base_dim (去掉末尾的 _数字)
            base = col[4:]  # 去掉 "dim_"
            parts = base.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                base_col = "dim_" + parts[0]
            else:
                base_col = col
            
            # 从元数据中查找
            for dim, meta in self._dim_meta.items():
                if _normalize_col_name(dim) == base_col:
                    return dim
        return None


def _normalize_col_name(dim: str) -> str:
    """将维度名转换为 MySQL 列名（保留中文字符）"""
    key = dim.strip()
    # 只替换空格为下划线，保留所有中文字符
    key = key.replace(" ", "_")
    # 只移除真正危险的字符（控制字符等），保留中文
    key = "".join(c if (c.isalnum() or c in "_") or ord(c) > 127 else "_" for c in key)
    if not key.startswith("dim_"):
        key = "dim_" + key
    return key[:64]


class DimensionRetriever:
    """
    基于维度约束的检索器。

    支持两种检索模式：
    1. pure_filter: 仅用维度标签过滤（不用向量相似度）
    2. hybrid: 维度过滤 + 向量相似度排序
    """

    def __init__(self):
        self.config = get_config()
        self.parser = QueryParser()
        self.builder = DimensionConstraintBuilder()
        self._dim_meta: Dict[str, Dict] = {}
        self._v_core: List[str] = []
        self._tags_cache: Dict[str, Dict[str, List[str]]] = {}
        self._load_cache()

    def _load_cache(self):
        """加载维度元数据和已有标签"""
        if DIM_META_PATH.exists():
            with open(DIM_META_PATH, "r", encoding="utf-8") as f:
                self._dim_meta = json.load(f)

        if DIM_CORE_PATH.exists():
            with open(DIM_CORE_PATH, "r", encoding="utf-8") as f:
                self._v_core = json.load(f)

        if TAGS_PATH.exists():
            with open(TAGS_PATH, "r", encoding="utf-8") as f:
                self._tags_cache = json.load(f)

    # ==================== 公开 API ====================

    def parse_query(self, query_text: str, qid: str = "q_001") -> Dict[str, List[str]]:
        """
        解析查询 → 维度约束。
        复用 QueryParser（带 LLM + 缓存）。
        """
        # 构造 dim_meta（兼容 QueryParser 需要的格式）
        dim_meta = {}
        for dim in self._v_core:
            col = _normalize_col_name(dim)
            is_enum = self._dim_meta.get(dim, {}).get("is_enum", False)
            vals = self._dim_meta.get(dim, {}).get("values", [])
            dim_meta[dim] = {
                "is_enum": is_enum,
                "values": vals,
                "column": col
            }

        result = self.parser.parse(qid, query_text, dim_meta)
        return result

    def retrieve_by_constraints(
        self,
        constraints: Dict[str, List[str]],
        mode: str = "pure_filter",
        top_k: int = 10,
        include_tags: bool = True
    ) -> List[Dict[str, Any]]:
        """
        根据维度约束直接检索文档。

        Args:
            constraints: {dim: [val1, val2]} 维度约束
            mode: "pure_filter" | "hybrid"
            top_k: 返回数量
            include_tags: 是否在结果中包含维度标签

        Returns:
            [
                {"doc_id": "xxx", "dim_适宜人群": ["儿童"], "dim_疾病类别": ["发烧"], ...},
                ...
            ]
        """
        conn = _connect_rdb()
        if not conn:
            return self._retrieve_fallback(constraints, top_k)

        try:
            # Step 1: 构建 SQL 过滤
            where_clause, params = self.builder.build_sql_filter(constraints)
            if not where_clause:
                return []

            # Step 2: 查询 MySQL
            table = self.config.rdb.table
            dim_cols = [self.builder._dim_to_col(d) for d in constraints.keys()]
            dim_cols = [c for c in dim_cols if c]

            # 也查 doc_id 和 doc_text
            select_cols = [self.config.rdb.doc_id_column, "doc_text"]
            select_cols.extend(dim_cols)
            select_cols = list(dict.fromkeys(select_cols))  # 去重，保持顺序

            col_str = ", ".join(f"`{c}`" for c in select_cols)
            sql = f"SELECT {col_str} FROM `{table}` WHERE {where_clause} LIMIT {top_k * 2}"

            cur = conn.cursor(dictionary=True)
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
            cur.close()
            conn.close()

            # Step 3: 格式转换
            results = []
            for row in rows[:top_k]:
                doc_id = row.get(self.config.rdb.doc_id_column)
                record: Dict[str, Any] = {"doc_id": doc_id}

                # 提取维度标签
                if include_tags:
                    for dim, col in zip(constraints.keys(), dim_cols):
                        if col in row and row[col] is not None:
                            val = row[col]
                            if isinstance(val, str) and "," in val:
                                record[col] = val.split(",")
                            else:
                                record[col] = [val] if val else []

                # 原始文本片段
                if "doc_text" in row:
                    record["doc_text"] = (row["doc_text"] or "")[:300]

                results.append(record)

            return results

        except Exception as e:
            print(f"[错误] RDB 查询失败: {e}")
            return self._retrieve_fallback(constraints, top_k)

    def retrieve_hybrid(
        self,
        query_text: str,
        constraints: Dict[str, List[str]],
        top_k: int = 10,
        score_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        混合检索：先用维度标签粗筛 doc_id，再用 Milvus 向量精确匹配 chunks。

        流程：
        1. 维度约束 → 拿到候选 doc_id 集合
        2. Milvus filter: doc_id_link IN [候选 doc_ids] + 向量相似度排序
        3. 返回 top_k 个 chunks（带完整维度标签）
        """
        # Step 1: 粗筛 doc_ids
        candidates = self._粗筛_doc_ids(constraints, top_k=200)
        if not candidates:
            print("[警告] 维度过滤无结果，降级为纯向量检索")
            return self._pure_vector_search(query_text, top_k)

        # Step 2: Milvus 向量检索（带 doc_id 过滤）
        try:
            from query_engine import search_chunks, build_filter_expression
            from db_config import get_config

            # 构建 Milvus filter
            doc_ids_str = ", ".join(f'"{d}"' for d in candidates)
            milvus_filter = f'doc_id_link in [{doc_ids_str}]'

            # 执行向量检索
            chunks = search_chunks(
                query_text=query_text,
                top_k=top_k,
                anns_field="chunk_text_vec",
                expr=milvus_filter,
                return_profile=False
            )

        except Exception as e:
            print(f"[警告] Milvus 检索失败: {e}，降级为纯过滤")
            return self.retrieve_by_constraints(constraints, mode="pure_filter", top_k=top_k)

        # Step 3: 补全维度标签（从 RDB 查 doc_id 对应的标签）
        results = []
        for chunk in chunks:
            doc_id = chunk.get("doc_id_link")
            tags = self._get_doc_tags(doc_id)

            record = {
                "chunk_id": chunk.get("chunk_id"),
                "doc_id": doc_id,
                "doc_title": chunk.get("doc_title"),
                "chunk_text": chunk.get("chunk_text", ""),
                "score": chunk.get("score", 0),
                "tags": tags
            }
            results.append(record)

        return results

    def _粗筛_doc_ids(
        self,
        constraints: Dict[str, List[str]],
        top_k: int = 200
    ) -> List[str]:
        """用维度约束快速获取候选 doc_id 列表"""
        where_clause, params = self.builder.build_sql_filter(constraints)
        if not where_clause:
            return []

        conn = _connect_rdb()
        if not conn:
            return []

        try:
            table = self.config.rdb.table
            sql = f"SELECT `{self.config.rdb.doc_id_column}` FROM `{table}` WHERE {where_clause} LIMIT {top_k}"
            cur = conn.cursor()
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
            cur.close()
            conn.close()
            return [str(r[0]) for r in rows]
        except Exception as e:
            print(f"[错误] 粗筛 doc_id 失败: {e}")
            if conn:
                conn.close()
            return []

    def _get_doc_tags(self, doc_id: str) -> Dict[str, List[str]]:
        """获取单个文档的维度标签"""
        # 优先从缓存读
        if doc_id in self._tags_cache:
            return self._tags_cache[doc_id]

        # 缓存没有则查 RDB
        conn = _connect_rdb()
        if not conn:
            return {}

        try:
            table = self.config.rdb.table
            # 动态获取所有 dim_xxx 列
            dim_cols = [c for c in _fetch_rdb_columns(conn, self.config.rdb.database, table)
                        if c.startswith("dim_")]

            if not dim_cols:
                return {}

            col_str = ", ".join(f"`{c}`" for c in dim_cols)
            sql = f"SELECT {col_str} FROM `{table}` WHERE `{self.config.rdb.doc_id_column}` = %s LIMIT 1"
            cur = conn.cursor(dictionary=True)
            cur.execute(sql, (doc_id,))
            row = cur.fetchone()
            cur.close()
            conn.close()

            if not row:
                return {}

            # 按维度名分组，合并所有 _1, _2, ... 列的值
            tags = {}
            dim_values = {}  # {base_dim: [val1, val2, ...]}
            
            for col, val in row.items():
                if val is not None and val != "":
                    # 从列名提取维度名和索引
                    # 列名格式: dim_xxx_1, dim_xxx_2, ...
                    parts = col.rsplit('_', 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        base_dim = parts[0]
                        if base_dim not in dim_values:
                            dim_values[base_dim] = []
                        dim_values[base_dim].append(str(val))
                    else:
                        # 兼容旧格式：dim_xxx (不带数字后缀)
                        tags[col] = [str(val)]
            
            # 转换为 {dim_xxx_1: [val], dim_xxx_2: [val], ...} 格式
            for base_dim, vals in dim_values.items():
                for i, val in enumerate(vals):
                    col_name = f"{base_dim}_{i + 1}"
                    tags[col_name] = [val]

            return tags
        except Exception as e:
            if conn:
                conn.close()
            return {}

    def _pure_vector_search(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        """纯向量检索兜底"""
        try:
            from query_engine import search_chunks
            chunks = search_chunks(
                query_text=query_text,
                top_k=top_k,
                anns_field="chunk_text_vec",
                return_profile=False
            )
            return [{"chunk_id": c.get("chunk_id"),
                     "doc_id": c.get("doc_id_link"),
                     "chunk_text": c.get("chunk_text", ""),
                     "score": c.get("score", 0),
                     "tags": self._get_doc_tags(c.get("doc_id_link", ""))}
                    for c in chunks]
        except Exception as e:
            print(f"[错误] 向量检索失败: {e}")
            return []

    def _retrieve_fallback(
        self,
        constraints: Dict[str, List[str]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """从本地 tags_output.json 缓存中检索（不依赖 RDB 连接）"""
        if not self._tags_cache:
            print("[警告] 无 RDB 连接且无本地缓存，返回空结果")
            return []

        results = []
        for doc_id, tags in self._tags_cache.items():
            matched = True
            for dim, vals in constraints.items():
                col = _normalize_col_name(dim)
                doc_vals = tags.get(dim, tags.get(col, []))
                if isinstance(doc_vals, str):
                    doc_vals = [v.strip() for v in doc_vals.split(",")]
                if not any(v in doc_vals for v in vals if v):
                    matched = False
                    break
            if matched:
                results.append({
                    "doc_id": doc_id,
                    "tags": tags,
                    "match_type": "local_cache"
                })
            if len(results) >= top_k:
                break

        return results


# ===================== 辅助函数 =====================

def _connect_rdb():
    """连接到 MySQL RDB"""
    try:
        config = get_config()
        conn = mysql.connector.connect(
            host=config.rdb.host,
            port=config.rdb.port,
            user=config.rdb.user,
            password=config.rdb.password,
            database=config.rdb.database,
            charset="utf8mb4",
            collation="utf8mb4_unicode_ci",
            autocommit=False
        )
        return conn
    except Exception as e:
        print(f"[错误] RDB 连接失败: {e}")
        return None


def _fetch_rdb_columns(conn, database: str, table: str) -> List[str]:
    """获取 RDB 表列名"""
    cur = conn.cursor()
    cur.execute(
        "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
        "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s",
        (database, table)
    )
    cols = [r[0] for r in cur.fetchall()]
    cur.close()
    return cols


# ===================== 主程序入口 =====================

def main():
    parser_cli = argparse.ArgumentParser(
        description="Dimension-Aware Retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python dimension_retriever.py -q "儿童发烧咳嗽"
  python dimension_retriever.py -q "老年人腰腿痛" --mode hybrid --top_k 5
  python dimension_retriever.py -i
        """
    )
    parser_cli.add_argument("-q", "--query", type=str, help="查询文本")
    parser_cli.add_argument("--qid", type=str, default="q_001", help="查询 ID")
    parser_cli.add_argument("--mode", choices=["pure_filter", "hybrid"], default="hybrid",
                           help="检索模式: pure_filter=纯标签过滤, hybrid=标签+向量混合")
    parser_cli.add_argument("--top_k", type=int, default=10, help="返回结果数")
    parser_cli.add_argument("-i", "--interactive", action="store_true", help="交互模式")

    args = parser_cli.parse_args()

    retriever = DimensionRetriever()

    # 检查维度是否就绪
    if not retriever._v_core:
        print("[错误] 未检测到核心维度，请先运行 dimension_integration.py --step 1")
        return

    print("=" * 60)
    print(f"维度检索器就绪，共 {len(retriever._v_core)} 个维度")
    print(f"本地缓存: {len(retriever._tags_cache)} 篇文档")
    print("=" * 60)

    def handle_query(query_text: str, qid: str):
        print(f"\n{'='*60}")
        print(f"查询: {query_text}")
        print("=" * 60)

        # Step 1: 解析 query → 维度约束
        print("\n[Step 1] 解析查询意图...")
        constraints = retriever.parse_query(query_text, qid)
        print(f"  维度约束: {constraints}")

        if not constraints:
            print("  [警告] 未解析出维度约束，退化为纯向量检索")
            results = retriever._pure_vector_search(query_text, args.top_k)
        elif args.mode == "pure_filter":
            # Step 2: 纯过滤检索
            print("\n[Step 2] 执行纯维度过滤检索...")
            results = retriever.retrieve_by_constraints(
                constraints, mode="pure_filter", top_k=args.top_k
            )
        else:
            # Step 2: 混合检索
            print("\n[Step 2] 执行混合检索（维度粗筛 + 向量精排）...")
            results = retriever.retrieve_hybrid(
                query_text=query_text,
                constraints=constraints,
                top_k=args.top_k
            )

        # Step 3: 打印结果
        print(f"\n[Step 3] 检索结果 ({len(results)} 条):")
        for i, r in enumerate(results, 1):
            doc_id = r.get("doc_id", "")
            chunk_text = r.get("chunk_text", r.get("doc_text", ""))
            score = r.get("score", 0)
            tags = r.get("tags", r)

            # 过滤掉非标签字段用于显示
            display_tags = {k: v for k, v in tags.items()
                           if k.startswith("dim_") or k in retriever._v_core}

            print(f"\n  --- 结果 {i} ---")
            print(f"  doc_id: {doc_id}")
            if score:
                print(f"  相似度分数: {score:.4f}")
            if display_tags:
                print(f"  匹配标签: {json.dumps(display_tags, ensure_ascii=False)}")
            if chunk_text:
                preview = chunk_text[:150].replace("\n", " ")
                print(f"  文本预览: {preview}...")

    if args.interactive or not args.query:
        print("\n交互模式（输入 quit 退出）:")
        qid_counter = 1
        while True:
            q = input("\n请输入查询: ").strip()
            if q.lower() in ("quit", "exit", "q"):
                break
            if not q:
                continue
            handle_query(q, f"q_{qid_counter:03d}")
            qid_counter += 1
    else:
        handle_query(args.query, args.qid)


if __name__ == "__main__":
    main()
