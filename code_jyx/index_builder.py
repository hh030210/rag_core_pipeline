import os
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

try:
    import numpy as np
except ImportError:
    np = None
try:
    from FlagEmbedding import BGEM3FlagModel
except ImportError:
    BGEM3FlagModel = None
try:
    from .schema_v2 import (
        SCHEMA_VERSION,
        canonicalize_tag_details,
        dimension_path,
        load_schema,
        normalize_label,
        normalize_text,
        schema_maps,
    )
except ImportError:
    from schema_v2 import (
        SCHEMA_VERSION,
        canonicalize_tag_details,
        dimension_path,
        load_schema,
        normalize_label,
        normalize_text,
        schema_maps,
    )

class IndexConfig:
    DATA_DIR = "./experiment_data"
    PATH_TAGS_OPT = os.path.join(DATA_DIR, "tags_debug_top10.json")
    
    # 输出文件
    PATH_INVERTED_INDEX = os.path.join(DATA_DIR, "inverted_index_med_D.json") # 标签->文档索引
    PATH_DIM_META = os.path.join(DATA_DIR, "dimension_metadata_med_D.json")   # 维度的元数据(是否枚举, 值域列表)
    PATH_TAG_VECTORS = os.path.join(DATA_DIR, "tag_vectors_med_D.pkl")        # 开放维度的向量索引
    SCHEMA_VERSION = SCHEMA_VERSION
    PATH_TAGS_V2 = os.path.join(DATA_DIR, "tags_output_v2.json")
    PATH_INVERTED_INDEX_V2 = os.path.join(DATA_DIR, "inverted_index_v2.json")
    PATH_DIM_META_V2 = os.path.join(DATA_DIR, "dimension_metadata_v2.json")
    PATH_HIERARCHY_V2 = os.path.join(DATA_DIR, "dimension_hierarchy_v2.json")
    
    # 判定阈值：如果某维度的唯一值数量 <= 50，视为“伪枚举”
    ENUM_THRESHOLD = 0 #现在不考虑枚举类型，都按描述算
    
    # 原始定义的枚举类 (强制枚举)
    FORCE_ENUMS = []


def _v2_documents(tag_output):
    if not isinstance(tag_output, Mapping):
        return []
    if isinstance(tag_output.get("documents"), Mapping):
        return list(tag_output["documents"].items())
    return [(key, value) for key, value in tag_output.items() if key != "schema_version"]


def build_qdrant_payload_v2(doc_id, document_tags, schema, *, include_empty=True):
    """Build an array-valued Qdrant payload for one document/chunk."""

    schema = load_schema(schema, allow_legacy=False, strict=False)
    maps = schema_maps(schema)
    raw_tags = document_tags.get("tags", document_tags) if isinstance(document_tags, Mapping) else {}
    legacy_input = not (isinstance(document_tags, Mapping) and document_tags.get("schema_version") == SCHEMA_VERSION)
    tags, details = canonicalize_tag_details(raw_tags, schema, legacy=legacy_input)
    # 标签值和 tag_details 是两个互补视图：前者用于 Qdrant keyword array
    # 过滤，后者保存原始标签、证据和置信度。不能只根据 tags 重新构造，
    # 否则已有 evidence/raw_label 会静默退化为空字符串。
    if isinstance(document_tags, Mapping) and isinstance(document_tags.get("tag_details"), Mapping):
        _provided_tags, provided_details = canonicalize_tag_details(
            document_tags.get("tag_details", {}), schema, legacy=legacy_input
        )
        for dimension_id, items in provided_details.items():
            if items:
                details[dimension_id] = items
    payload = {"doc_id": str(doc_id), "schema_version": SCHEMA_VERSION}
    paths = document_tags.get("dimension_paths", []) if isinstance(document_tags, Mapping) else []
    if not isinstance(paths, list):
        paths = [paths] if paths else []
    payload["dimension_paths"] = [normalize_text(path) for path in paths if normalize_text(path)]
    for leaf_id in maps["leaves"]:
        labels = list(tags.get(leaf_id, []))
        if labels or include_empty:
            # This is intentionally a list.  Never use '; '.join here: Qdrant
            # keyword-array filters depend on preserving array semantics.
            payload[f"dim_{leaf_id}"] = labels
    payload["tag_details"] = details
    return payload


MYSQL_V2_DDL = (
    "CREATE TABLE IF NOT EXISTS dimension_schema ("
    "schema_version VARCHAR(16) NOT NULL, dimension_id VARCHAR(128) NOT NULL, "
    "name VARCHAR(255) NOT NULL, parent_id VARCHAR(128) NULL, level TINYINT NOT NULL, "
    "path VARCHAR(1024) NOT NULL, description TEXT, indexable BOOLEAN NOT NULL, "
    "PRIMARY KEY (schema_version, dimension_id), "
    "INDEX idx_dimension_schema_parent (schema_version, parent_id)"
    ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
    "CREATE TABLE IF NOT EXISTS document_dimension_tag ("
    "doc_id VARCHAR(255) NOT NULL, dimension_id VARCHAR(128) NOT NULL, "
    "label VARCHAR(512) NOT NULL, normalized_label VARCHAR(512) NOT NULL, "
    "evidence TEXT, confidence DOUBLE NULL, schema_version VARCHAR(16) NOT NULL, "
    "UNIQUE KEY uq_document_dimension_label (doc_id, dimension_id, normalized_label), "
    "INDEX idx_dimension_label_doc (dimension_id, normalized_label, doc_id), "
    "INDEX idx_document_schema (doc_id, schema_version)"
    ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
)


def mysql_v2_schema_rows(schema):
    """Produce parameterized rows for the two new tables; no DB write occurs."""

    schema = load_schema(schema, allow_legacy=False, strict=False)
    maps = schema_maps(schema)
    schema_rows = []
    for dimension_id, node in maps["nodes"].items():
        schema_rows.append((
            SCHEMA_VERSION,
            dimension_id,
            node["name"],
            node.get("parent_id"),
            int(node["level"]),
            dimension_path(schema, dimension_id),
            node.get("description", ""),
            bool(node.get("indexable")),
        ))
    return schema_rows


def build_index_v2(tag_output, schema):
    """Build postings and hierarchy indexes from array-valued document tags."""

    schema = load_schema(schema, allow_legacy=False, strict=False)
    maps = schema_maps(schema)
    postings: Dict[str, Dict[str, List[str]]] = {leaf_id: {} for leaf_id in maps["leaves"]}
    label_sets = {leaf_id: set() for leaf_id in maps["leaves"]}
    doc_count = 0
    multi_label_docs = {leaf_id: 0 for leaf_id in maps["leaves"]}
    tag_details_by_doc = {}
    for raw_doc_id, document in _v2_documents(tag_output):
        doc_id = str(raw_doc_id)
        doc_count += 1
        raw_tags = document.get("tags", document) if isinstance(document, Mapping) else {}
        legacy_input = tag_output.get("schema_version") != SCHEMA_VERSION if isinstance(tag_output, Mapping) else True
        tags, details = canonicalize_tag_details(raw_tags, schema, legacy=legacy_input)
        if isinstance(document, Mapping) and isinstance(document.get("tag_details"), Mapping):
            _provided_tags, provided_details = canonicalize_tag_details(
                document.get("tag_details", {}), schema, legacy=legacy_input
            )
            for dimension_id, items in provided_details.items():
                if items:
                    details[dimension_id] = items
        tag_details_by_doc[doc_id] = details
        for leaf_id in maps["leaves"]:
            labels = tags.get(leaf_id, [])
            if len(labels) > 1:
                multi_label_docs[leaf_id] += 1
            seen_in_doc = set()
            for label in labels:
                normalized = normalize_label(label)
                if not normalized or normalized in seen_in_doc:
                    continue
                seen_in_doc.add(normalized)
                label_sets[leaf_id].add(normalized)
                postings[leaf_id].setdefault(normalized, [])
                if doc_id not in postings[leaf_id][normalized]:
                    postings[leaf_id][normalized].append(doc_id)

    metadata = {}
    for dimension_id, node in maps["nodes"].items():
        metadata[dimension_id] = {
            "id": dimension_id,
            "name": node["name"],
            "parent_id": node.get("parent_id"),
            "level": node["level"],
            "path": dimension_path(schema, dimension_id),
            "description": node.get("description", ""),
            "indexable": bool(node.get("indexable")),
            "value_policy": "multi",
            "max_labels": int(node.get("max_labels", 5)),
            "unique_label_count": len(label_sets.get(dimension_id, set())),
            "multi_label_document_count": multi_label_docs.get(dimension_id, 0),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "postings": postings,
        "hierarchy": {
            "children": maps["children"],
            "ancestors": maps["ancestors"],
        },
        "dimension_metadata": metadata,
        "tag_details": tag_details_by_doc,
        "stats": {
            "document_count": doc_count,
            "leaf_count": len(maps["leaves"]),
            "tag_count": sum(len(values) for values in postings.values()),
            "multi_label_documents": multi_label_docs,
        },
    }


class IndexBuilderV2:
    """Persistent v2 index builder.  Existing legacy paths are untouched."""

    def __init__(self, schema, output_dir="./experiment_data"):
        self.schema = load_schema(schema, allow_legacy=False, strict=False)
        self.output_dir = Path(output_dir)

    def build(self, tag_output):
        return build_index_v2(tag_output, self.schema)

    def save(self, index, output_dir=None):
        output_dir = Path(output_dir or self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        files = {
            "inverted_index": output_dir / "inverted_index_v2.json",
            "dimension_metadata": output_dir / "dimension_metadata_v2.json",
            "hierarchy": output_dir / "dimension_hierarchy_v2.json",
        }
        for key, path in files.items():
            value = index["postings"] if key == "inverted_index" else index[key]
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        return {key: str(path) for key, path in files.items()}

    def dry_run_stats(self, tag_output):
        index = self.build(tag_output)
        stats = index["stats"]
        stats["array_payload_verified"] = True
        stats["schema_version"] = SCHEMA_VERSION
        return stats

class IndexBuilder:
    def __init__(self):
        print("初始化索引构建器...")
        self.encoder = None # 懒加载
        
    def build(self):
        # 1. 加载优化后的标签数据
        # 格式: {doc_id: {dim: [val1, val2]}}
        with open(IndexConfig.PATH_TAGS_OPT, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)
            
        print(f"加载了 {len(doc_data)} 条文档标签。")
        
        # 2. 构建倒排索引 & 统计值域
        # inverted_index: {dim: {val: [doc_id1, doc_id2]}}
        inverted_index = defaultdict(lambda: defaultdict(list))
        dim_value_sets = defaultdict(set)
        
        for doc_id, tags in doc_data.items():
            for dim, vals in tags.items():
                if not vals: continue
                # 兼容处理
                if isinstance(vals, str): vals = [vals]
                
                for v in vals:
                    inverted_index[dim][v].append(doc_id)
                    dim_value_sets[dim].add(v)
                    
        # 3. 维度元数据分析 (区分 Enum vs Open)
        dim_meta = {}
        dims_to_vectorize = [] # 需要做向量索引的维度
        
        print("\n=== 维度属性分析 ===")
        for dim, val_set in dim_value_sets.items():
            val_list = sorted(list(set(val_set)))
            count = len(val_list)
            
            # 判断逻辑：强制枚举 OR 值域较窄
            is_enum = (dim in IndexConfig.FORCE_ENUMS) or (count <= IndexConfig.ENUM_THRESHOLD)
            
            dim_meta[dim] = {
                "is_enum": is_enum,
                "value_count": count,
                "values": val_list if is_enum else [] # 如果是枚举，直接存值域
            }
            
            tag_type = "枚举 (Enum)" if is_enum else "开放 (Open)"
            print(f"维度 [{dim}]: {count} 个值 -> {tag_type}")
            
            if not is_enum:
                dims_to_vectorize.append((dim, val_list))
                
        # 4. 构建开放维度的向量索引 (用于后续 Matcher)
        if dims_to_vectorize:
            print("\n正在构建开放维度的向量索引...")
            current_dir = os.path.dirname(__file__)
            embedding_model_path = os.path.join(current_dir, 'bge-m3')
            self.encoder = BGEM3FlagModel(embedding_model_path, use_fp16=True, device='cuda')
            
            tag_vectors = {} # {dim: {'vals': [], 'vecs': np.array}}
            
            for dim, val_list in dims_to_vectorize:
                print(f"  Encoding {dim} ({len(val_list)} values)...")
                embeddings = self.encoder.encode(val_list, return_dense=True)['dense_vecs']
                tag_vectors[dim] = {
                    'values': val_list,
                    'vectors': embeddings
                }
            
            import pickle
            with open(IndexConfig.PATH_TAG_VECTORS, 'wb') as f:
                pickle.dump(tag_vectors, f)
        
        # 5. 保存结果
        with open(IndexConfig.PATH_INVERTED_INDEX, 'w', encoding='utf-8') as f:
            json.dump(inverted_index, f, ensure_ascii=False)
            
        with open(IndexConfig.PATH_DIM_META, 'w', encoding='utf-8') as f:
            json.dump(dim_meta, f, ensure_ascii=False, indent=2)
            
        print("\n索引构建完成！")

if __name__ == "__main__":
    builder = IndexBuilder()
    builder.build()
