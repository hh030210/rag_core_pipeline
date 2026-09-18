"""v2 层次化维度发现、叶子多标签抽取和倒排索引。"""

from __future__ import annotations

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .schema_v2 import (
    SCHEMA_VERSION,
    canonicalize_tag_details,
    dimension_path,
    load_schema,
    make_dimension_node,
    schema_maps,
)


def default_schema() -> Dict[str, Any]:
    groups = [
        ("content", "内容属性", "对象、位置和历史文化内容。", [
            ("entity", "涉及对象", "文本明确提到的人物、人群或对象。"),
            ("location", "地理位置", "文本明确提到的地名、区域和景区位置。"),
            ("history", "历史沿革", "文本明确提到的年代、沿革、事件和由来。"),
        ]),
        ("landscape", "景观属性", "景观构成、特色和游览内容。", [
            ("composition", "景观组成", "建筑、景点、山水和空间组成。"),
            ("feature", "景观特色", "景观特点、规模、数量和观赏价值。"),
            ("culture", "文化内涵", "文化意义、宗教内涵、艺术和典故。"),
        ]),
        ("operation", "运营服务", "面向游客的时间、票务和服务信息。", [
            ("opening", "开放时间", "开门、闭园、营业和开放时段。"),
            ("ticketing", "票务规则", "门票、票价、预约、优惠和购票规则。"),
            ("service", "游客服务", "厕所、餐饮、讲解、寄存和其他服务设施。"),
        ]),
        ("transport", "交通游览", "到达路线和景区内部游览。", [
            ("route", "交通方式", "公交、地铁、自驾和到达路线。"),
            ("facility", "交通设施", "停车场、接驳车、索道和游船。"),
        ]),
        ("season", "时令游览", "季节、天气和游览时机。", [
            ("seasonal", "最佳时节", "季节、天气和适合游览的时间。"),
            ("activity", "游览活动", "节庆、活动和互动体验。"),
        ]),
    ]
    nodes = []
    for parent_id, parent_name, parent_desc, leaves in groups:
        nodes.append(make_dimension_node(name=parent_name, parent_id=None, level=1,
                                         indexable=False, description=parent_desc,
                                         dimension_id=parent_id))
        for leaf_id, leaf_name, description in leaves:
            nodes.append(make_dimension_node(name=leaf_name, parent_id=parent_id,
                                             level=2, indexable=True,
                                             description=description,
                                             dimension_id=f"{parent_id}.{leaf_id}"))
    return {"schema_version": SCHEMA_VERSION, "dimensions": nodes,
            "generation_method": "validated_fallback"}


def _records_to_docs(records: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [{"doc_id": str(item["doc_id"]), "doc_text": str(item.get("doc_text", ""))}
            for item in records]


def build_tags(records, schema, settings, miner=None) -> Dict[str, Any]:
    leaves = [node for node in schema["dimensions"]
              if node.get("level") == 2 and node.get("indexable") is True]
    documents: Dict[str, Any] = {}

    try:
        workers = max(1, int(os.getenv("LLM_CONCURRENCY", "1")))
    except ValueError:
        workers = 1

    def extract_one(record: Dict[str, Any], worker_state: threading.local | None = None):
        if settings.mock:
            extracted = _mock_extract(str(record.get("doc_text", "")), leaves)
        else:
            active_miner = miner
            if worker_state is not None:
                active_miner = getattr(worker_state, "miner", None)
                if active_miner is None:
                    from .llm_service import DimensionMiningWithQwen
                    active_miner = DimensionMiningWithQwen(
                        api_key=settings.llm_api_key,
                        model_name=settings.llm_model,
                        base_url=settings.llm_base_url,
                    )
                    worker_state.miner = active_miner
            extracted = active_miner.extract_batch_dimensions_v2(
                str(record.get("doc_text", "")), leaves,
                max_dimensions_per_request=settings.max_dimensions_per_request,
                max_text_chars=settings.tag_text_chars,
            )
        return extracted

    def consume(index: int, record: Dict[str, Any], extracted: Dict[str, Any]) -> None:
        doc_id = str(record["doc_id"])
        tags, details = canonicalize_tag_details(extracted or {}, schema)
        maps = schema_maps(schema)
        paths = sorted({dimension_path(schema, leaf_id) for leaf_id in tags if leaf_id in maps["leaves"]})
        documents[doc_id] = {
            "tags": tags,
            "tag_details": details,
            "dimension_paths": paths,
            "schema_version": SCHEMA_VERSION,
        }

    if workers > 1 and not settings.mock and records:
        print(f"[维度抽取] 启用受控并发: {workers} workers")
        worker_state = threading.local()
        completed = 0
        results: Dict[int, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(extract_one, record, worker_state): (index, record)
                for index, record in enumerate(records, 1)
            }
            for future in as_completed(futures):
                index, record = futures[future]
                results[index] = future.result()
                completed += 1
                if completed % 10 == 0 or completed == len(records):
                    print(f"[维度抽取] {completed}/{len(records)}")
        for index, record in enumerate(records, 1):
            consume(index, record, results[index])
    else:
        for index, record in enumerate(records, 1):
            consume(index, record, extract_one(record))
            if index % 50 == 0:
                print(f"[维度抽取] {index}/{len(records)}")
    return {"schema_version": SCHEMA_VERSION, "documents": documents}


def _mock_extract(text: str, leaves: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    rules = {
        "location": ["北京", "杭州", "衢州", "河南", "浙江", "西湖", "颐和园", "少林寺"],
        "opening": ["开放", "营业", "开门", "上午", "下午", "每日"],
        "ticketing": ["门票", "票价", "免费", "预约", "购票", "元"],
        "entity": ["儿童", "青少年", "成人", "游客", "孔子"],
        "history": ["历史", "始建", "修建", "朝代", "年代"],
        "feature": ["特色", "景观", "面积", "海拔", "高度"],
        "route": ["公交", "地铁", "自驾", "路线", "交通"],
        "seasonal": ["春季", "夏季", "秋季", "冬季", "季节"],
    }
    result = {}
    for leaf in leaves:
        key = leaf["id"].split(".")[-1]
        values = []
        for token in rules.get(key, []):
            if token in text:
                values.append({"label": token, "raw_label": token,
                               "evidence": token, "confidence": 1.0})
        if values:
            result[leaf["id"]] = values
    return result


def build_inverted_index(tag_output: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    maps = schema_maps(schema)
    postings = {leaf_id: {} for leaf_id in maps["leaves"]}
    documents = tag_output.get("documents", {})
    for doc_id, document in documents.items():
        tags = document.get("tags", {}) if isinstance(document, dict) else {}
        for leaf_id in maps["leaves"]:
            values = tags.get(leaf_id, [])
            if not isinstance(values, list):
                values = [values]
            for label in values:
                normalized = str(label).strip()
                if not normalized:
                    continue
                postings[leaf_id].setdefault(normalized, [])
                if doc_id not in postings[leaf_id][normalized]:
                    postings[leaf_id][normalized].append(str(doc_id))
    metadata = {}
    for node_id, node in maps["nodes"].items():
        labels = postings.get(node_id, {})
        metadata[node_id] = {
            **node,
            "path": dimension_path(schema, node_id),
            "unique_label_count": len(labels),
            "values": sorted(labels),
        }
    return {"schema_version": SCHEMA_VERSION, "postings": postings,
            "hierarchy": {"children": maps["children"], "ancestors": maps["ancestors"]},
            "dimension_metadata": metadata}


def run_dimensions(records, settings, schema_path: str = "") -> Dict[str, Any]:
    settings.ensure_dirs()
    if schema_path:
        schema = load_schema(schema_path, allow_legacy=True, strict=False)
    elif settings.mock or not settings.llm_api_key:
        schema = default_schema()
    else:
        from .llm_service import DimensionMiningWithQwen
        miner_for_schema = DimensionMiningWithQwen(
            api_key=settings.llm_api_key, model_name=settings.llm_model,
            base_url=settings.llm_base_url)
        schema = miner_for_schema.generate_candidate_schema(
            [str(item.get("doc_text", "")) for item in records]
        )
        schema = load_schema(schema, allow_legacy=False, strict=True)

    (settings.run_dir / "V_cand_v2.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    (settings.run_dir / "V_core_v2.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

    miner = None
    if not settings.mock:
        from .llm_service import DimensionMiningWithQwen
        miner = DimensionMiningWithQwen(
            api_key=settings.llm_api_key, model_name=settings.llm_model,
            base_url=settings.llm_base_url)
    tag_output = build_tags(records, schema, settings, miner=miner)
    tags_path = settings.run_dir / "tags_output_v2.json"
    tags_path.write_text(json.dumps(tag_output, ensure_ascii=False, indent=2), encoding="utf-8")
    index = build_inverted_index(tag_output, schema)
    (settings.run_dir / "inverted_index_v2.json").write_text(
        json.dumps(index["postings"], ensure_ascii=False, indent=2), encoding="utf-8")
    (settings.run_dir / "dimension_metadata_v2.json").write_text(
        json.dumps(index["dimension_metadata"], ensure_ascii=False, indent=2), encoding="utf-8")
    (settings.run_dir / "dimension_hierarchy_v2.json").write_text(
        json.dumps(index["hierarchy"], ensure_ascii=False, indent=2), encoding="utf-8")
    return {"schema": schema, "tags": tag_output, "index": index}
