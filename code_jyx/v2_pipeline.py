"""Small, non-destructive v2 pipeline adapters and dry-run utilities."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .dimension_generate import V2_THRESHOLDS, diagnose_leaf_dimensions
from .index_builder import IndexBuilderV2, build_qdrant_payload_v2
from .schema_v2 import SCHEMA_VERSION, load_schema, make_dimension_node, normalize_label
from .tag_generate import TagGeneratorV2


def load_records(source: Optional[str], *, limit: int = 50) -> List[Dict[str, str]]:
    """Load up to ``limit`` text records for a dry-run without touching a DB."""

    if not source:
        return []
    path = Path(source)
    records: List[Dict[str, str]] = []
    if path.is_dir():
        for file_path in sorted(path.rglob("*.md")):
            records.append({"doc_id": file_path.stem, "doc_text": file_path.read_text(encoding="utf-8", errors="ignore")})
            if len(records) >= limit:
                break
        return records
    if not path.exists():
        return []
    if path.suffix.lower() == ".jsonl":
        raw_items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            raw_items = raw
        elif isinstance(raw, dict) and isinstance(raw.get("documents"), list):
            raw_items = raw["documents"]
        elif isinstance(raw, dict) and isinstance(raw.get("chunks"), list):
            raw_items = raw["chunks"]
        elif isinstance(raw, dict):
            raw_items = [{"doc_id": key, **value} if isinstance(value, dict) else {"doc_id": key, "doc_text": value} for key, value in raw.items()]
        else:
            raw_items = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("doc_id", item.get("id", item.get("chunk_id", index))))
        text = str(item.get("doc_text", item.get("text", item.get("content", ""))) or "")
        if text:
            records.append({"doc_id": doc_id, "doc_text": text})
        if len(records) >= limit:
            break
    return records


def make_mock_schema():
    """Deterministic two-level schema used by tests and local dry-runs."""

    content = make_dimension_node(
        name="内容属性", parent_id=None, level=1, indexable=False,
        description="文本描述的对象、地点和历史信息分组。", dimension_id="content",
    )
    service = make_dimension_node(
        name="服务属性", parent_id=None, level=1, indexable=False,
        description="面向游客的时间、票务和交通信息分组。", dimension_id="service",
    )
    nodes = [content, service]
    nodes.extend([
        make_dimension_node(name="涉及对象", parent_id="content", level=2, description="文本中明确出现的人群或对象。", dimension_id="content.entity"),
        make_dimension_node(name="地理位置", parent_id="content", level=2, description="文本中明确出现的地点、区域或景区位置。", dimension_id="content.location"),
        make_dimension_node(name="开放时间", parent_id="service", level=2, description="文本中明确出现的开放、营业或参观时间。", dimension_id="service.opening_hours"),
        make_dimension_node(name="票务信息", parent_id="service", level=2, description="文本中明确出现的票价、门票或购票信息。", dimension_id="service.ticketing"),
    ])
    return {"schema_version": SCHEMA_VERSION, "dimensions": nodes}


class MockMiner:
    """No-network multi-label extractor for dry-run and end-to-end tests."""

    PATTERNS = {
        "content.entity": ["儿童", "青少年", "成人", "游客", "孔子"],
        "content.location": ["北京", "杭州", "衢州", "河南", "浙江", "颐和园", "西湖", "少林寺"],
        "service.opening_hours": ["开放", "营业", "参观", "每日", "上午", "下午"],
        "service.ticketing": ["门票", "票价", "免费", "购票", "元"],
    }

    def extract_batch_dimensions_v2(self, text, dimensions, **_kwargs):
        text = str(text or "")
        output = {}
        for node in dimensions:
            node_id = str(node["id"])
            values = []
            for token in self.PATTERNS.get(node_id, []):
                if token in text:
                    values.append({"label": token, "raw_label": token, "evidence": token, "confidence": 1.0})
            if values:
                output[node_id] = values
        return output


def _tag_counts(tag_output):
    raw_count = 0
    normalized_count = 0
    for _doc_id, document in (tag_output.get("documents", {}) or {}).items():
        for values in (document.get("tags", {}) or {}).values():
            raw_values = values if isinstance(values, list) else [values]
            raw_count += len(raw_values)
            normalized_count += len({normalize_label(value) for value in raw_values if normalize_label(value)})
    return raw_count, normalized_count


def run_dry_run(
    *,
    source: Optional[str] = None,
    schema_path: Optional[str] = None,
    tags_path: Optional[str] = None,
    output_dir: str = "./experiment_data/dimension_v2_dry_run",
    limit: int = 30,
    mock: bool = True,
) -> Dict[str, Any]:
    """Run a 20–50 record non-destructive v2 migration/index dry-run."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if schema_path:
        raw_schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        schema = load_schema(raw_schema, allow_legacy=True, strict=False)
    else:
        schema = make_mock_schema()
    schema_file = output / "V_core_v2.json"
    schema_file.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "V_cand_v2.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

    if tags_path:
        tag_output = json.loads(Path(tags_path).read_text(encoding="utf-8"))
    else:
        records = load_records(source, limit=max(20, min(50, int(limit))))
        if not records:
            records = [
                {"doc_id": f"mock-{i}", "doc_text": "儿童和青少年可参观北京景点，每日上午开放，门票免费。"}
                for i in range(max(20, min(50, int(limit))))
            ]
        tag_output = TagGeneratorV2(schema, miner=MockMiner()).run(records)
    tags_file = output / "tags_output_v2.json"
    tags_file.write_text(json.dumps(tag_output, ensure_ascii=False, indent=2), encoding="utf-8")

    builder = IndexBuilderV2(schema, output_dir=output)
    index = builder.build(tag_output)
    index_files = builder.save(index, output)
    diagnostics = diagnose_leaf_dimensions(tag_output, schema, thresholds=V2_THRESHOLDS)
    raw_count, normalized_count = _tag_counts(tag_output)
    docs = len(tag_output.get("documents", {}))
    leaf_stats = {}
    for dimension_id, stat in diagnostics["dimensions"].items():
        if stat.get("indexable"):
            multi = stat.get("average_labels_per_document", 0.0) > 1.0
            multi_docs = index["stats"]["multi_label_documents"].get(dimension_id, 0)
            leaf_stats[dimension_id] = {
                **stat,
                "multi_label_documents": multi_docs,
                "multi_label_document_ratio": round(multi_docs / docs, 6) if docs else 0.0,
            }
    report = {
        "schema_version": SCHEMA_VERSION,
        "dry_run": True,
        "mock_llm": bool(mock and not tags_path),
        "document_count": docs,
        "tag_count_before_dedup": raw_count,
        "tag_count_after_dedup": normalized_count,
        "duplicate_labels_removed": max(0, raw_count - normalized_count),
        "leaf_dimensions": leaf_stats,
        "index_stats": index["stats"],
        "files": {
            "V_cand_v2": str(output / "V_cand_v2.json"),
            "V_core_v2": str(schema_file),
            "tags_output_v2": str(tags_file),
            **index_files,
        },
    }
    (output / "dry_run_report_v2.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
