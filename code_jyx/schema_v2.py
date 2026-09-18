"""Pure helpers for the versioned hierarchical dimension schema.

This module deliberately has no database, vector-model or LLM dependency so
that schema validation, migration and index semantics can be tested locally.
"""

from __future__ import annotations

import hashlib
import json
import re
import string
import unicodedata
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "2.0"
LEVEL_ONE = 1
LEVEL_TWO = 2
_NULL_VALUES = {"", "null", "none", "NULL", "NONE", "无", "未提及", "未提供"}
_OUTER_PUNCTUATION = string.punctuation + "，。！？；：、（）【】《》“”‘’「」『』"
_STABLE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")


class SchemaValidationError(ValueError):
    """Raised when a v2 schema violates a structural invariant."""

    def __init__(self, errors: Sequence[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def normalize_text(value: Any) -> str:
    """Normalize Unicode and whitespace without changing semantic content."""

    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\u00a0", " ").replace("\u3000", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_label(value: Any) -> str:
    """Return the canonical key used by postings and query matching.

    Punctuation is only removed at the outside of a label.  Internal
    punctuation is retained because it can carry meaning in domain terms.
    """

    text = normalize_text(value)
    text = text.strip(_OUTER_PUNCTUATION + " \t\r\n")
    return re.sub(r"\s+", " ", text)


def as_label_list(value: Any, *, legacy: bool = False) -> List[str]:
    """Convert old string/list values to an explicit list.

    A semicolon is split only when ``legacy=True``.  This prevents a genuine
    modern label containing punctuation from being silently reinterpreted.
    """

    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw_values: Iterable[Any] = value
    elif isinstance(value, str):
        raw_values = value.split(";") if legacy and ";" in value else [value]
    else:
        raw_values = [value]

    result: List[str] = []
    seen = set()
    for item in raw_values:
        label = normalize_text(item)
        if not label or label in _NULL_VALUES or label.upper() in {"NULL", "NONE"}:
            continue
        if label not in seen:
            result.append(label)
            seen.add(label)
    return result


def _ascii_slug(text: str) -> str:
    try:
        from pypinyin import lazy_pinyin

        candidate = "_".join(lazy_pinyin(text))
    except Exception:
        candidate = ""
    candidate = re.sub(r"[^a-zA-Z0-9]+", "_", candidate).strip("_").lower()
    return candidate


def stable_dimension_id(name: Any, parent_id: Optional[str] = None) -> str:
    """Create a stable non-Chinese database key from semantic identity.

    When pypinyin is available the readable slug is used; otherwise a stable
    digest is used instead of putting mutable Chinese names into a primary
    key.  The parent is part of the identity for a leaf node.
    """

    normalized_name = normalize_text(name)
    identity = f"{normalize_text(parent_id)}::{normalized_name}" if parent_id else normalized_name
    readable = _ascii_slug(identity)
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:10]
    if readable:
        return f"dim_{readable[:42]}_{digest}"
    return f"dim_{digest}"


def is_stable_dimension_id(value: Any) -> bool:
    """Return whether an id is a lowercase English/pinyin-style slug."""

    return bool(_STABLE_ID_PATTERN.fullmatch(normalize_text(value)))


def make_dimension_node(
    *,
    name: str,
    parent_id: Optional[str],
    level: int,
    description: str = "",
    indexable: Optional[bool] = None,
    aliases: Optional[Sequence[str]] = None,
    max_labels: int = 5,
    dimension_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a normalized schema node."""

    if indexable is None:
        indexable = level == LEVEL_TWO
    node = {
        "id": dimension_id or stable_dimension_id(name, parent_id),
        "name": normalize_text(name),
        "parent_id": parent_id,
        "level": int(level),
        "indexable": bool(indexable),
        "description": normalize_text(description),
        "aliases": [normalize_text(v) for v in (aliases or []) if normalize_text(v)],
        "value_policy": "multi",
        "max_labels": max(1, int(max_labels)),
    }
    return node


def legacy_to_v2(value: Any) -> Dict[str, Any]:
    """Convert a legacy ``List[str]``/flat mapping to a v2 tree.

    Legacy dimensions do not contain enough evidence to infer meaningful
    groups.  They are therefore placed below one explicit compatibility
    parent.  New generation does not use this parent; it is only a safe
    loader for old artifacts.
    """

    if isinstance(value, Mapping):
        if value.get("schema_version") == SCHEMA_VERSION and isinstance(value.get("dimensions"), list):
            return deepcopy(dict(value))
        if isinstance(value.get("dimensions"), list):
            legacy_dimensions = value["dimensions"]
        else:
            legacy_dimensions = list(value.keys())
    elif isinstance(value, (list, tuple, set)):
        legacy_dimensions = list(value)
    else:
        raise TypeError("旧 Schema 必须是 List[str] 或包含 dimensions 的对象")

    parent = make_dimension_node(
        name="信息轴",
        parent_id=None,
        level=LEVEL_ONE,
        indexable=False,
        description="旧版平面维度的兼容分组，不代表新的语义分类。",
        dimension_id="group_information",
    )
    dimensions = [parent]
    seen = set()
    for raw in legacy_dimensions:
        name = normalize_text(raw)
        if not name or name in seen:
            continue
        seen.add(name)
        dimensions.append(
            make_dimension_node(
                name=name,
                parent_id=parent["id"],
                level=LEVEL_TWO,
                description=f"旧版维度“{name}”的兼容叶子节点。",
            )
        )
    return {"schema_version": SCHEMA_VERSION, "dimensions": dimensions}


def validate_schema(schema: Mapping[str, Any], *, strict: bool = True) -> List[str]:
    """Validate ids, levels, parent links, cycles and sibling uniqueness."""

    errors: List[str] = []
    if not isinstance(schema, Mapping):
        return ["schema 必须是对象"]
    if schema.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version 必须为 {SCHEMA_VERSION}")
    dimensions = schema.get("dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        errors.append("dimensions 必须是非空数组")
        return errors

    by_id: Dict[str, Mapping[str, Any]] = {}
    sibling_names: Dict[Optional[str], set] = {}
    for index, node in enumerate(dimensions):
        prefix = f"dimensions[{index}]"
        if not isinstance(node, Mapping):
            errors.append(f"{prefix} 必须是对象")
            continue
        node_id = normalize_text(node.get("id"))
        name = normalize_text(node.get("name"))
        parent_id = node.get("parent_id")
        parent_id = normalize_text(parent_id) or None
        level = node.get("level")
        if not node_id:
            errors.append(f"{prefix}.id 不能为空")
        elif not is_stable_dimension_id(node_id):
            errors.append(f"{prefix}.id 必须是稳定的小写英文/拼音 slug: {node_id}")
        elif node_id in by_id:
            errors.append(f"重复 dimension id: {node_id}")
        else:
            by_id[node_id] = node
        if not name:
            errors.append(f"{prefix}.name 不能为空")
        sibling_key = (parent_id, name.casefold())
        if sibling_key in sibling_names:
            errors.append(f"同级维度名称重复: {name}")
        sibling_names[sibling_key] = set()
        if level not in {LEVEL_ONE, LEVEL_TWO}:
            errors.append(f"{prefix}.level 必须为 1 或 2")
        if level == LEVEL_ONE and (parent_id is not None or node.get("indexable") is not False):
            errors.append(f"一级维度必须 parent_id=null 且 indexable=false: {node_id or prefix}")
        if level == LEVEL_TWO and (not parent_id or node.get("indexable") is not True):
            errors.append(f"二级叶子必须有父节点且 indexable=true: {node_id or prefix}")
        if node.get("value_policy", "multi") != "multi":
            errors.append(f"所有节点 value_policy 必须为 multi: {node_id or prefix}")
        try:
            if int(node.get("max_labels", 0)) < 1:
                errors.append(f"max_labels 必须为正数: {node_id or prefix}")
        except (TypeError, ValueError):
            errors.append(f"max_labels 非法: {node_id or prefix}")

    for node_id, node in by_id.items():
        parent_id = normalize_text(node.get("parent_id")) or None
        if parent_id and parent_id not in by_id:
            errors.append(f"parent_id 不存在: {node_id} -> {parent_id}")
        if parent_id and by_id.get(parent_id, {}).get("level") != LEVEL_ONE:
            errors.append(f"叶子父节点必须是一级维度: {node_id} -> {parent_id}")

    children: Dict[str, List[str]] = {}
    for node_id, node in by_id.items():
        parent_id = normalize_text(node.get("parent_id")) or None
        if parent_id:
            children.setdefault(parent_id, []).append(node_id)
    if strict:
        for node_id, node in by_id.items():
            if node.get("level") == LEVEL_ONE and len(children.get(node_id, [])) < 2:
                errors.append(f"一级维度至少需要两个语义独立叶子: {node_id}")

    # The current two-level representation cannot contain an indirect cycle;
    # still perform a generic parent walk so malformed input is diagnosed.
    for node_id in by_id:
        seen = set()
        current = node_id
        while current:
            if current in seen:
                errors.append(f"维度树存在环: {node_id}")
                break
            seen.add(current)
            parent = normalize_text(by_id.get(current, {}).get("parent_id")) or None
            current = parent
    return errors


def load_schema(value: Any, *, allow_legacy: bool = True, strict: bool = True) -> Dict[str, Any]:
    """Load a schema object or JSON path and validate it."""

    if isinstance(value, (str, Path)):
        path = Path(value)
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    if isinstance(value, Mapping) and value.get("schema_version") == SCHEMA_VERSION:
        schema = deepcopy(dict(value))
    elif allow_legacy:
        schema = legacy_to_v2(value)
        strict = False
    else:
        raise SchemaValidationError(["非 v2 Schema 且未启用兼容加载"])
    errors = validate_schema(schema, strict=strict)
    if errors:
        raise SchemaValidationError(errors)
    return schema


def schema_maps(schema: Mapping[str, Any]) -> Dict[str, Any]:
    """Return reusable id/parent/leaf/path maps."""

    schema = load_schema(schema, allow_legacy=False, strict=False)
    nodes = {str(node["id"]): dict(node) for node in schema["dimensions"]}
    children: Dict[str, List[str]] = {}
    ancestors: Dict[str, List[str]] = {}
    paths: Dict[str, List[str]] = {}
    for node_id, node in nodes.items():
        parent = normalize_text(node.get("parent_id")) or None
        if parent:
            children.setdefault(parent, []).append(node_id)
            ancestors[node_id] = [parent]
            paths[node_id] = [parent, node_id]
        else:
            ancestors[node_id] = []
            paths[node_id] = [node_id]
    leaves = [node_id for node_id, node in nodes.items() if node.get("level") == LEVEL_TWO and node.get("indexable") is True]
    return {"nodes": nodes, "children": children, "ancestors": ancestors, "paths": paths, "leaves": leaves}


def dimension_path(schema: Mapping[str, Any], dimension_id: str) -> str:
    maps = schema_maps(schema)
    names = [maps["nodes"][part]["name"] for part in maps["paths"].get(dimension_id, [dimension_id]) if part in maps["nodes"]]
    return " / ".join(names)


def make_v2_schema(nodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Normalize and validate a generated node list."""

    normalized = []
    for raw in nodes:
        node = dict(raw)
        level = int(node.get("level", 2))
        normalized.append(
            make_dimension_node(
                name=node.get("name", ""),
                parent_id=normalize_text(node.get("parent_id")) or None,
                level=level,
                description=node.get("description", ""),
                indexable=node.get("indexable", level == LEVEL_TWO),
                aliases=node.get("aliases", []),
                max_labels=node.get("max_labels", 5),
                dimension_id=normalize_text(node.get("id")) or None,
            )
        )
    schema = {"schema_version": SCHEMA_VERSION, "dimensions": normalized}
    errors = validate_schema(schema)
    if errors:
        raise SchemaValidationError(errors)
    return schema


def canonicalize_tag_details(
    tags: Mapping[str, Any],
    schema: Mapping[str, Any],
    *,
    legacy: bool = False,
) -> Tuple[Dict[str, List[str]], Dict[str, List[Dict[str, Any]]]]:
    """Normalize a document's tags while preserving raw labels/evidence."""

    maps = schema_maps(schema)
    allowed = set(maps["leaves"])
    key_to_leaf = {}
    for leaf_id in maps["leaves"]:
        node = maps["nodes"][leaf_id]
        key_to_leaf[leaf_id] = leaf_id
        key_to_leaf[normalize_text(node.get("name"))] = leaf_id
        for alias in node.get("aliases", []) or []:
            key_to_leaf[normalize_text(alias)] = leaf_id
    normalized_tags: Dict[str, List[str]] = {}
    details: Dict[str, List[Dict[str, Any]]] = {}
    for dimension_id, raw_values in (tags or {}).items():
        dimension_id = key_to_leaf.get(str(dimension_id), key_to_leaf.get(normalize_text(dimension_id), str(dimension_id)))
        if dimension_id not in allowed:
            continue
        if legacy and isinstance(raw_values, str):
            values = as_label_list(raw_values, legacy=True)
        else:
            values = raw_values if isinstance(raw_values, list) else [raw_values]
        for item in values:
            if isinstance(item, Mapping):
                raw_label = normalize_text(item.get("raw_label", item.get("label", "")))
                label = normalize_text(item.get("label", raw_label))
                evidence = normalize_text(item.get("evidence", ""))
                confidence = item.get("confidence")
            else:
                raw_label = normalize_text(item)
                label = raw_label
                evidence = ""
                confidence = None
            normalized_label = normalize_label(label)
            if not normalized_label:
                continue
            normalized_tags.setdefault(dimension_id, [])
            if normalized_label not in {normalize_label(v) for v in normalized_tags[dimension_id]}:
                normalized_tags[dimension_id].append(label)
                detail = {
                    "label": label,
                    "raw_label": raw_label or label,
                    "normalized_label": normalized_label,
                    "evidence": evidence,
                }
                if confidence is not None:
                    try:
                        detail["confidence"] = float(confidence)
                    except (TypeError, ValueError):
                        pass
                details.setdefault(dimension_id, []).append(detail)
    return normalized_tags, details
