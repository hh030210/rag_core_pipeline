"""隔离的 v4 查询解析器。

v4 的目标不是把所有问题强行转成标签，而是把查询拆成两种可审计的信号：

* ``labels``：问题中出现、可以进入叶子维度倒排索引的规范/原子标签；
* ``intent_terms``：没有规范标签字面值，但能说明查询意图的短语，例如“几点开门”。

两者都保留在约束中，但只有前者走精确标签匹配，后者走同叶子维度的语义召回。
当 LLM 失败或返回空约束时，继续使用 QueryParserV2 中的 schema 约束确定性恢复。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, List, Mapping, Optional

from .query_parser import QueryParserV2, _as_list, _clean_terms
from .schema_v2 import SCHEMA_VERSION, normalize_label, normalize_text


QUERY_PARSER_V4_VERSION = "4.4"


# These rules identify the property being asked about. Merely mentioning an
# object (for example “陵墓” or “皇帝”) is intentionally not enough to make its
# type/role a required retrieval dimension.
_STRICT_INTENT_RULES = {
    "opening_period": (
        r"开放时间|营业时间|开门时间|关门时间|几点(?:开|关)|什么时候(?:开|关)|"
        r"几点开放|开门|关门|闭馆|闭园|夜游时间"
    ),
    "ticket_policy": (
        r"预约|预订|预定|怎么买票|如何购票|购票渠道|退票|改签|取票|实名|"
        r"免票|优惠|打折|证件.{0,4}入园|二次入园"
    ),
    "ticket_type": (
        r"票价|门票.{0,5}(?:多少|价格|收费|费用)|多少钱|收费标准|"
        r"成人票|全价票|联票|套票"
    ),
    "transportation_mode": (
        r"怎么去|怎么到|如何到达|交通方式|交通路线|公交|地铁|自驾|出租车|"
        r"高铁|火车站|机场|缆车|索道|游船"
    ),
    "visitor_service": (
        r"厕所|卫生间|餐饮|餐厅|吃饭|住宿|寄存|存包|医疗|充电|母婴室|"
        r"租借|电话|wifi|Wi-Fi|饮水|导览|急救|讲解器|注意事项|游览须知|"
        r"允许|禁止|能不能|是否可以|宠物|携带|适合老人|适合小孩|游览路线|行程安排"
    ),
    "location_hierarchy": r"在哪里|在哪儿|位于哪里|具体位置|地址|哪个地方|什么地方",
    "entity_type": r"是什么类型|属于哪种|哪类(?:建筑|景点|场所)|是什么(?:建筑|景点|场所)",
    "person_role": r"是什么身份|人物身份|担任什么|哪位皇帝|是谁主持|主持者是谁",
    "historical_event": (
        r"历史|始建|建于|修建|建造|营建|重建|重修|迁建|毁坏|朝代|哪年|"
        r"多少年|多久|谁建|谁造|谁修|生前|合葬|历史事件"
    ),
    "cultural_connotation": (
        r"为什么叫|为何叫|名称由来|名字由来|得名|来历|寓意|象征|文化(?:价值|意义|内涵)|"
        r"传说|典故|祭祀|信仰|文物价值"
    ),
    "artistic_feature": (
        r"艺术|风格|雕刻|雕塑|绘画|壁画|书法|石刻|工艺|造型|题材|作品|建筑特色"
    ),
    "landscape_composition": (
        r"占地(?:多少|面积)|面积(?:多大|多少)|多高|高度|多长|长度|规模|布局|结构|组成|"
        r"几座|几个|多少个|多少座|有哪些.{0,8}(?:建筑|景点|殿|塔|洞|佛像|陵墓|景观)|"
        r"(?:里面|内部).{0,6}(?:有|包括|组成)"
    ),
    "visual_feature": r"形状|色彩|颜色|光影|外观|视觉效果|看起来|像什么|什么样子",
}


class QueryParserV4(QueryParserV2):
    """Schema-constrained parser with intent-slot preservation and diagnostics."""

    @staticmethod
    def _strict_intents(query_text: str) -> Dict[str, List[str]]:
        query = normalize_text(query_text)
        selected: Dict[str, List[str]] = {}
        for dimension_id, pattern in _STRICT_INTENT_RULES.items():
            hits = [match.group(0) for match in re.finditer(pattern, query)]
            if hits:
                selected[dimension_id] = list(dict.fromkeys(hits))

        # Price/category and purchase policy are neighboring dimensions. A
        # discount or eligibility question is policy-only unless price is
        # explicitly requested; this prevents “学生票优惠” from becoming a
        # brittle two-dimension AND.
        if "ticket_policy" in selected and "ticket_type" in selected:
            if not re.search(r"票价|多少钱|价格|收费|费用|收费标准", query):
                selected.pop("ticket_type", None)

        # “游览路线” is an in-park service intent, not necessarily a transport
        # mode. Keep transportation only when an actual travel mode/arrival
        # expression is present.
        if "visitor_service" in selected and "transportation_mode" in selected:
            if not re.search(r"怎么去|怎么到|如何到达|公交|地铁|自驾|出租车|高铁|火车站|机场", query):
                selected.pop("transportation_mode", None)
        return selected

    def _recover_constraints(
        self, query_text: str, llm_constraints: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        broad = super()._recover_constraints(query_text, llm_constraints)
        by_dimension = {item["dimension_id"]: item for item in broad}
        strict = self._strict_intents(query_text)
        if not strict:
            # Controlled recall fallback: choose exactly one property-bearing
            # dimension. Generic subject dimensions are excluded because a
            # mentioned building/person/place is not necessarily what the
            # user asks about.
            cue_candidates = self._intent_candidates(query_text)
            excluded = {"entity_type", "person_role", "location_hierarchy"}
            ranked = sorted(
                (
                    (score, dimension_id, terms)
                    for dimension_id, (score, terms) in cue_candidates.items()
                    if dimension_id not in excluded
                ),
                key=lambda value: (-value[0], value[1]),
            )
            if ranked:
                _score, dimension_id, terms = ranked[0]
                strict[dimension_id] = terms
            else:
                fallback = next(
                    (
                        item for item in broad
                        if item.get("dimension_id") not in excluded
                    ),
                    None,
                )
                if fallback:
                    dimension_id = str(fallback["dimension_id"])
                    strict[dimension_id] = list(fallback.get("intent_terms", []))
                else:
                    # Object-only/recommendation query: retain at most one
                    # explicit anchor rather than returning no dimension at
                    # all. Location is most discriminative, followed by entity
                    # type and person role.
                    anchor_priority = {
                        "location_hierarchy": 0,
                        "entity_type": 1,
                        "person_role": 2,
                    }
                    anchors = sorted(
                        (
                            item for item in broad
                            if item.get("dimension_id") in anchor_priority
                            and item.get("labels")
                        ),
                        key=lambda item: anchor_priority[str(item["dimension_id"])],
                    )
                    if anchors:
                        dimension_id = str(anchors[0]["dimension_id"])
                        strict[dimension_id] = list(anchors[0].get("intent_terms", []))
        # Keep the broad constraints for recall, but give exactly one queried
        # property the main role. Other broad hits become auxiliary evidence:
        # they may recall candidates but cannot dominate ranking.
        main_dimension = ""
        if strict:
            query = normalize_text(query_text)
            main_dimension = min(
                strict,
                key=lambda dim: (query.find(strict[dim][0]) if strict[dim] else len(query), dim),
            )
        composite = bool(re.search(r"同时|分别|以及|并且|还有|和.*(?:分别|各自)", normalize_text(query_text)))
        secondary = set(strict) - {main_dimension} if composite else set()

        # A strict routing rule may identify a dimension which was absent from
        # broad recovery; create it so the main intent is never lost.
        for dimension_id, intent_terms in strict.items():
            if dimension_id not in self.maps["leaves"]:
                continue
            item = by_dimension.setdefault(dimension_id, {
                "dimension_id": dimension_id, "labels": [], "intent_terms": [],
                "match": "ANY", "source": "strict_intent",
            })
            item["intent_terms"] = _clean_terms(
                list(item.get("intent_terms", [])) + intent_terms, limit=8
            )
            item["source"] = (
                f"{item.get('source')}+strict_intent"
                if item.get("source") != "strict_intent" else "strict_intent"
            )
            item["confidence"] = max(float(item.get("confidence", 0.0) or 0.0), 0.9)

        result: List[Dict[str, Any]] = []
        for dimension_id, item in by_dimension.items():
            item = dict(item)
            item["role"] = (
                "main" if dimension_id == main_dimension else
                "secondary" if dimension_id in secondary else "auxiliary"
            )
            result.append(item)
        return result

    def _cache_key(self, qid: Any, query_text: str) -> str:
        digest = hashlib.sha1(normalize_text(query_text).encode("utf-8")).hexdigest()
        return (
            f"{SCHEMA_VERSION}:query_parser_{QUERY_PARSER_V4_VERSION}:"
            f"{self.vocabulary_signature}:{qid}:{digest}"
        )

    def _load_cache(self) -> None:
        if not os.path.exists(self.cache_file):
            return
        try:
            with open(self.cache_file, "r", encoding="utf-8") as handle:
                value = json.load(handle)
            if (
                isinstance(value, dict)
                and value.get("schema_version") == SCHEMA_VERSION
                and value.get("parser_version") == QUERY_PARSER_V4_VERSION
                and value.get("vocabulary_signature") == self.vocabulary_signature
            ):
                self.cache = value.get("items", {})
        except (OSError, json.JSONDecodeError):
            self.cache = {}

    def _save_cache(self) -> None:
        parent = os.path.dirname(self.cache_file)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.cache_file, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema_version": SCHEMA_VERSION,
                    "parser_version": QUERY_PARSER_V4_VERSION,
                    "vocabulary_signature": self.vocabulary_signature,
                    "items": self.cache,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

    def _allowed_nodes(self) -> Dict[str, Dict[str, Any]]:
        """Only expose indexable leaves to the LLM and exact parser."""

        allowed: Dict[str, Dict[str, Any]] = {}
        for node_id in self.maps["leaves"]:
            node = self.maps["nodes"][node_id]
            allowed[str(node_id)] = node
            allowed[normalize_text(node.get("name", ""))] = node
            for alias in _as_list(node.get("aliases")):
                alias = normalize_text(alias)
                if alias:
                    allowed[alias] = node
        return allowed

    @staticmethod
    def _slot(item: Mapping[str, Any], index: int) -> Dict[str, Any]:
        labels = list(item.get("labels", []) or [])
        intent_terms = list(item.get("intent_terms", []) or [])
        return {
            "slot_id": f"slot_{index}",
            "dimension_id": str(item.get("dimension_id", "")),
            "labels": labels,
            "intent_terms": intent_terms,
            "required": True,
        }

    def parse(self, qid: Any, query_text: str) -> Dict[str, Any]:
        key = self._cache_key(qid, query_text)
        if key in self.cache:
            return self.cache[key]

        # Passing leaves only makes it impossible for a model to invent a
        # parent as a retrievable tag. Parent expansion remains a retrieval
        # operation when an explicitly stored parent constraint is encountered.
        nodes = [self.maps["nodes"][node_id] for node_id in self.maps["leaves"]]
        llm_error = ""
        try:
            parser_method = getattr(self.miner, "parse_query_intent_v4", None)
            if parser_method is None:
                parsed = self.miner.parse_query_intent_v2(query_text, nodes)
            else:
                parsed = parser_method(query_text, nodes)
        except Exception as exc:
            parsed = {}
            llm_error = f"{type(exc).__name__}: {exc}"

        llm_constraints = self._clean_llm_constraints(parsed)
        constraints = self._recover_constraints(query_text, llm_constraints)
        main_dimension = next(
            (item["dimension_id"] for item in constraints if item.get("role") == "main"), ""
        )
        secondary_dimensions = [
            item["dimension_id"] for item in constraints if item.get("role") == "secondary"
        ]
        raw_diagnostics = parsed.get("diagnostics", {}) if isinstance(parsed, dict) else {}
        if not isinstance(raw_diagnostics, dict):
            raw_diagnostics = {}
        sources = sorted({str(item.get("source", "unknown")) for item in constraints})
        result = {
            "schema_version": SCHEMA_VERSION,
            "parser_version": QUERY_PARSER_V4_VERSION,
            "constraints": constraints,
            "routing": {
                "main_dimension": main_dimension,
                "secondary_dimensions": secondary_dimensions,
            },
            "required_slots": [self._slot(item, index) for index, item in enumerate(constraints, 1)],
            "diagnostics": {
                "llm_constraint_count": int(raw_diagnostics.get("llm_constraint_count", len(llm_constraints)) or 0),
                "llm_intent_only_count": int(
                    raw_diagnostics.get(
                        "llm_intent_only_count",
                        sum(bool(item.get("intent_terms")) and not item.get("labels") for item in llm_constraints),
                    )
                    or 0
                ),
                "recovered_constraint_count": len(constraints),
                # ``fallback_used`` answers whether the LLM supplied no usable
                # constraint. Deterministic cue/lexical enrichment is exposed
                # separately so a valid LLM result is not mislabeled as a
                # fallback merely because it was augmented.
                "fallback_used": not bool(llm_constraints),
                "deterministic_enrichment_used": any(
                    item.get("source") != "llm_v4" for item in constraints
                ),
                "recovery_sources": sources,
                "llm_attempts": int(raw_diagnostics.get("llm_attempts", 1) or 1),
                "llm_error": str(raw_diagnostics.get("llm_error", "") or llm_error),
            },
        }
        self.cache[key] = result
        self._save_cache()
        return result
