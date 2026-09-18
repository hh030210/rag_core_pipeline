import os
import json
import hashlib
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from .llm_service import DimensionMiningWithQwen
    from .schema_v2 import SCHEMA_VERSION, load_schema, normalize_label, normalize_text, schema_maps
except ImportError:
    from llm_service import DimensionMiningWithQwen
    from schema_v2 import SCHEMA_VERSION, load_schema, normalize_label, normalize_text, schema_maps


QUERY_PARSER_VERSION = "2.1"

# 这些词用于“模型没有返回约束”时恢复查询意图。它们不是标签，也不会直接
# 作为精确标签写入倒排索引；只用于选择叶子维度，并生成该维度的语义子查询。
# 这样可以处理“几点开门”“修建了多久”“怎么预约”等没有规范标签字面值的问题。
_QUERY_CUE_ALIASES = {
    "历史事件": (
        "历史", "建于", "修建", "建造", "营建", "朝代", "皇帝", "陵墓", "战争", "破坏",
        "事件", "故事", "来历", "由来", "典故", "传说", "人物", "贡献", "名称", "名字",
        "多久", "多少年", "几年", "生前", "合葬", "为什么", "供奉", "先祖", "祠堂", "碑",
        "画", "刻", "恢复祭祀", "恢复", "掌门人", "公爵", "没收", "定制", "瓷瓶", "文字",
    ),
    "文化背景": (
        "文化", "价值", "意义", "寓意", "象征", "传承", "非物质", "非遗", "祭祀", "信仰",
        "中医", "佛教", "孔子", "供奉", "文物", "仿制品", "传统文化", "祭孔", "典礼", "对联",
        "讲究", "身份", "掌门人", "传统", "青花",
    ),
    "景观特色": (
        "景观", "特色", "海拔", "高度", "高差", "长度", "多高", "多长", "面积", "数量",
        "多少个", "多少对", "几座", "几个", "动物", "看点", "全景", "俯瞰", "观景", "拍照",
    ),
    "景观组成": (
        "建筑", "结构", "布局", "空间", "组成", "洞窟", "佛像", "路线", "景点", "桥",
        "殿", "神道", "神路", "门", "规模", "什么样", "全景", "俯瞰", "观景", "拍照", "展厅",
        "互动体验", "精华路线", "山洞", "看头", "西庑", "大堂",
    ),
    "季节景观": (
        "春天", "夏天", "秋天", "冬天", "春季", "夏季", "秋季", "冬季", "季节", "雨天",
        "晴天", "天气", "赏花", "避暑",
    ),
    "开放时段": (
        "开放", "营业", "开门", "关门", "闭馆", "闭园", "夜游", "晚上", "夜间", "几点",
        "什么时候开", "什么时候关",
    ),
    "开放时间": (
        "开放", "营业", "开门", "关门", "闭馆", "闭园", "夜游", "晚上", "夜间", "几点",
        "什么时候开", "什么时候关",
    ),
    "票务规则": (
        "门票", "票价", "多少钱", "价格", "收费", "费用", "优惠", "免费", "买票", "购票",
        "预约", "预订", "预定", "退票", "取票", "联票", "打折", "有效期",
    ),
    "票务信息": (
        "门票", "票价", "多少钱", "价格", "收费", "费用", "优惠", "免费", "买票", "购票",
        "预约", "预订", "预定", "退票", "取票", "联票", "打折", "有效期",
    ),
    "服务设施": (
        "设施", "厕所", "卫生间", "餐饮", "餐厅", "吃饭", "住宿", "寄存", "存包", "代步",
        "观光车", "医疗", "充电", "充电宝", "母婴室", "配套", "租借", "礼品", "纪念品", "书店",
        "买", "购买", "卖", "电话", "wifi", "Wi-Fi", "直饮水", "饮水", "导览图", "导览",
        "急救", "紧急", "讲解器",
    ),
    "游览须知": (
        "注意", "规定", "允许", "禁止", "能不能", "是否可以", "怎么到", "怎么走", "入口",
        "徒步", "登山", "携带", "宠物", "报名", "要求", "路线", "活动", "体验", "互动", "交通",
        "机场", "火车站", "坐车", "公交", "高速", "方便", "二次入园",
    ),
    "位置层级": (
        "在哪里", "在哪", "位于", "位置", "地址", "地理", "海拔", "高差", "哪个地方", "什么地方",
    ),
    "地理位置": (
        "在哪里", "在哪", "位于", "位置", "地址", "地理", "海拔", "高差", "哪个地方",
    ),
    "周边区域": (
        "附近", "周边", "相邻", "之间", "关系", "串联", "其他景区", "另一个景区", "离得近",
    ),
}

_QUERY_STOP_TERMS = {
    "请问", "请具体说明", "我想了解", "想了解", "一下", "景区中", "景区里", "景区内", "景区",
    "其中", "有关", "相关", "怎么回事", "是什么", "哪些", "哪个", "多少", "如何", "是否",
    "有没有", "有无", "能否", "可以吗", "吗", "呢", "的", "了", "和", "与", "及", "还有",
}


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _clean_terms(values: Iterable[Any], limit: int = 8) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        term = normalize_text(value)
        if not term or term in _QUERY_STOP_TERMS or len(term) < 2:
            continue
        key = normalize_label(term)
        if key and key not in seen:
            result.append(term)
            seen.add(key)
        if len(result) >= limit:
            break
    return result


class QueryParserV2:
    """Versioned hierarchical parser with deterministic recovery.

    The LLM remains the primary parser.  The recovery path is deliberately
    query-local and schema-constrained:

    * exact vocabulary labels are attached only to the leaf that owns them;
    * intent cues select a small set of leaf dimensions when labels are absent;
    * intent-only constraints carry ``intent_terms`` and are resolved by the
      dimension semantic fallback, never treated as exact tag matches.

    ``label_vocabulary`` may be the v2 postings shape
    ``{leaf_id: {normalized_label: doc_ids}}`` or ``{leaf_id: [labels]}``.
    """

    def __init__(
        self,
        schema,
        cache_file="./experiment_data/query_cache_v2.json",
        miner=None,
        label_vocabulary: Optional[Mapping[str, Any]] = None,
    ):
        self.schema = load_schema(schema, allow_legacy=False, strict=False)
        self.maps = schema_maps(self.schema)
        self.miner = miner or DimensionMiningWithQwen()
        self.cache_file = cache_file
        self.label_vocabulary = self._normalize_vocabulary(label_vocabulary)
        self.vocabulary_signature = self._vocabulary_signature(self.label_vocabulary)
        self.cache: Dict[str, Any] = {}
        self._load_cache()

    @staticmethod
    def _normalize_vocabulary(value: Optional[Mapping[str, Any]]) -> Dict[str, List[str]]:
        normalized: Dict[str, List[str]] = {}
        if not isinstance(value, Mapping):
            return normalized
        for dimension_id, raw_labels in value.items():
            labels = raw_labels.keys() if isinstance(raw_labels, Mapping) else _as_list(raw_labels)
            cleaned = []
            seen = set()
            for raw_label in labels:
                label = normalize_text(raw_label)
                key = normalize_label(label)
                if len(key) < 2 or key in seen:
                    continue
                cleaned.append(label)
                seen.add(key)
            if cleaned:
                normalized[str(dimension_id)] = cleaned
        return normalized

    @staticmethod
    def _vocabulary_signature(vocabulary: Mapping[str, Sequence[str]]) -> str:
        compact = [(key, sorted(values)) for key, values in sorted(vocabulary.items())]
        return hashlib.sha1(json.dumps(compact, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]

    def _cache_key(self, qid, query_text):
        digest = hashlib.sha1(normalize_text(query_text).encode("utf-8")).hexdigest()
        return f"{SCHEMA_VERSION}:query_parser_{QUERY_PARSER_VERSION}:{self.vocabulary_signature}:{qid}:{digest}"

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as handle:
                    value = json.load(handle)
                if (
                    isinstance(value, dict)
                    and value.get("schema_version") == SCHEMA_VERSION
                    and value.get("parser_version") == QUERY_PARSER_VERSION
                ):
                    self.cache = value.get("items", {})
            except (OSError, json.JSONDecodeError):
                self.cache = {}

    def _save_cache(self):
        parent = os.path.dirname(self.cache_file)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.cache_file, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema_version": SCHEMA_VERSION,
                    "parser_version": QUERY_PARSER_VERSION,
                    "vocabulary_signature": self.vocabulary_signature,
                    "items": self.cache,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

    def _allowed_nodes(self) -> Dict[str, Dict[str, Any]]:
        allowed: Dict[str, Dict[str, Any]] = {}
        for node_id, node in self.maps["nodes"].items():
            allowed[str(node_id)] = node
            allowed[normalize_text(node.get("name"))] = node
            for alias in _as_list(node.get("aliases")):
                alias = normalize_text(alias)
                if alias:
                    allowed[alias] = node
        return allowed

    def _clean_llm_constraints(self, parsed: Any) -> List[Dict[str, Any]]:
        if not isinstance(parsed, dict):
            return []
        raw_constraints = parsed.get("constraints", [])
        # Compatibility with the old {dimension: [labels]} response.
        if not raw_constraints:
            raw_constraints = [
                {"dimension_id": key, "labels": value, "match": "ANY"}
                for key, value in parsed.items()
                if key not in {"schema_version", "parser_version", "diagnostics"}
            ]
        if not isinstance(raw_constraints, list):
            return []

        allowed = self._allowed_nodes()
        cleaned: List[Dict[str, Any]] = []
        for item in raw_constraints:
            if not isinstance(item, dict):
                continue
            raw_dimension = normalize_text(item.get("dimension_id", item.get("dimension", "")))
            node = allowed.get(raw_dimension)
            if not node:
                continue
            labels = item.get("labels", item.get("values", []))
            labels = _as_list(labels)
            cleaned_labels = []
            seen = set()
            for label in labels:
                label = normalize_text(label)
                key = normalize_label(label)
                if key and key not in seen:
                    cleaned_labels.append(label)
                    seen.add(key)
            intent_terms = _clean_terms(item.get("intent_terms", []))
            match = str(item.get("match", "ANY")).upper()
            if match not in {"ANY", "ALL"}:
                match = "ANY"
            # Keep a constraint with intent_terms even when it has no canonical
            # label.  It is resolved through semantic dimension recall later.
            if not cleaned_labels and not intent_terms:
                continue
            cleaned.append({
                "dimension_id": str(node["id"]),
                "labels": cleaned_labels,
                "intent_terms": intent_terms,
                "match": match,
                "source": "llm",
            })
        return cleaned

    def _exact_vocabulary_hits(self, query_text: str) -> Dict[str, List[str]]:
        normalized_query = normalize_text(query_text)
        hits: Dict[str, List[str]] = {}
        candidates: List[Tuple[int, str, str]] = []
        for dimension_id, labels in self.label_vocabulary.items():
            if dimension_id not in self.maps["leaves"]:
                continue
            for label in labels:
                key = normalize_label(label)
                if len(key) < 2 or key in _QUERY_STOP_TERMS:
                    continue
                if key in normalized_query:
                    candidates.append((len(key), dimension_id, label))
        # Prefer longer phrases.  Short labels that are contained in a longer
        # hit are usually generic fragments and add little retrieval value.
        candidates.sort(reverse=True)
        selected: List[Tuple[int, str, str]] = []
        for length, dimension_id, label in candidates:
            if any(dimension_id == d and label in other for _, d, other in selected):
                continue
            selected.append((length, dimension_id, label))
            hits.setdefault(dimension_id, []).append(label)
            if len(hits[dimension_id]) >= 5:
                continue
        return hits

    @staticmethod
    def _cue_terms_for_node(node: Mapping[str, Any]) -> List[str]:
        name = normalize_text(node.get("name", ""))
        terms = list(_QUERY_CUE_ALIASES.get(name, ()))
        terms.extend(_as_list(node.get("aliases")))
        # The description is useful for generated schemas whose Chinese name
        # differs from the built-in scenic schema.
        description = normalize_text(node.get("description", ""))
        if description:
            terms.extend(re.findall(r"[一-鿿]{2,8}", description))
        return list(dict.fromkeys(term for term in terms if len(normalize_text(term)) >= 2))

    def _intent_candidates(self, query_text: str) -> Dict[str, Tuple[int, List[str]]]:
        normalized_query = normalize_text(query_text)
        result: Dict[str, Tuple[int, List[str]]] = {}
        for leaf_id in self.maps["leaves"]:
            node = self.maps["nodes"][leaf_id]
            matched = []
            score = 0
            for term in self._cue_terms_for_node(node):
                term = normalize_text(term)
                if term and term in normalized_query:
                    matched.append(term)
                    # Longer cues are more specific.  A two-character cue is
                    # still useful but cannot dominate a descriptive phrase.
                    score += 2 if len(term) >= 3 else 1
            if score:
                result[leaf_id] = (score, list(dict.fromkeys(matched)))
        return result

    def _query_focus_terms(self, query_text: str, cue_terms: Sequence[str]) -> List[str]:
        # The full query is already sent to the semantic subquery.  Here we
        # only keep explicit cue phrases; greedily slicing arbitrary Chinese
        # spans can create false terms such as “少年” from “多少年”.
        terms = list(cue_terms)
        if not terms:
            query = normalize_text(query_text)
            query = re.sub(r"^(请问|请具体说明|我想了解一下|我想了解|想了解一下|想了解)", "", query)
            query = re.sub(r"[？?！!。；;，,]+$", "", query)
            if 2 <= len(query) <= 20:
                terms.append(query)
        return _clean_terms(terms, limit=8)

    def _recover_constraints(self, query_text: str, llm_constraints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        by_dimension: Dict[str, Dict[str, Any]] = {}
        for constraint in llm_constraints:
            by_dimension[constraint["dimension_id"]] = {
                **constraint,
                "labels": list(constraint.get("labels", [])),
                "intent_terms": list(constraint.get("intent_terms", [])),
            }

        exact_hits = self._exact_vocabulary_hits(query_text)
        for dimension_id, labels in exact_hits.items():
            item = by_dimension.setdefault(
                dimension_id,
                {
                    "dimension_id": dimension_id,
                    "labels": [],
                    "intent_terms": [],
                    "match": "ANY",
                    "source": "lexical_label",
                },
            )
            existing = {normalize_label(label) for label in item["labels"]}
            item["labels"].extend(label for label in labels if normalize_label(label) not in existing)
            item["source"] = "llm+lexical_label" if item.get("source") == "llm" else "lexical_label"

        cue_candidates = self._intent_candidates(query_text)
        if cue_candidates:
            best_score = max(score for score, _ in cue_candidates.values())
            # Retain strong dimensions and close runners-up for genuinely
            # composite questions; cap the number to avoid broad over-routing.
            selected = [
                (dimension_id, score, terms)
                for dimension_id, (score, terms) in cue_candidates.items()
                if score >= max(1, best_score - 2)
            ]
            selected.sort(key=lambda item: (-item[1], item[0]))
            for dimension_id, _score, cue_terms in selected[:4]:
                item = by_dimension.setdefault(
                    dimension_id,
                    {
                        "dimension_id": dimension_id,
                        "labels": [],
                        "intent_terms": [],
                        "match": "ANY",
                        "source": "intent_cue",
                    },
                )
                item["intent_terms"] = _clean_terms(
                    list(item.get("intent_terms", [])) + self._query_focus_terms(query_text, cue_terms),
                    limit=8,
                )
                if item.get("source") == "llm":
                    item["source"] = "llm+intent_cue"

        # If the LLM produced a valid label constraint, do not manufacture
        # unrelated dimensions.  If it produced nothing, exact labels and
        # intent candidates above are the complete deterministic recovery.
        result = []
        for dimension_id, item in by_dimension.items():
            labels = list(dict.fromkeys(
                normalize_text(label) for label in item.get("labels", []) if normalize_label(label)
            ))
            intent_terms = _clean_terms(item.get("intent_terms", []), limit=8)
            if not labels and not intent_terms:
                continue
            item["labels"] = labels[:5]
            item["intent_terms"] = intent_terms
            item["match"] = "ALL" if str(item.get("match", "ANY")).upper() == "ALL" else "ANY"
            item["confidence"] = 1.0 if labels else 0.55
            result.append(item)
        return result

    def parse(self, qid, query_text):
        key = self._cache_key(qid, query_text)
        if key in self.cache:
            return self.cache[key]
        nodes = [self.maps["nodes"][node_id] for node_id in self.maps["nodes"]]
        try:
            parsed = self.miner.parse_query_intent_v2(query_text, nodes)
        except Exception as exc:
            parsed = {}
            llm_error = f"{type(exc).__name__}: {exc}"
        else:
            llm_error = ""
        llm_constraints = self._clean_llm_constraints(parsed)
        constraints = self._recover_constraints(query_text, llm_constraints)
        result = {
            "schema_version": SCHEMA_VERSION,
            "parser_version": QUERY_PARSER_VERSION,
            "constraints": constraints,
            "diagnostics": {
                "llm_constraint_count": len(llm_constraints),
                "recovered_constraint_count": len(constraints),
                "fallback_used": bool(not llm_constraints or any(item.get("source") != "llm" for item in constraints)),
                "llm_error": llm_error,
            },
        }
        self.cache[key] = result
        self._save_cache()
        return result

class QueryParser:
    def __init__(self, cache_file="./experiment_data/query_cache.json"):
        self.miner = DimensionMiningWithQwen()
        self.cache_file = cache_file
        self.cache = {}
        self._load_cache()
        
    def _load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                self.cache = json.load(f)
    
    def _save_cache(self):
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def parse(self, qid, query_text, dim_meta, schema_dim_fields=None):
        """
        解析 Query，优先读缓存。

        :param dim_meta: IndexBuilder 生成的维度元数据
        :param schema_dim_fields: Milvus schema 中的实际 dim_xxx 字段名列表
                               用于约束 LLM 输出使用正确的维度名
        """
        qid = str(qid)

        # 1. 读缓存
        if qid in self.cache:
            return self.cache[qid]

        # 2. 构造动态配置 (区分 Enum 和 Open)
        enum_map = {}
        open_dims = []

        for dim, meta in dim_meta.items():
            if meta['is_enum']:
                enum_map[dim] = meta['values']
            else:
                open_dims.append(dim)

        # 3. 调用 LLM
        all_dims = list(enum_map.keys()) + open_dims

        parsed_result = self.miner.parse_query_intent(
            query_text, all_dims,
            enum_values_map=enum_map,
            schema_dim_fields=schema_dim_fields
        )

        # 4. 写入缓存
        self.cache[qid] = parsed_result
        self._save_cache()

        return parsed_result
