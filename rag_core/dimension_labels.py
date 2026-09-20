"""Canonical concept ids shared by query and document dimension labels."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from .schema_v2 import normalize_label


# Precision-oriented defaults. Runs may extend or override these through
# dimension_label_ontology.json without changing retrieval code.
DEFAULT_CONCEPT_ALIASES: Dict[str, Dict[str, list[str]]] = {
    "entity_type": {
        "imperial_tomb": ["陵墓", "帝陵", "皇陵", "帝王陵墓"],
        "temple": ["寺庙", "寺院", "佛寺"],
        "grotto": ["石窟", "洞窟"],
        "palace_hall": ["殿宇", "宫殿", "大殿"],
    },
    "person_role": {
        "emperor": ["皇帝", "帝王", "君主"],
        "empress": ["皇后", "后妃"],
        "monk": ["僧人", "和尚", "高僧"],
    },
    "historical_event": {
        "construction": ["修建", "建造", "营建", "兴建", "始建"],
        "reconstruction": ["重建", "重修", "复建", "修复"],
        "destruction": ["毁坏", "焚毁", "损毁", "破坏"],
        "relocation": ["迁建", "迁移", "迁都"],
    },
    "transportation_mode": {
        "bus": ["公交", "公交车", "公共汽车", "巴士"],
        "metro": ["地铁", "轨道交通"],
        "self_drive": ["自驾", "驾车", "开车"],
        "walk": ["步行", "徒步"],
        "cableway": ["索道", "缆车"],
        "boat": ["游船", "乘船", "船"],
    },
    "ticket_policy": {
        "reservation": ["预约", "预订", "提前预约", "预约购票", "实名预约"],
        "online_purchase": ["线上购票", "网上购票", "在线购票", "公众号购票"],
        "free_admission": ["免票", "免费", "免费开放"],
        "discount": ["优惠", "优惠政策", "半价", "折扣"],
    },
    "ticket_type": {
        "adult_ticket": ["成人票", "全价票"],
        "student_ticket": ["学生票", "学生优惠票"],
        "child_ticket": ["儿童票", "少儿票"],
        "senior_ticket": ["老人票", "老年票"],
        "combined_ticket": ["联票", "套票", "通票"],
    },
    "opening_period": {
        "all_day": ["全天开放", "全天"],
        "closed": ["闭馆", "闭园", "不开放"],
    },
    "visitor_service": {
        "luggage_storage": ["寄存", "行李寄存"],
        "guided_tour": ["讲解", "导览", "导游服务", "讲解服务"],
        "parking": ["停车", "停车场"],
        "accessible_service": ["无障碍", "无障碍服务", "无障碍设施"],
    },
}


def _key(value: Any) -> str:
    value = normalize_label(value).casefold()
    return re.sub(r"[\s\u3000，。！？；：、（）()【】\[\]{}‘’“”\"'·_\-]+", "", value)


class CanonicalLabelResolver:
    """Map surface labels to stable, dimension-scoped concept ids."""

    def __init__(self, run_dir: str | Path, vocabulary: Mapping[str, Iterable[str]]):
        ontology: Dict[str, Dict[str, list[str]]] = {
            dim: {concept: list(aliases) for concept, aliases in concepts.items()}
            for dim, concepts in DEFAULT_CONCEPT_ALIASES.items()
        }
        path = Path(run_dir) / "dimension_label_ontology.json"
        if not path.exists():
            # Keep the project-wide reviewed ontology available to new run
            # directories while allowing a run-specific file to override it.
            path = Path(__file__).resolve().parents[1] / "data" / "dimension_label_ontology.json"
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw = raw.get("dimensions", raw) if isinstance(raw, dict) else {}
            for dim, concepts in raw.items():
                if not isinstance(concepts, dict):
                    continue
                target = ontology.setdefault(str(dim), {})
                for concept, aliases in concepts.items():
                    if isinstance(aliases, str):
                        aliases = [aliases]
                    if isinstance(aliases, list):
                        target[str(concept)] = [str(value) for value in aliases]

        self.alias_to_concept: Dict[str, Dict[str, str]] = {}
        self.curated_aliases: Dict[str, list[str]] = {}
        for dim, concepts in ontology.items():
            target = self.alias_to_concept.setdefault(dim, {})
            for concept, aliases in concepts.items():
                target[_key(concept)] = str(concept)
                self.curated_aliases.setdefault(dim, []).append(str(concept))
                for alias in aliases:
                    if _key(alias):
                        target[_key(alias)] = str(concept)
                        self.curated_aliases.setdefault(dim, []).append(str(alias))

        # Every indexed value remains valid if no curated synonym exists.
        for dim, labels in vocabulary.items():
            target = self.alias_to_concept.setdefault(str(dim), {})
            for label in labels:
                normalized = _key(label)
                if normalized:
                    contained = [
                        (len(alias), concept) for alias, concept in target.items()
                        if len(alias) >= 2 and alias in normalized
                    ]
                    concept = max(contained)[1] if contained else f"label:{normalized}"
                    target.setdefault(normalized, concept)

    def canonical(self, dimension_id: str, label: Any) -> str:
        normalized = _key(label)
        if not normalized:
            return ""
        aliases = self.alias_to_concept.get(str(dimension_id), {})
        if normalized in aliases:
            return aliases[normalized]
        contained = [
            (len(alias), concept) for alias, concept in aliases.items()
            if len(alias) >= 2 and alias in normalized
        ]
        if contained:
            return max(contained)[1]
        return f"label:{normalized}"

    def concepts(self, dimension_id: str, labels: Iterable[Any]) -> set[str]:
        return {
            concept for concept in (
                self.canonical(dimension_id, label) for label in labels
            ) if concept
        }

    def labels_in_text(self, dimension_id: str, text: Any) -> list[str]:
        """Return curated ontology surfaces explicitly present in a query.

        Indexed labels are deliberately not scanned here: doing so would turn
        every incidental noun into a query constraint.  Only reviewed ontology
        aliases may enrich a dimension that the query parser already selected.
        """
        normalized_text = _key(text)
        matches = []
        for alias in self.curated_aliases.get(str(dimension_id), []):
            key = _key(alias)
            if len(key) >= 2 and key in normalized_text and alias not in matches:
                matches.append(alias)
        return matches
