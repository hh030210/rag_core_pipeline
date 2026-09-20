"""POI aliases and scenic-area hierarchy used for query-side scope resolution.

The index only stores a document-level ``source_file`` for the current corpus.
Queries, however, usually mention a child POI (for example ``长陵``) rather than
the document-level scenic area (``明十三陵``).  This small, explicit registry
bridges that granularity mismatch without making a child POI a dimension tag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class PoiEntity:
    """A query entity and the document-level scenic area it belongs to."""

    name: str
    scenic_area: str
    aliases: tuple[str, ...] = ()
    entity_type: str = "poi"

    def mentions(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


# This is intentionally a compact, curated seed rather than a fuzzy NER model.
# All entries are scoped to the seven scenic areas present in this evaluation
# corpus.  New entities can be added safely without changing retrieval code.
DEFAULT_POI_ENTITIES: tuple[PoiEntity, ...] = (
    PoiEntity("明十三陵", "明十三陵", ("十三陵", "十三陵景区", "明十三陵景区"), "scenic_area"),
    PoiEntity("长陵", "明十三陵", ("长陵陵寝", "长陵景区")),
    PoiEntity("定陵", "明十三陵", ("定陵陵寝", "定陵景区")),
    PoiEntity("康陵", "明十三陵"),
    PoiEntity("昭陵", "明十三陵"),
    PoiEntity("十三陵神道", "明十三陵", ("神道", "十三陵神路")),
    PoiEntity("棂星门", "明十三陵", ("龙凤门", "火焰牌坊")),
    PoiEntity("石牌坊", "明十三陵"),
    PoiEntity("祾恩殿", "明十三陵", ("祾思殿", "金丝楠木大殿")),
    PoiEntity("颐和园", "颐和园", ("颐和园景区",), "scenic_area"),
    PoiEntity("昆明湖", "颐和园"),
    PoiEntity("西堤", "颐和园", ("昆明湖西堤",)),
    PoiEntity("铜牛", "颐和园", ("昆明湖铜牛",)),
    PoiEntity("万寿山", "颐和园"),
    PoiEntity("佛香阁", "颐和园"),
    PoiEntity("西湖", "西湖", ("西湖景区",), "scenic_area"),
    PoiEntity("苏堤", "西湖"),
    PoiEntity("白堤", "西湖"),
    PoiEntity("杨公堤", "西湖", ("杨堤",)),
    PoiEntity("断桥", "西湖", ("断桥残雪",)),
    PoiEntity("雷峰塔", "西湖"),
    PoiEntity("少林寺", "少林寺", ("少林景区", "少林寺景区"), "scenic_area"),
    PoiEntity("塔林", "少林寺"),
    PoiEntity("三皇寨", "少林寺"),
    PoiEntity("初祖庵", "少林寺"),
    PoiEntity("二祖庵", "少林寺"),
    PoiEntity("常住院", "少林寺"),
    PoiEntity("龙门石窟", "龙门石窟", ("龙门", "龙门景区"), "scenic_area"),
    PoiEntity("卢舍那大佛", "龙门石窟", ("卢舍那",)),
    PoiEntity("奉先寺", "龙门石窟"),
    PoiEntity("龙门二十品", "龙门石窟"),
    PoiEntity("张家界", "张家界", ("张家界景区", "张家界国家森林公园"), "scenic_area"),
    PoiEntity("袁家界", "张家界"),
    PoiEntity("天子山", "张家界"),
    PoiEntity("天下第一桥", "张家界"),
    PoiEntity("南孔庙", "南孔庙", ("衢州孔庙", "南宗孔庙"), "scenic_area"),
)


class PoiRegistry:
    """Resolve longest query aliases to the document-level scenic scope."""

    def __init__(self, available_scenic_areas: Iterable[str],
                 entities: Iterable[PoiEntity] = DEFAULT_POI_ENTITIES):
        self.available_scenic_areas = {str(item).strip() for item in available_scenic_areas if str(item).strip()}
        self.entities = tuple(
            entity for entity in entities if entity.scenic_area in self.available_scenic_areas
        )
        self._by_mention: Dict[str, List[PoiEntity]] = {}
        for entity in self.entities:
            for mention in entity.mentions():
                mention = str(mention).strip()
                if mention:
                    self._by_mention.setdefault(mention, []).append(entity)

    def resolve(self, query: str) -> List[Dict[str, str]]:
        """Return all matching entities and their parent scenic areas.

        Multiple scopes are retained for explicit comparison questions.  The
        retrieval layer performs an OR over scopes, while entity metadata is
        retained for diagnostics.
        """
        matches: List[Dict[str, str]] = []
        seen = set()
        for mention in sorted(self._by_mention, key=len, reverse=True):
            if mention not in query:
                continue
            for entity in self._by_mention[mention]:
                key = (mention, entity.name, entity.scenic_area)
                if key in seen:
                    continue
                seen.add(key)
                matches.append({
                    "mention": mention,
                    "canonical_name": entity.name,
                    "scenic_area": entity.scenic_area,
                    "entity_type": entity.entity_type,
                })
        return matches

    def scenic_scopes(self, query: str) -> List[str]:
        return list(dict.fromkeys(item["scenic_area"] for item in self.resolve(query)))
