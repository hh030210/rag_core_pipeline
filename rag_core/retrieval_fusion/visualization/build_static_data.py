"""Build static viewer data files so index.html can be opened without a server."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

VIEWER_DIR = Path(__file__).resolve().parent
ENGINE_DIR = VIEWER_DIR.parent / "fusion_engine"
SNAPSHOT_NAME = re.compile(r"^retrieval_fusion_[^/\\]+\.json$")
COMPACT_FIELDS = (
    "chunk_id", "score", "semantic_score", "dimension_score",
    "normalized_semantic_score", "normalized_dimension_score", "lexical_score",
    "original_fusion_rank", "original_fusion_score", "effective_semantic_score",
    "semantic_score_imputed", "score_components", "semantic_rank", "dimension_rank",
    "fusion_branch", "dimension_paths", "matched_dimensions", "matches",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_static_manifest(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    return json.loads(text.split("=", 1)[1].strip().rstrip(";"))


def read_static_detail(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    return json.loads(text.rsplit(" = ", 1)[1].strip().rstrip(";"))


def add_experiment_metrics(base: dict, summary: dict) -> dict:
    """Replace the previous offline row with dimension-score fusion metrics."""
    if not summary:
        return base
    total = summary.get("snapshots", base.get("total_queries", 0))
    metrics = summary.get("metrics", {})
    mapped = metrics.get("gold_mapped", summary.get("gold_mapped", base.get("mapped_queries", 0)))
    sources = {
        "semantic": ("semantic", "语义检索"),
        "dimension": ("dimension", "维度检索"),
        "fusion": ("online", "线上融合"),
        "fusion_new": ("dimension_score_blend", "离线融合（维度分数）"),
    }
    base_rows = []
    for key, (source, label) in sources.items():
        values = metrics.get(source)
        if not values:
            old = next((row for row in base.get("metrics", []) if row.get("key") == key), {})
            if "hits" in old and "ranking_scores" in old:
                base_rows.append({**old, "label": label})
                continue
            scores = old.get("scores", {})
            base_rows.append({
                "key": key, "label": label,
                "hits": {metric: round(scores.get(metric, 0) * mapped / 100)
                         for metric in ("Hit@1", "Hit@5", "Hit@10")},
                "ranking_scores": {metric: scores.get(metric, 0) / 100
                                   for metric in ("MRR@10", "nDCG@5")},
            })
        else:
            base_rows.append({
                "key": key, "label": label,
                "hits": {"Hit@1": values["hit_at_1"], "Hit@5": values["hit_at_5"],
                         "Hit@10": values["hit_at_10"]},
                "ranking_scores": {"MRR@10": values["mrr_at_10"],
                                    "nDCG@5": values["ndcg_at_5"]},
            })
    return {"total_queries": total, "mapped_queries": mapped,
            "unmapped_queries": total - mapped, "metrics": base_rows,
            "strategy": summary.get("strategy")}


def summarize_static_route(rows: list[tuple[list[dict], set[str]]]) -> dict:
    total = len(rows)
    mapped = [(items, gold) for items, gold in rows if gold]
    ranks = [first_gold_rank(items, gold) for items, gold in mapped]
    return {
        "hit_at_1": sum(rank is not None and rank <= 1 for rank in ranks),
        "hit_at_5": sum(rank is not None and rank <= 5 for rank in ranks),
        "hit_at_10": sum(rank is not None and rank <= 10 for rank in ranks),
        "mrr_at_10": sum(1 / rank for rank in ranks if rank is not None and rank <= 10) / (total or 1),
        "ndcg_at_5": sum(ndcg_at_5(items, gold) for items, gold in rows) / (total or 1),
    }


def rebuild_from_experiment_cache(assets_dir: Path, experiment_dir: Path) -> bool:
    """Merge experiment rankings into the historical static viewer cohort."""
    index_path = assets_dir / "snapshot_index.js"
    summary_path = experiment_dir / "summary.json"
    if not index_path.is_file() or not summary_path.is_file():
        return False
    previous = read_static_manifest(index_path)
    summary = read_json(summary_path)
    comparisons_by_name = {}
    comparison_path = experiment_dir / "comparison.jsonl"
    if comparison_path.is_file():
        for line in comparison_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                comparisons_by_name[row["source_snapshot"]] = row

    detail_files = previous.get("detail_files", {})
    items = []
    semantic_rows, dimension_rows = [], []
    for item in previous.get("items", []):
        name = item["name"]
        experiment_path = experiment_dir / name
        detail_path = assets_dir / detail_files[name]
        if not experiment_path.is_file() or not detail_path.is_file():
            items.append(item)
            continue
        experiment = read_json(experiment_path)
        detail = read_static_detail(detail_path)
        gold_ids = set(comparisons_by_name.get(name, {}).get("gold_chunk_ids", []))
        semantic_rows.append((detail["routes"].get("semantic", []), gold_ids))
        dimension_rows.append((detail["routes"].get("dimension", []), gold_ids))
        detail["routes"] = {
            "semantic": detail["routes"].get("semantic", []),
            "dimension": detail["routes"].get("dimension", []),
            "old": detail["routes"].get("old", []),
            "new": compact(experiment.get("fusion_candidates", [])),
        }
        detail["gold"] = {**(detail.get("gold") or {}),
                           "gold_chunk_ids": comparisons_by_name.get(name, {}).get("gold_chunk_ids", [])}
        payload = json.dumps(detail, ensure_ascii=False, separators=(",", ":"))
        script = (
            "window.RETRIEVAL_FUSION_DETAILS = window.RETRIEVAL_FUSION_DETAILS || {};\n"
            f"window.RETRIEVAL_FUSION_DETAILS[{json.dumps(name, ensure_ascii=False)}] = {payload};\n"
        )
        detail_path.write_text(script, encoding="utf-8")
        comparison = comparisons_by_name.get(name, {})
        item = {key: item.get(key) for key in (
            "name", "query", "has_new", "gold_count", "semantic_rank",
            "dimension_rank", "old_rank",
        )}
        item["has_new"] = True
        item["new_rank"] = comparison.get("new_rank")
        items.append(item)
    summary = {**summary, "metrics": {
        **summary.get("metrics", {}),
        "semantic": summarize_static_route(semantic_rows),
        "dimension": summarize_static_route(dimension_rows),
    }}
    manifest = {**previous, "items": items,
                "evaluation": add_experiment_metrics(previous.get("evaluation", {}), summary)}
    index_path.write_text(
        "window.RETRIEVAL_FUSION_INDEX = " +
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    return True


def comparisons(new_dir: Path) -> dict[str, dict]:
    path = new_dir / "comparison.jsonl"
    if not path.is_file():
        return {}
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        name = row.get("source_snapshot")
        if isinstance(name, str) and SNAPSHOT_NAME.fullmatch(name):
            result[name] = row
    return result


def compact(items: list[dict]) -> list[dict]:
    return [
        {"rank": rank, **{key: item[key] for key in COMPACT_FIELDS if key in item}}
        for rank, item in enumerate(items, 1)
    ]


def first_gold_rank(items: list[dict], gold: set[str]) -> int | None:
    return next((rank for rank, item in enumerate(items, 1) if item.get("chunk_id") in gold), None)


def ndcg_at_5(items: list[dict], gold: set[str]) -> float:
    ideal_count = min(len(gold), 5)
    if not ideal_count:
        return 0.0
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    actual = sum(
        1 / math.log2(rank + 1)
        for rank, item in enumerate(items[:5], 1)
        if item.get("chunk_id") in gold
    )
    return actual / ideal


def evaluation_summary(paths: list[Path], old_dir: Path, new_dir: Path,
                       comparison_by_name: dict[str, dict]) -> dict:
    systems = {
        "semantic": {"label": "语义检索"},
        "dimension": {"label": "维度检索"},
        "fusion": {"label": "融合检索（修改前）"},
        "fusion_new": {"label": "新融合检索（修改后）"},
    }
    mapped = 0
    for path in paths:
        comparison = comparison_by_name.get(path.name, {})
        gold = set(comparison.get("gold_chunk_ids") or [])
        if not gold:
            continue
        mapped += 1
        old = read_json(old_dir / path.name)
        new_path = new_dir / path.name
        new = read_json(new_path) if new_path.is_file() else {}
        rankings = {
            "semantic": old.get("semantic_candidates") or [],
            "dimension": old.get("dimension_candidates") or [],
            "fusion": old.get("fusion_candidates") or old.get("fusion_results") or [],
            "fusion_new": new.get("fusion_candidates") or [],
        }
        for key, items in rankings.items():
            rank = first_gold_rank(items, gold)
            values = systems[key]
            values["hit1"] = values.get("hit1", 0) + int(rank is not None and rank <= 1)
            values["hit5"] = values.get("hit5", 0) + int(rank is not None and rank <= 5)
            values["hit10"] = values.get("hit10", 0) + int(rank is not None and rank <= 10)
            values["mrr10"] = values.get("mrr10", 0.0) + (1 / rank if rank is not None and rank <= 10 else 0.0)
            values["ndcg5"] = values.get("ndcg5", 0.0) + ndcg_at_5(items, gold)

    metrics = [
        ("Hit@1", "hit1"), ("Hit@5", "hit5"), ("Hit@10", "hit10"),
        ("MRR@10", "mrr10"), ("nDCG@5", "ndcg5"),
    ]
    total_queries = len(paths)
    return {
        "total_queries": total_queries,
        "mapped_queries": mapped,
        "unmapped_queries": total_queries - mapped,
        "metrics": [
            {
                "key": key,
                "label": values["label"],
                "hits": {
                    metric: values.get(source, 0)
                    for metric, source in metrics[:3]
                },
                "ranking_scores": {
                    metric: (values.get(source, 0.0) / total_queries if total_queries else 0.0)
                    for metric, source in metrics[3:]
                },
            }
            for key, values in systems.items()
        ],
    }


def detail_row(name: str, old_dir: Path, new_dir: Path, comparison: dict) -> dict:
    old = read_json(old_dir / name)
    new_path = new_dir / name
    new = read_json(new_path) if new_path.is_file() else None
    if new and new.get("query") != old.get("query"):
        raise ValueError(f"原/新快照 query 不一致：{name}")

    semantic = old.get("semantic_candidates") or []
    dimension = old.get("dimension_candidates") or []
    original = old.get("fusion_candidates") or old.get("fusion_results") or []
    updated = (new.get("fusion_candidates") or []) if new else []
    if new and {item["chunk_id"] for item in original} != {item["chunk_id"] for item in updated}:
        raise ValueError(f"原/新快照候选集不一致：{name}")

    routes = {
        "semantic": compact(semantic),
        "dimension": compact(dimension),
        "old": compact(original),
        "new": compact(updated),
    }
    chunks = old.get("chunks") if isinstance(old.get("chunks"), dict) else {}
    for item in semantic + dimension + original + updated:
        chunk_id = str(item.get("chunk_id") or "")
        if not chunk_id or (chunks.get(chunk_id) or {}).get("text"):
            continue
        chunks[chunk_id] = {
            "title": item.get("chunk_gen_title") or item.get("doc_title") or "",
            "source_file": item.get("source_file") or "",
            "text": item.get("chunk_text_full") or item.get("chunk_text") or "",
        }

    diagnostics = (new.get("diagnostics") or {}) if new else {}
    return {
        "name": name,
        "query": old.get("original_query") or old.get("query") or "",
        "retrieval_query": old.get("retrieval_query") or old.get("query") or "",
        "created_at": old.get("created_at"),
        "query_analysis": old.get("query_analysis") or {},
        "strategy": new.get("strategy") if new else None,
        "configuration": (new.get("configuration") or {}) if new else {},
        "diagnostics": diagnostics,
        "semantic_top1_gap": (
            new.get("semantic_top1_gap", diagnostics.get("semantic_top1_gap")) if new else None
        ),
        "effective_dimension_weight": (
            new.get("effective_dimension_weight", diagnostics.get("effective_dimension_weight"))
            if new else None
        ),
        "gold": comparison,
        "routes": routes,
        "chunks": chunks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ENGINE_DIR / "output")
    parser.add_argument("--output-new", type=Path, default=ENGINE_DIR / "output_new")
    parser.add_argument("--assets-dir", type=Path, default=VIEWER_DIR)
    parser.add_argument("--experiment-dir", type=Path,
                        default=ENGINE_DIR / "output_new")
    args = parser.parse_args()
    old_dir = args.output.expanduser().resolve()
    new_dir = args.output_new.expanduser().resolve()
    assets_dir = args.assets_dir.expanduser().resolve()
    experiment_dir = args.experiment_dir.expanduser().resolve()
    if rebuild_from_experiment_cache(assets_dir, experiment_dir):
        manifest = read_static_manifest(assets_dir / "snapshot_index.js")
        detail_bytes = sum((assets_dir / path).stat().st_size
                           for path in manifest.get("detail_files", {}).values()
                           if (assets_dir / path).is_file())
        print(f"Built dimension-score fusion comparison: {len(manifest.get('items', []))} questions")
        print(f"Index: {assets_dir / 'snapshot_index.js'}")
        print(f"Details: {assets_dir / 'snapshot_details'} ({detail_bytes / (1024 * 1024):.1f} MiB)")
        return 0
    if not old_dir.is_dir():
        parser.error(f"Missing original fusion output: {old_dir}")
    if not new_dir.is_dir():
        parser.error(f"Missing new fusion output: {new_dir}")

    comparison_by_name = comparisons(new_dir)
    paths = sorted(path for path in old_dir.glob("retrieval_fusion_*.json") if SNAPSHOT_NAME.fullmatch(path.name))
    if not paths:
        parser.error(f"No retrieval snapshots found in {old_dir}")

    detail_dir = assets_dir / "snapshot_details"
    detail_dir.mkdir(parents=True, exist_ok=True)
    items, detail_files = [], {}
    total_detail_bytes = 0
    for index, path in enumerate(paths):
        name = path.name
        comparison = comparison_by_name.get(name, {})
        has_new = (new_dir / name).is_file()
        detail = detail_row(name, old_dir, new_dir, comparison)
        items.append({
            "name": name,
            "query": comparison.get("query") or detail["query"] or path.stem,
            "has_new": has_new,
            "gold_count": comparison.get("gold_count"),
            "semantic_rank": comparison.get("semantic_rank"),
            "dimension_rank": comparison.get("dimension_rank"),
            "old_rank": comparison.get("old_rank"),
            "new_rank": comparison.get("new_rank") if has_new else None,
        })
        detail_path = f"snapshot_details/detail_{index:04d}.js"
        detail_files[name] = detail_path
        payload = json.dumps(detail, ensure_ascii=False, separators=(",", ":"))
        script = (
            "window.RETRIEVAL_FUSION_DETAILS = window.RETRIEVAL_FUSION_DETAILS || {};\n"
            f"window.RETRIEVAL_FUSION_DETAILS[{json.dumps(name, ensure_ascii=False)}] = {payload};\n"
        )
        output_path = assets_dir / detail_path
        output_path.write_text(script, encoding="utf-8")
        total_detail_bytes += output_path.stat().st_size

    manifest = {
        "items": items,
        "detail_files": detail_files,
        "evaluation": evaluation_summary(paths, old_dir, new_dir, comparison_by_name),
    }
    manifest_payload = json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))
    (assets_dir / "snapshot_index.js").write_text(
        f"window.RETRIEVAL_FUSION_INDEX = {manifest_payload};\n",
        encoding="utf-8",
    )
    print(f"Built static snapshot data: {len(items)} questions")
    print(f"Index: {assets_dir / 'snapshot_index.js'}")
    print(f"Details: {detail_dir} ({total_detail_bytes / (1024 * 1024):.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


