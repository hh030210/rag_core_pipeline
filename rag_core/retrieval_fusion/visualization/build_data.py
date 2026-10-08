"""Build the local viewer's JSON files from a retrieval run directory."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


VIEWER_DIR = Path(__file__).resolve().parent
REPO_ROOT = VIEWER_DIR.parents[2]
DEFAULT_RUN_DIR = REPO_ROOT / "result" / "real_merged7_deepseek_v4pro_rerank_20260920_run1"

ROUTE_FIELDS = {
    "semantic_results": "semantic",
    "dimension_results": "dimension",
    "baseline_results": "fusion",
}

COMPACT_FIELDS = (
    "rank",
    "score",
    "final_score",
    "fused_score",
    "semantic_score",
    "normalized_semantic_score",
    "dimension_score",
    "normalized_dimension_score",
    "base_score",
    "semantic_rank",
    "dimension_rank",
    "rerank_score",
    "answerability",
    "quality_score",
    "rerank_status",
)


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _first_existing(candidates: Iterable[Path], description: str) -> Path:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    shown = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {description}. Checked:\n  {shown}")


def resolve_sources(
    run_dir: str | Path,
    evaluation_path: str | Path | None = None,
    chunks_path: str | Path | None = None,
) -> tuple[Path, Path]:
    run_dir = _resolve_path(run_dir)
    evaluation = (
        _resolve_path(evaluation_path)
        if evaluation_path
        else _first_existing(
            (
                run_dir / "evaluation_noexpansion" / "results.jsonl",
                run_dir / "evaluation" / "results.jsonl",
                run_dir / "results.jsonl",
            ),
            "evaluation results JSONL",
        )
    )
    chunks = (
        _resolve_path(chunks_path)
        if chunks_path
        else _first_existing(
            (
                run_dir / "chunks" / "chunks.json",
                run_dir / "chunks.json",
                run_dir / "chunks" / "all_chunks_chunks.json",
            ),
            "chunk corpus JSON",
        )
    )
    return evaluation, chunks


def find_default_run_dir() -> Path:
    """Use the newest complete run under runs/, falling back to the bundled experiment."""
    runs_dir = REPO_ROOT / "runs"
    complete_runs: list[tuple[float, Path]] = []
    if runs_dir.is_dir():
        for run_dir in runs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            try:
                evaluation, chunks = resolve_sources(run_dir)
            except (FileNotFoundError, OSError):
                continue
            modified = max(evaluation.stat().st_mtime, chunks.stat().st_mtime)
            complete_runs.append((modified, run_dir))
    if complete_runs:
        return max(complete_runs, key=lambda item: item[0])[1]
    return DEFAULT_RUN_DIR


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            records.append(value)
    return records


def _read_rerank_rows(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    if path.suffix.lower() == ".jsonl":
        rows = _read_jsonl(path)
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            rows = value.get("rows", [value])
        elif isinstance(value, list):
            rows = value
        else:
            raise ValueError("Rerank input must be a JSON array/object or JSONL file")
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = row.get("id", row.get("dataset_index"))
        if row_id is not None:
            output[str(row_id)] = row
    return output


def _as_items(value: Any, field_name: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array")
    if any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{field_name} items must be JSON objects")
    return value


def _number(item: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = item.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return float(value)
    return None


def _discover_run_artifact(
    run_dir: Path,
    filename: str,
    explicit_path: str | Path | None = None,
) -> Path | None:
    if explicit_path:
        path = _resolve_path(explicit_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    for directory in (run_dir, *run_dir.parents):
        candidate = directory / filename
        if candidate.is_file():
            return candidate
        if directory == REPO_ROOT:
            break
        try:
            directory.relative_to(REPO_ROOT)
        except ValueError:
            break
    return None


def _load_dimension_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        return {}
    output: dict[str, dict[str, str]] = {}
    for dimension_id, item in value.items():
        if not isinstance(item, dict):
            continue
        key = str(item.get("id") or dimension_id)
        output[key] = {
            "name": str(item.get("name") or key),
            "path": str(item.get("path") or item.get("name") or key),
        }
    return output


def _load_chunk_dimension_tags(
    path: Path | None,
    dimension_metadata: dict[str, dict[str, str]],
) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    documents = value.get("documents", {}) if isinstance(value, dict) else {}
    if not isinstance(documents, dict):
        return {}
    output: dict[str, dict[str, Any]] = {}
    for chunk_id, document in documents.items():
        if not isinstance(document, dict):
            continue
        tags = document.get("tags", {})
        details = document.get("tag_details", {})
        if not isinstance(tags, dict):
            continue
        dimensions = []
        for dimension_id, raw_labels in tags.items():
            labels = raw_labels if isinstance(raw_labels, list) else [raw_labels]
            detail_rows = details.get(dimension_id, []) if isinstance(details, dict) else []
            if not isinstance(detail_rows, list):
                detail_rows = []
            detail_by_label = {
                str(item.get("label") or "").strip(): item
                for item in detail_rows
                if isinstance(item, dict)
            }
            meta = dimension_metadata.get(str(dimension_id), {})
            compact_labels = []
            for label in labels:
                if label is None or not str(label).strip():
                    continue
                label_text = str(label)
                detail = detail_by_label.get(label_text, {})
                compact = {"label": label_text}
                confidence = _number(detail, "confidence")
                if confidence is not None:
                    compact["confidence"] = confidence
                compact_labels.append(compact)
            if compact_labels:
                dimensions.append(
                    {
                        "dimension_id": str(dimension_id),
                        "dimension_name": meta.get("name", str(dimension_id)),
                        "dimension_path": meta.get("path", str(dimension_id)),
                        "labels": compact_labels,
                    }
                )
        if dimensions:
            output[str(chunk_id)] = {"dimensions": dimensions}
    return output


def _query_dimensions(
    source: dict[str, Any],
    dimension_metadata: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    query_analysis = source.get("query_analysis") or {}
    constraints = query_analysis.get("constraints", {}) if isinstance(query_analysis, dict) else {}
    if not isinstance(constraints, dict):
        return []
    output = []
    for dimension_id, constraint in constraints.items():
        if not isinstance(constraint, dict):
            continue
        meta = dimension_metadata.get(str(dimension_id), {})
        labels = constraint.get("labels", [])
        intent_terms = constraint.get("intent_terms", [])
        output.append(
            {
                "dimension_id": str(dimension_id),
                "dimension_name": meta.get("name", str(dimension_id)),
                "dimension_path": meta.get("path", str(dimension_id)),
                "labels": [str(label) for label in labels if label is not None],
                "intent_terms": [str(term) for term in intent_terms if term is not None],
                "match": str(constraint.get("match") or "ANY"),
                "role": str(constraint.get("role") or "auxiliary"),
            }
        )
    return output


def _compact_dimension_matches(value: Any) -> list[dict[str, Any]]:
    matches = _as_items(value, "dimension matches")
    fields = (
        "dimension_id",
        "dimension_path",
        "label",
        "similarity",
        "idf_bonus",
        "role",
        "source",
        "evidence",
    )
    return [
        {name: match[name] for name in fields if match.get(name) is not None}
        for match in matches
    ]


def _normalized(values: list[float | None]) -> list[float | None]:
    present = [value for value in values if value is not None]
    if not present:
        return [None] * len(values)
    low, high = min(present), max(present)
    if math.isclose(low, high):
        return [1.0 if value is not None else None for value in values]
    return [round((value - low) / (high - low), 8) if value is not None else None for value in values]


def _compact_route(items: list[dict[str, Any]], route: str) -> list[dict[str, Any]]:
    score_names = {
        "semantic": ("semantic_score", "score"),
        "dimension": ("dimension_score", "score"),
        "fusion": ("final_score", "fused_score", "score"),
    }
    normalized_name = {
        "semantic": "normalized_semantic_score",
        "dimension": "normalized_dimension_score",
        "fusion": None,
    }[route]
    raw_scores = [_number(item, *score_names[route]) for item in items]
    normalized = _normalized(raw_scores)
    output: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        chunk_id = item.get("chunk_id") or item.get("doc_id")
        if not chunk_id:
            raise ValueError(f"{route} result at rank {index + 1} has no chunk_id")
        compact: dict[str, Any] = {"chunk_id": str(chunk_id), "rank": item.get("rank") or index + 1}
        for name in COMPACT_FIELDS:
            value = item.get(name)
            if value is not None:
                compact[name] = value
        score = raw_scores[index]
        if score is not None:
            compact.setdefault("score", score)
        if route == "dimension":
            if item.get("matched_dimensions") is not None:
                compact["matched_dimensions"] = item["matched_dimensions"]
            if item.get("dimension_paths") is not None:
                compact["dimension_paths"] = item["dimension_paths"]
            compact["matches"] = _compact_dimension_matches(item.get("matches", []))
        if route == "fusion":
            compact.setdefault("final_score", score)
            compact.setdefault("fused_score", score)
            if item.get("sem_rank") is not None:
                compact.setdefault("semantic_rank", item["sem_rank"])
            if item.get("dim_rank") is not None:
                compact.setdefault("dimension_rank", item["dim_rank"])
        elif normalized_name:
            existing = item.get(normalized_name)
            compact[normalized_name] = existing if isinstance(existing, (int, float)) else normalized[index]
        output.append(compact)
    return output


def _compact_rerank(items: Any, field_name: str) -> list[dict[str, Any]]:
    values = _as_items(items, field_name)
    output = []
    for index, item in enumerate(values):
        chunk_id = item.get("chunk_id") or item.get("doc_id")
        if not chunk_id:
            raise ValueError(f"{field_name} result at rank {index + 1} has no chunk_id")
        compact = {"chunk_id": str(chunk_id), "rank": item.get("rank") or index + 1}
        for name in COMPACT_FIELDS:
            if item.get(name) is not None:
                compact[name] = item[name]
        if item.get("sem_rank") is not None:
            compact.setdefault("semantic_rank", item["sem_rank"])
        if item.get("dim_rank") is not None:
            compact.setdefault("dimension_rank", item["dim_rank"])
        output.append(compact)
    return output


def _chunk_content(item: dict[str, Any]) -> dict[str, str] | None:
    text = item.get("chunk_text_full") or item.get("chunk_text") or item.get("doc_text") or item.get("text")
    if text is None:
        return None
    return {
        "title": str(item.get("chunk_gen_title") or item.get("doc_title") or item.get("title") or ""),
        "source_file": str(item.get("source_file") or ""),
        "text": str(text),
    }


def _load_chunks(path: Path) -> dict[str, dict[str, str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        rows = value.get("chunks", value.get("records", []))
    else:
        rows = value
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON array of chunks")
    contents: dict[str, dict[str, str]] = {}
    for item in rows:
        if not isinstance(item, dict):
            continue
        chunk_id = item.get("chunk_id") or item.get("doc_id")
        content = _chunk_content(item)
        if chunk_id and content:
            contents[str(chunk_id)] = content
    if not contents:
        raise ValueError(f"No chunk text found in {path}")
    return contents


def _add_missing_content(contents: dict[str, dict[str, str]], items: Iterable[dict[str, Any]]) -> None:
    for item in items:
        chunk_id = item.get("chunk_id")
        if chunk_id is not None and str(chunk_id) not in contents:
            content = _chunk_content(item)
            if content:
                contents[str(chunk_id)] = content


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_viewer_data(
    run_dir: str | Path | None = None,
    evaluation_path: str | Path | None = None,
    chunks_path: str | Path | None = None,
    rerank_path: str | Path | None = None,
    tags_path: str | Path | None = None,
    dimension_metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    run_dir = find_default_run_dir() if run_dir is None else _resolve_path(run_dir)
    evaluation, chunks = resolve_sources(run_dir, evaluation_path, chunks_path)
    tags_source = _discover_run_artifact(run_dir, "tags_output_v2.json", tags_path)
    dimension_metadata_source = _discover_run_artifact(
        run_dir, "dimension_metadata_v2.json", dimension_metadata_path
    )
    dimension_metadata = _load_dimension_metadata(dimension_metadata_source)
    chunk_dimension_tags = _load_chunk_dimension_tags(tags_source, dimension_metadata)
    rows = _read_jsonl(evaluation)
    if not rows:
        raise ValueError(f"Evaluation file is empty: {evaluation}")
    if rerank_path:
        resolved_rerank_path = _resolve_path(rerank_path)
    else:
        resolved_rerank_path = next(
            (path for path in (run_dir / "rerank_results.jsonl", run_dir / "rerank_results.json") if path.is_file()),
            None,
        )
    rerank_rows = _read_rerank_rows(resolved_rerank_path)
    chunk_contents = _load_chunks(chunks)

    output_rows: list[dict[str, Any]] = []
    candidate_ids: set[str] = set()
    available_routes: set[str] = set()
    for index, source in enumerate(rows, start=1):
        retrieval = source.get("retrieval") or {}
        if not isinstance(retrieval, dict):
            raise ValueError(f"Evaluation row {index} has an invalid retrieval field")
        row_id = str(source.get("id", index))
        rerank = rerank_rows.get(row_id, {})
        semantic = _compact_route(_as_items(retrieval.get("semantic_results"), "semantic_results"), "semantic")
        dimension = _compact_route(_as_items(retrieval.get("dimension_results"), "dimension_results"), "dimension")
        baseline = _compact_route(_as_items(retrieval.get("fusion_results"), "fusion_results"), "fusion")
        deepseek = _compact_rerank(rerank.get("deepseek_results"), "deepseek_results")
        answerability = _compact_rerank(rerank.get("answerability_results"), "answerability_results")

        routes = {
            "semantic_results": semantic,
            "dimension_results": dimension,
            "fusion_base_results": baseline,
            "baseline_results": baseline,
            "deepseek_results": deepseek,
            "answerability_results": answerability,
        }
        for route_name, route_items in routes.items():
            if route_items:
                available_routes.add(route_name)
            for item in route_items:
                candidate_ids.add(item["chunk_id"])
                _add_missing_content(chunk_contents, (item,))

        gold = source.get("gold") or {}
        gold_ids = source.get("gold_chunk_ids") or (gold.get("chunk_ids") if isinstance(gold, dict) else None) or []
        output_rows.append(
            {
                "dataset_index": index,
                "id": row_id,
                "question": source.get("question") or "",
                "retrieval_query": source.get("retrieval_query") or source.get("question") or "",
                "query_dimensions": _query_dimensions(source, dimension_metadata),
                "source_run": run_dir.name,
                "gold_chunk_ids": list(dict.fromkeys(str(chunk_id) for chunk_id in gold_ids)),
                "semantic_count": len(semantic),
                "semantic_chunk_ids": [item["chunk_id"] for item in semantic],
                "semantic_results": semantic,
                "dimension_count": len(dimension),
                "dimension_chunk_ids": [item["chunk_id"] for item in dimension],
                "dimension_results": dimension,
                "fusion_base_results": baseline,
                "baseline_strategy": "evaluation fusion_results",
                "baseline_results": baseline,
                "deepseek_results": deepseek,
                "answerability_results": answerability,
            }
        )

    missing_content = sorted(candidate_ids - chunk_contents.keys())
    if missing_content:
        examples = ", ".join(missing_content[:5])
        raise ValueError(
            f"Chunk corpus is missing text for {len(missing_content)} retrieval candidates. "
            f"Check that evaluation and chunks are from the same run. Examples: {examples}"
        )

    _write_json_atomic(VIEWER_DIR / "fusion_routes.json", output_rows)
    _write_json_atomic(VIEWER_DIR / "rerank_chunk_contents.json", chunk_contents)
    _write_json_atomic(VIEWER_DIR / "chunk_dimension_tags.json", chunk_dimension_tags)
    return {
        "evaluation": evaluation,
        "chunks": chunks,
        "tags": tags_source,
        "dimension_metadata": dimension_metadata_source,
        "question_count": len(output_rows),
        "chunk_count": len(chunk_contents),
        "tagged_chunk_count": len(chunk_dimension_tags),
        "candidate_chunk_count": len(candidate_ids),
        "available_routes": sorted(available_routes),
        "rerank_input": resolved_rerank_path,
    }


def add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Run directory containing evaluation results and chunk data (default: newest complete run under runs/)",
    )
    parser.add_argument("--evaluation", help="Explicit evaluation results.jsonl path")
    parser.add_argument("--chunks", help="Explicit chunk corpus JSON path")
    parser.add_argument("--tags", help="Optional tags_output_v2.json path for chunk dimension labels")
    parser.add_argument(
        "--dimension-metadata",
        help="Optional dimension_metadata_v2.json path for readable dimension names",
    )
    parser.add_argument(
        "--rerank-results",
        help="Optional JSON/JSONL with per-question deepseek_results and/or answerability_results",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_source_arguments(parser)
    args = parser.parse_args()
    result = build_viewer_data(
        args.run_dir,
        args.evaluation,
        args.chunks,
        args.rerank_results,
        args.tags,
        args.dimension_metadata,
    )
    print(f"Updated {VIEWER_DIR / 'fusion_routes.json'}")
    print(f"Updated {VIEWER_DIR / 'rerank_chunk_contents.json'}")
    print(f"Updated {VIEWER_DIR / 'chunk_dimension_tags.json'}")
    print(f"Questions: {result['question_count']}; chunks: {result['chunk_count']}")
    print(f"Chunks with dimension labels: {result['tagged_chunk_count']}")
    if result["tags"] is None:
        print("No tags_output_v2.json found; chunk dimension labels will be empty.")
    if result["dimension_metadata"] is None:
        print("No dimension_metadata_v2.json found; raw dimension IDs will be shown as names.")
    print("Available routes: " + ", ".join(result["available_routes"]))
    if "deepseek_results" not in result["available_routes"] or "answerability_results" not in result["available_routes"]:
        print("No source data for one or more rerank routes; those routes will be omitted from the viewer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
