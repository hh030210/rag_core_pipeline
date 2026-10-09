"""Cross-validate one-search adaptive centroid weights using query-only features."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .experiment import DEFAULT_MODEL, DEFAULT_RUN, LAB_DIR, load_jsonl, query_parts, score_metrics, unique
from .vector_search import DenseVectorSearch, l2_normalize

STRATEGIES = [
    "current_question_weight_2_centroid",
    "fixed_question_mass_30", "fixed_question_mass_32",
    "fixed_question_mass_33", "fixed_question_mass_34",
    "fixed_question_mass_50", "fixed_question_mass_60",
]
FEATURE_NAMES = [
    "subquery_count", "question_chars", "mean_subquery_chars",
    "total_subquery_chars", "subquery_to_question_char_ratio",
    "mean_query_subquery_cosine", "min_query_subquery_cosine",
    "std_query_subquery_cosine", "mean_pairwise_subquery_cosine",
]


def cosine(left: List[float], right: List[float]) -> float:
    a, b = l2_normalize(left), l2_normalize(right)
    return sum(x * y for x, y in zip(a, b))


def feature_thresholds(values: List[float]) -> List[float]:
    ordered = sorted(values)
    if not ordered:
        return []
    thresholds = []
    for quantile in (0.2, 0.4, 0.6, 0.8):
        position = quantile * (len(ordered) - 1)
        low = int(math.floor(position))
        high = int(math.ceil(position))
        value = (ordered[low] + ordered[high]) / 2.0
        if all(abs(value - previous) > 1e-9 for previous in thresholds):
            thresholds.append(value)
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--baseline-results", type=Path, default=None)
    parser.add_argument("--variant-rankings", type=Path,
                        default=LAB_DIR / "outputs/centroid_variants_refined_20261008/per_query_rankings.jsonl")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    baseline_path = args.baseline_results or args.run_dir / "evaluation_after_pull_4896b80" / "results.jsonl"
    rows = [row for row in load_jsonl(baseline_path) if (row.get("gold") or {}).get("chunk_ids")]
    records = load_jsonl(args.variant_rankings)
    if len(rows) != len(records):
        raise RuntimeError(f"mapped rows={len(rows)} != ranking rows={len(records)}")
    manifest = json.loads((args.run_dir / "store_manifest.json").read_text(encoding="utf-8"))
    output = args.output_dir or LAB_DIR / "outputs" / ("adaptive_centroid_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)

    query_data: List[Dict[str, Any]] = []
    all_texts: List[str] = []
    for row in rows:
        question = str(row.get("question") or "").strip()
        parts = [part for part in unique(query_parts(row)) if part != question]
        query_data.append({"question": question, "parts": parts})
        all_texts.extend([question, *parts])
    all_texts = unique(all_texts)
    print(f"[adaptive-centroid] encoding {len(all_texts)} query texts", flush=True)
    searcher = DenseVectorSearch(
        qdrant_url=manifest.get("qdrant_url", "http://127.0.0.1:6333"),
        collection=manifest["collection"],
        model_path=str(args.model_path), device=args.device,
    )
    encoded = searcher.encode(all_texts)
    vectors = {text: vector for text, vector in zip(all_texts, encoded)}

    features: Dict[str, List[float]] = {name: [] for name in FEATURE_NAMES}
    for item in query_data:
        question, parts = item["question"], item["parts"]
        qchars = len(question)
        lengths = [len(part) for part in parts]
        sims = [cosine(vectors[question], vectors[part]) for part in parts]
        pair_sims = [cosine(vectors[parts[i]], vectors[parts[j]])
                     for i in range(len(parts)) for j in range(i + 1, len(parts))]
        features["subquery_count"].append(float(len(parts)))
        features["question_chars"].append(float(qchars))
        features["mean_subquery_chars"].append(sum(lengths) / len(lengths) if lengths else 0.0)
        features["total_subquery_chars"].append(float(sum(lengths)))
        features["subquery_to_question_char_ratio"].append(sum(lengths) / max(qchars, 1))
        features["mean_query_subquery_cosine"].append(sum(sims) / len(sims) if sims else 0.0)
        features["min_query_subquery_cosine"].append(min(sims) if sims else 0.0)
        mean_sim = sum(sims) / len(sims) if sims else 0.0
        features["std_query_subquery_cosine"].append(
            math.sqrt(sum((value - mean_sim) ** 2 for value in sims) / len(sims)) if sims else 0.0)
        features["mean_pairwise_subquery_cosine"].append(
            sum(pair_sims) / len(pair_sims) if pair_sims else 0.0)

    rankings = {name: [record["rankings"][name][:20] for record in records] for name in STRATEGIES}
    metric_values: Dict[str, List[Tuple[int, float, float]]] = {}
    for name, lists in rankings.items():
        values = []
        for row, ranked in zip(rows, lists):
            gold = set((row.get("gold") or {}).get("chunk_ids") or [])
            first = next((index for index, chunk_id in enumerate(ranked, 1) if chunk_id in gold), None)
            hit = int(first is not None and first <= 15)
            mrr = 1.0 / first if first is not None and first <= 10 else 0.0
            ndcg = sum(1.0 / math.log2(index + 1) for index, chunk_id in enumerate(ranked[:5], 1)
                       if chunk_id in gold)
            ideal = sum(1.0 / math.log2(index + 1) for index in range(1, min(5, len(gold)) + 1))
            values.append((hit, mrr, ndcg / ideal if ideal else 0.0))
        metric_values[name] = values

    folds = [int(hashlib.sha256(str(row.get("retrieval_query") or row.get("question") or "").encode("utf-8")).hexdigest()[:8], 16) % 5
             for row in rows]
    print(f"[adaptive-centroid] queries={len(rows)} candidate_strategies={len(STRATEGIES)}", flush=True)

    candidate_models: Dict[str, Dict[str, Any]] = {
        name: {"type": "fixed", "strategy": name} for name in STRATEGIES
    }
    gate_pairs = list(__import__("itertools").combinations([
        "current_question_weight_2_centroid", "fixed_question_mass_32",
        "fixed_question_mass_50", "fixed_question_mass_60",
    ], 2))
    predictions: List[List[str] | None] = [None] * len(rows)
    selected_by_fold: Dict[str, Dict[str, Any]] = {}
    policy_name_by_fold: Dict[str, str] = {}

    def choice(model: Dict[str, Any], row_index: int) -> str:
        if model["type"] == "fixed":
            return model["strategy"]
        low_strategy = model["low_strategy"]
        high_strategy = model["high_strategy"]
        return low_strategy if features[model["feature"]][row_index] <= model["threshold"] else high_strategy

    def training_objective(model: Dict[str, Any], indices: List[int]) -> Tuple[int, float, float]:
        hit_sum, mrr_sum, ndcg_sum = 0, 0.0, 0.0
        for row_index in indices:
            strategy = choice(model, row_index)
            hit, mrr, ndcg = metric_values[strategy][row_index]
            hit_sum += hit
            mrr_sum += mrr
            ndcg_sum += ndcg
        return hit_sum, mrr_sum, ndcg_sum

    for fold in range(5):
        train = [i for i, assigned in enumerate(folds) if assigned != fold]
        test = [i for i, assigned in enumerate(folds) if assigned == fold]
        models = dict(candidate_models)
        for feature_name in FEATURE_NAMES:
            thresholds = feature_thresholds([features[feature_name][i] for i in train])
            for left, right in gate_pairs:
                for threshold in thresholds:
                    for low_strategy, high_strategy in ((left, right), (right, left)):
                        name = f"gate_{feature_name}_{threshold:.4f}_{low_strategy}_else_{high_strategy}"
                        models[name] = {
                            "type": "gate", "feature": feature_name, "threshold": threshold,
                            "low_strategy": low_strategy, "high_strategy": high_strategy,
                        }
        chosen_name = max(models, key=lambda name: training_objective(models[name], train))
        chosen = models[chosen_name]
        selected_by_fold[str(fold)] = chosen
        policy_name_by_fold[str(fold)] = chosen_name
        for row_index in test:
            predictions[row_index] = rankings[choice(chosen, row_index)][row_index]
        print(f"[adaptive-centroid] fold={fold} selected={chosen_name}", flush=True)

    cv_metrics = score_metrics(rows, [ranking or [] for ranking in predictions])
    fixed_metrics = {name: score_metrics(rows, ranking) for name, ranking in rankings.items()}
    summary = {
        "experiment": "adaptive_one_search_centroid_policy",
        "collection": manifest["collection"], "mapped_queries": len(rows),
        "features": FEATURE_NAMES, "fixed_strategies": STRATEGIES,
        "gate_feature_thresholds_from_train_only": [0.2, 0.4, 0.6, 0.8],
        "grouped_5fold_cv_selected_policy_by_fold": selected_by_fold,
        "grouped_5fold_cv_policy_name_by_fold": policy_name_by_fold,
        "grouped_5fold_cv_metrics": cv_metrics,
        "fixed_strategy_metrics": fixed_metrics,
        "fold_rule": "SHA256(retrieval_query) modulo five",
        "search_cost": "Each policy selects one weighted query vector from query-only features and makes one Qdrant search.",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "per_query_rankings.jsonl").open("w", encoding="utf-8") as handle:
        for i, row in enumerate(rows):
            handle.write(json.dumps({
                "id": row.get("id"), "retrieval_query": row.get("retrieval_query"),
                "features": {name: features[name][i] for name in FEATURE_NAMES},
                "gold_chunk_ids": (row.get("gold") or {}).get("chunk_ids", []),
                "cv_selected_ranking": predictions[i],
            }, ensure_ascii=False) + "\n")
    print(f"[adaptive-centroid] results={output}")
    print(json.dumps({"cv_metrics": cv_metrics, "policies": policy_name_by_fold}, ensure_ascii=False))


if __name__ == "__main__":
    main()
