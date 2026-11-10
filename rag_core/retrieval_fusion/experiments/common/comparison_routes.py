"""Score all three routes of an experiment against its mapped Golden chunks."""
from rag_core.evaluation import route_metrics

ROUTES = ("semantic", "dimension", "fusion")


def average_precision(hits, gold_ids, k):
    gold_ids = set(map(str, gold_ids))
    if not gold_ids:
        return 0.0
    found = 0
    precision_sum = 0.0
    for rank, hit in enumerate(hits[:k], start=1):
        if str(hit.get("chunk_id", "")) in gold_ids:
            found += 1
            precision_sum += found / rank
    return precision_sum / min(len(gold_ids), k)


def evaluate_routes(retrieval, gold_ids, ks, public_hit):
    routes = {}
    depth = max(ks)
    for route in ROUTES:
        candidates = retrieval.get(f"{route}_candidates") or []
        metrics = route_metrics(candidates, gold_ids, ks)
        for k in ks:
            metrics[f"average_precision_at_{k}"] = average_precision(candidates, gold_ids, k)
        routes[route] = {
            "results": [public_hit(hit, rank) for rank, hit in enumerate(candidates[:depth], 1)],
            "metrics": metrics,
        }
    return routes


def summarize_routes(rows, ks, summarize):
    return {
        route: summarize([
            {"results": row["routes"][route]["results"], "gold_chunk_ids": row["gold_chunk_ids"]}
            for row in rows
        ], ks)
        for route in ROUTES
    }
