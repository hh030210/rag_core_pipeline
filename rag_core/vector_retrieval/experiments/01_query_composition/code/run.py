"""Read-only vector query-composition comparison on saved evaluation inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from uuid import uuid4
import time
from importlib import import_module
from pathlib import Path

from rag_core.embedding import EmbeddingModel
from rag_core.evaluation import route_metrics
from rag_core.storage import VectorStore
from rag_core.vector_retrieval import VectorRetriever

HERE = Path(__file__).resolve().parent.parent
STRATEGIES = (
    "original_only", "subqueries_concat", "original_plus_subqueries_concat",
    "original_plus_subqueries_dedup", "subqueries_comma",
    "original_subqueries_centroid_w2", "original_subqueries_rrf",
)
KS = (1, 5, 10, 15, 20)


def unique(items):
    return list(dict.fromkeys(items))


def prepare(row):
    original = row.get("original_query") or row.get("question") or row.get("query")
    if not isinstance(original, str) or not original.strip():
        raise ValueError("Every row needs a nonempty original_query/question/query")
    original = original.strip()
    analysis = row.get("query_analysis") or {}
    parts = row.get("subqueries")
    source = "subqueries"
    if parts is None:
        parts = analysis.get("subqueries")
        source = "query_analysis.subqueries"
    if parts is None:
        retrieval = row.get("retrieval_query")
        if not isinstance(retrieval, str) or not retrieval.strip():
            raise ValueError("Missing subqueries and retrieval_query; do not generate new expansions")
        parts = retrieval.split("|")
        source = "retrieval_query split by |"
    if not isinstance(parts, list) or any(not isinstance(p, str) for p in parts):
        raise ValueError("subqueries must be a list of strings")
    # Preserve duplicates and text exactly for the current concatenation control.
    parts = [p for p in parts if p.strip()]
    # Saved joined text has separator spaces; trim only reconstructed components.
    if source == "retrieval_query split by |":
        parts = [p.strip() for p in parts]
    fallback = not parts
    parts = parts or [original]
    texts = {
        "original_only": original,
        "subqueries_concat": " | ".join(parts),
        "original_plus_subqueries_concat": " | ".join([original, *parts]),
        "original_plus_subqueries_dedup": " | ".join(unique([original, *parts])),
        "subqueries_comma": "，".join(parts),
    }
    gold = row.get("gold") or {}
    gold_ids = gold.get("chunk_ids") or gold.get("gold_chunk_ids") or row.get("gold_chunk_ids") or []
    if not isinstance(gold_ids, list):
        raise ValueError("Gold chunk IDs must be a list")
    return {
        "original": original, "parts": parts, "texts": texts,
        "component_texts": unique([original, *parts]),
        "spots": analysis.get("spot_names") or [],
        "gold": unique([str(cid) for cid in gold_ids]),
        "relevance": row.get("gold_relevance") or gold.get("relevance") or {},
        "subquery_source": source, "empty_subqueries_fallback": fallback,
    }


def centroid(vectors, weights):
    normalized = [VectorRetriever._unit_vector(v) for v in vectors]
    return VectorRetriever._unit_vector([
        math.fsum(float(v[i]) * w for v, w in zip(normalized, weights))
        for i in range(len(normalized[0]))
    ])


def rrf(lists, depth, constant=60):
    scores, items = {}, {}
    for hits in lists:
        for rank, hit in enumerate(hits, 1):
            cid = str(hit["chunk_id"])
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (constant + rank)
            items.setdefault(cid, hit)
    ordered = sorted(scores, key=lambda cid: -scores[cid])
    return [dict(items[cid], score=scores[cid], rank=i)
            for i, cid in enumerate(ordered[:depth], 1)]


def summarize(records):
    result = {}
    for name in STRATEGIES:
        metrics = [r["strategies"][name]["metrics"] for r in records]
        n = len(metrics)
        mean = lambda key: math.fsum(float(m[key]) for m in metrics) / n
        item = {"query_count": n}
        for k in KS:
            item[f"hit@{k}"] = mean(f"hit_at_{k}")
            item[f"gold_recall@{k}"] = mean(f"gold_recall_at_{k}")
            item[f"mrr@{k}"] = mean(f"rr_at_{k}")
            item[f"ndcg@{k}"] = mean(f"ndcg_at_{k}")
        ranks = [m["first_gold_rank"] for m in metrics if m["first_gold_rank"]]
        item["mean_first_gold_rank_on_hits"] = sum(ranks) / len(ranks) if ranks else None
        item["miss_count"] = sum(m["miss"] for m in metrics)
        result[name] = item
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-dataset", "--input-results", dest="input_dataset", type=Path, required=True,
                        help="Online evaluation dataset directory with saved queries and Golden")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pool", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0, help="0=all; positive=smoke only")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.pool < max(KS) or args.limit < 0:
        parser.error("--pool must be >=20 and --limit must be >=0")
    target = (args.output_dir or HERE / "output").resolve()
    if not target.is_relative_to(HERE / "output"):
        parser.error("--output-dir must stay inside this experiment folder")
    replace_output = import_module("rag_core.retrieval_fusion.experiments.06_dimension_score.code.run").replace_output
    replace_output(target, lambda staged: _run(args, staged))


def _run(args, output):
    source = args.input_dataset
    if source.name == 'index.json':
        source = source.parent
    input_manifest = json.loads((source / 'index.json').read_text(encoding='utf-8'))
    raw = []
    for relative in input_manifest['detail_files'].values():
        detail = json.loads((source / relative).read_text(encoding='utf-8'))
        raw.append(detail.get('evaluation') or {
            'id': detail['name'], 'question': detail['query'],
            'retrieval_query': detail['retrieval_query'], 'query_analysis': detail['query_analysis'],
            'gold': detail['gold'], 'retrieval': {'semantic_results': detail['routes']['semantic']}})
    data = json.dumps(input_manifest, ensure_ascii=False).encode('utf-8')
    dataset = output / 'dataset'
    details = dataset / 'details'
    details.mkdir(parents=True)
    index_manifest = {'schema_version': 1, 'batch_id': uuid4().hex,
                      'input_batch_id': input_manifest.get('batch_id'),
                      'items': [], 'detail_files': {}}
    prepared = [(row, prepare(row)) for row in raw]
    mapped = [(r, q) for r, q in prepared if q["gold"]]
    excluded = len(raw) - len(mapped)
    if args.limit:
        mapped = mapped[:args.limit]
    if not mapped:
        raise ValueError("No mapped gold queries")
    manifest = json.loads((args.run_dir / "store_manifest.json").read_text())
    if manifest.get("backend") != "qdrant":
        raise ValueError("This experiment requires an existing Qdrant collection")
    embedder = EmbeddingModel(args.model_path, device=args.device,
                              dimension=manifest.get("vector_dim", 1024), mock=False)
    store = VectorStore(run_dir=args.run_dir, backend="qdrant",
                        qdrant_url=manifest["qdrant_url"],
                        collection=manifest["collection"],
                        vector_dim=manifest.get("vector_dim", 1024))
    retriever = VectorRetriever(embeddings=embedder, vector_store=store)
    records = []
    parity = {"checked": 0, "exact_order": 0}
    start = time.perf_counter()
    for index, (row, query) in enumerate(mapped, 1):
        all_texts = unique([*query["texts"].values(), *query["component_texts"]])
        vectors = dict(zip(all_texts, embedder.encode(all_texts)))
        cache = {}
        def search(text=None, vector=None):
            if text is not None and text in cache:
                return cache[text]
            hits = retriever.search(
                query["parts"], args.pool, original_query=query["original"],
                query_vector=vector if vector is not None else
                    VectorRetriever._unit_vector(vectors[text]),
                spot_names=query["spots"])
            if text is not None:
                cache[text] = hits
            return hits
        rankings = {name: search(text) for name, text in query["texts"].items()}
        components = query["component_texts"]
        weights = [2.0 if t == query["original"] else 1.0 for t in components]
        rankings["original_subqueries_centroid_w2"] = search(
            vector=centroid([vectors[t] for t in components], weights))
        rankings["original_subqueries_rrf"] = rrf(
            [search(t) for t in components], args.pool)
        baseline = rankings["subqueries_concat"]
        saved = (row.get("retrieval") or {}).get("semantic_results")
        parity_equal = None
        if saved is not None:
            # Saved lists may be truncated; compare their entire available prefix.
            saved_ids = [str(h["chunk_id"]) for h in saved]
            fresh_ids = [str(h["chunk_id"]) for h in baseline]
            parity_equal = fresh_ids[:len(saved_ids)] == saved_ids
            parity["checked"] += 1
            parity["exact_order"] += int(parity_equal)
        record = {
            "id": row.get("id", str(index)), "original_query": query["original"],
            "subqueries": query["parts"], "subquery_source": query["subquery_source"],
            "empty_subqueries_fallback": query["empty_subqueries_fallback"],
            "gold_chunk_ids": query["gold"], "spot_names": query["spots"],
            "saved_baseline_prefix_equal": parity_equal, "strategies": {},
        }
        for name in STRATEGIES:
            hits = rankings[name]
            record["strategies"][name] = {
                "query_text": query["texts"].get(name),
                "component_texts": components if name not in query["texts"] else None,
                "metrics": route_metrics(hits, query["gold"], KS, query["relevance"]),
                "results": [{"chunk_id": str(h["chunk_id"]), "rank": i,
                             "score": float(h["score"])} for i, h in enumerate(hits, 1)],
            }
        name = f'query_{index:06d}.json'
        record['name'] = name
        record['batch_id'] = index_manifest['batch_id']
        record['gold_relevance'] = query['relevance']
        (details / name).write_text(json.dumps(record, ensure_ascii=False) + '\n', encoding='utf-8')
        index_manifest['items'].append({'name': name, 'query': query['original']})
        index_manifest['detail_files'][name] = 'details/' + name
        records.append(record)
        if index % 25 == 0 or index == len(mapped):
            print(f"[query-composition] {index}/{len(mapped)} elapsed={time.perf_counter()-start:.1f}s", flush=True)
    summary = {
        "input_results": str(args.input_dataset), "input_sha256": hashlib.sha256(data).hexdigest(),
        "run_dir": str(args.run_dir), "collection": manifest["collection"],
        "qdrant_url": manifest["qdrant_url"], "model": args.model_path, "device": args.device,
        "encoder_mode": embedder._mode, "input_count": len(raw),
        "excluded_unmapped_count": excluded, "evaluated_count": len(records),
        "empty_subqueries_count": sum(r["empty_subqueries_fallback"] for r in records),
        "pool": args.pool, "ks": KS, "smoke_only": bool(args.limit),
        "baseline_parity": parity, "strategy_metrics": summarize(records),
        "seconds": time.perf_counter()-start,
        "note": "Exploratory comparison on saved inputs; no new LLM expansion. RRF uses one search per unique component; other methods use one search.",
    }
    index_manifest['configuration'] = {k: v for k, v in summary.items() if k != 'strategy_metrics'}
    (dataset / 'index.json').write_text(json.dumps(index_manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summarize(records), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

