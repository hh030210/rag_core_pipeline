# Isolated dense-vector retrieval lab

This is an isolated copy of the semantic/vector retrieval route. It does not modify the main `rag_core/retrieval.py`, the production Qdrant collection, the original run directory, or main evaluation outputs.

## Copied behavior

- `embedding.py` is copied from `rag_core/embedding.py` (BGE-M3 dense encoding).
- `vector_search.py` reproduces the semantic branch in `rag_core/retrieval.py` and `VectorStore.semantic_search`: encode a query, search Qdrant's `chunk_text_vec`, then apply the same scenic-spot post-filter.
- Experiments read saved query expansions, mapped gold chunk IDs, spot names, the collection manifest, and the existing Qdrant collection. New files are written here.

## First improvement: weighted centroid (preserved control)

The current baseline embeds all subqueries joined by ` | ` and makes one dense search. The first improved strategy embeds the original question and every subquery separately, L2-normalizes each vector, forms one weighted centroid, then performs one Qdrant search:

`normalize(2 * normalize(question_vector) + sum(normalize(subquery_vector_i)))`

The original question anchors overall intent with weight 2; each subquery has weight 1. Both methods use the same collection, BGE-M3 model, pool depth 20, and scenic-spot filter.

## Main-experiment results

There were 402 mapped queries. The fresh concatenation baseline reproduced the saved candidate order exactly on all 402 queries.

| Metric | Existing concatenated query | Weighted vector centroid | Change |
|---|---:|---:|---:|
| Hit@1 | 50.25% | 53.48% | +3.23 pp |
| Hit@5 | 78.36% | 84.33% | +5.97 pp |
| Hit@10 | 88.81% | 91.29% | +2.48 pp |
| Hit@15 | 91.29% | **94.53%** | **+3.24 pp** |
| Gold Recall@15 | 87.71% | 90.23% | +2.52 pp |
| MRR@10 | 0.6278 | 0.6646 | +0.0367 |
| nDCG@5 | 0.6161 | 0.6646 | +0.0485 |

At Hit@15, 14 queries changed from miss to hit and 1 changed from hit to miss: net +13 (367 to 380 of 402). A grouped 5-fold sweep tested original-query weights 0, 0.5, 1, 2, 4, and 8; every fold selected weight 2. The fold-held-out predictions pooled across the five folds also scored 94.53% Hit@15.

Subquery RRF also reached 94.53% Hit@15, but had lower MRR@10 (0.6343) and nDCG@5 (0.6339). The weighted centroid was the stronger overall ranking strategy in this run.

## Reproduce

Run from the repository root using the same Python environment:

    cd /home/humq/rag_core_pipeline
    PYTHONDONTWRITEBYTECODE=1 /home/humq/envs/denoise_qa/bin/python -m rag_core.vector_retrieval_lab.experiment --output-dir rag_core/vector_retrieval_lab/outputs/repeat_full
    PYTHONDONTWRITEBYTECODE=1 /home/humq/envs/denoise_qa/bin/python -m rag_core.vector_retrieval_lab.centroid_sweep --output-dir rag_core/vector_retrieval_lab/outputs/repeat_centroid_sweep

The first script compares concatenation, original-question-only search, comma-joined subqueries, RRF, min-max score fusion, and vector centroids. The second tunes centroid weights with grouped 5-fold validation. `--limit 5` is available for a smoke run; do not use it for final metrics.

Results from this run are in `outputs/full_20261008/` and `outputs/centroid_sweep_full_20261008/`. Per-query rankings and aggregate summaries are saved there.

## Limits

This is an offline comparison on the saved mapped evaluation set, not an independent new test set. Validate the selected strategy on a fresh evaluation set before adopting it in the main pipeline. The installed FlagEmbedding path emitted a compatibility warning and fell back to SentenceTransformer; exact 402/402 candidate-order parity against the saved baseline confirms that this run reproduced the existing encoder behavior for the evaluated queries.

## Two-search follow-up

A second experiment tested whether a centroid search followed by one additional search could improve ranking. Each two-search strategy retrieves up to 50 candidates for the weighted centroid and up to 50 for a second query (the original question, joined subqueries, or the most novel subquery). The two ranked lists are merged with reciprocal-rank fusion or min-max score fusion. The candidate pool is then truncated to 20 results and the same spot filter is applied.

The best full-set two-search variant used the weighted centroid plus the original question, merged by equal-weight RRF (`k=60`). On the 402-query set it tied the centroid's Hit@15 at 94.53%, with MRR@10 0.6688 and nDCG@5 0.6684. Since this variant was selected from the same full set, treat these figures as exploratory.

Grouped 5-fold selection gives a more conservative comparison:

| Strategy | Searches/query | Hit@15 | MRR@10 | nDCG@5 |
|---|---:|---:|---:|---:|
| Concatenated-query baseline | 1 | 91.29% | 0.6278 | 0.6161 |
| Weighted centroid | 1 | **94.53%** | 0.6646 | 0.6646 |
| CV-selected two-search fusion | 2 | 94.28% | **0.6680** | **0.6659** |

For this pair of retrievals, two-search fusion did not improve held-out Hit@15 over the one-search centroid. A follow-up fusion sweep used top-50 candidates from several centroid variants; its CV result was Hit@15 94.78%, MRR@10 0.6614, and nDCG@5 0.6631. The adaptive one-search policy below matches that Hit@15 and ranks better on MRR/nDCG at half the Qdrant calls. The fusion sweep is saved under outputs/rank_fusion_pool50_20261008/.

Reproduce the full follow-up with:

    PYTHONDONTWRITEBYTECODE=1 /home/humq/envs/denoise_qa/bin/python -m rag_core.vector_retrieval_lab.dual_search_sweep --output-dir rag_core/vector_retrieval_lab/outputs/repeat_dual_search

## Current best candidate: adaptive one-search centroid

The follow-up sweep found a better-balanced single-search policy. It preserves the concatenation route and the earlier question-weight-2 centroid as controls. For each query, count characters across unique subqueries, excluding exact copies of the original question:

- At 39 characters or fewer, give 60% of the centroid weight to the original question and divide 40% equally across subqueries.
- Above 39 characters, give 32% to the original question and divide 68% equally across subqueries.
- L2-normalize each encoded vector, compose and normalize the centroid, then make one Qdrant search.

This adapts the balance between the user's question and the expanded aspects without an extra retrieval call. The simple threshold is supported by grouped 5-fold selection: all five training folds chose the same direction, with cutoffs between 38.5 and 41.5 characters. The held-out predictions were:

| Strategy | Hit@1 | Hit@5 | Hit@10 | Hit@15 | Gold Recall@15 | MRR@10 | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Existing question-weight-2 centroid | 53.48% | 84.33% | 91.29% | 94.53% | 90.23% | 0.6646 | 0.6646 |
| Adaptive centroid, grouped 5-fold | **54.73%** | **84.33%** | **91.54%** | **94.78%** | **90.48%** | **0.6717** | **0.6700** |

The fixed 39-character policy was then replayed through the actual Qdrant search path on all 402 mapped queries. It used one search per query and matched the precomputed 60%/32% component rankings exactly for 402/402 queries. The replay scored Hit@1 54.98%, Hit@5 84.33%, Hit@10 91.54%, Hit@15 94.78%, Gold Recall@15 90.48%, MRR@10 0.6730, and nDCG@5 0.6711. These replay numbers use the same mapped evaluation set; use the grouped 5-fold figures as the more conservative comparison and validate on fresh queries before production adoption.

Reusable logic is in adaptive_centroid.py; adaptive_search(...) returns the filtered hits and the selected weight/character count. The search implementation and all new scripts remain in this isolated folder.

Reproduce the final fixed-threshold replay with:

    PYTHONDONTWRITEBYTECODE=1 /home/humq/envs/denoise_qa/bin/python -m rag_core.vector_retrieval_lab.adaptive_centroid_replay --output-dir rag_core/vector_retrieval_lab/outputs/repeat_adaptive_replay

The adaptive-policy sweep, constrained threshold validation, and exact replay are saved under outputs/adaptive_centroid_cv_20261008/, outputs/char_gate_h5_constrained_cv_20261008/, and outputs/adaptive_replay_threshold39_20261008/.
