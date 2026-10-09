# Dense vector query strategy comparison

- Mapped queries: 402 / saved main evaluation
- Collection: rag_core_sdu_v4pro_full_20260921_run1
- Embedding: /home/humq/rag_db_silm/model/bge-m3 on cpu
- Baseline pool: 20; multi-query candidate pool: 100; reported depth: 20
- Saved-baseline vs fresh concatenation exact order parity: 402/402

| Strategy | Hit@5 | Hit@10 | Hit@15 | Gold Recall@15 | MRR@10 | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| question_plus_subquery_vector_centroid | 84.33% | 91.29% | 94.53% | 90.23% | 0.6646 | 0.6646 |
| subquery_rrf_k20_pool100 | 82.09% | 90.80% | 94.53% | 90.19% | 0.6343 | 0.6339 |
| question_plus_subquery_rrf_k20_pool100 | 83.83% | 91.54% | 94.28% | 90.28% | 0.6584 | 0.6605 |
| subquery_rrf_k100_pool100 | 81.84% | 90.30% | 94.28% | 90.19% | 0.6326 | 0.6323 |
| subquery_rrf_k60_pool100 | 81.84% | 90.05% | 94.28% | 90.44% | 0.6325 | 0.6323 |
| question_only | 82.84% | 90.80% | 94.03% | 89.77% | 0.6683 | 0.6676 |
| question_plus_subquery_minmax_pool100 | 83.33% | 91.04% | 94.03% | 90.10% | 0.6640 | 0.6611 |
| question_plus_subquery_rrf_k60_pool100 | 83.83% | 91.54% | 94.03% | 89.90% | 0.6536 | 0.6573 |
| subquery_vector_centroid | 81.59% | 90.80% | 93.78% | 89.77% | 0.6566 | 0.6493 |
| joined_plus_subquery_rrf_k60_pool100 | 81.84% | 90.80% | 93.03% | 89.13% | 0.6464 | 0.6411 |
| subqueries_comma_joined | 78.61% | 89.05% | 92.04% | 88.64% | 0.6272 | 0.6189 |
| saved_baseline | 78.36% | 88.81% | 91.29% | 87.71% | 0.6278 | 0.6161 |
| concat_current | 78.36% | 88.81% | 91.29% | 87.71% | 0.6278 | 0.6161 |

Per-query top rankings are in per_query_rankings.jsonl.
