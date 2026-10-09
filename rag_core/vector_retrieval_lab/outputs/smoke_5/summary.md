# Dense vector query strategy comparison

- Mapped queries: 5 / saved main evaluation
- Collection: rag_core_sdu_v4pro_full_20260921_run1
- Embedding: /home/humq/rag_db_silm/model/bge-m3 on cpu
- Baseline pool: 20; multi-query candidate pool: 100; reported depth: 20
- Saved-baseline vs fresh concatenation exact order parity: 5/5

| Strategy | Hit@5 | Hit@10 | Hit@15 | Gold Recall@15 | MRR@10 | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| saved_baseline | 100.00% | 100.00% | 100.00% | 100.00% | 0.4467 | 0.5197 |
| concat_current | 100.00% | 100.00% | 100.00% | 100.00% | 0.4467 | 0.5197 |
| subquery_rrf_k20_pool100 | 100.00% | 100.00% | 100.00% | 100.00% | 0.3833 | 0.4177 |
| subquery_rrf_k60_pool100 | 100.00% | 100.00% | 100.00% | 100.00% | 0.3833 | 0.4177 |
| subquery_rrf_k100_pool100 | 100.00% | 100.00% | 100.00% | 100.00% | 0.3833 | 0.4177 |
| subqueries_comma_joined | 80.00% | 100.00% | 100.00% | 100.00% | 0.3733 | 0.3772 |
| question_plus_subquery_rrf_k60_pool100 | 100.00% | 100.00% | 100.00% | 100.00% | 0.3733 | 0.4598 |
| question_plus_subquery_rrf_k20_pool100 | 100.00% | 100.00% | 100.00% | 100.00% | 0.3733 | 0.4598 |
| question_plus_subquery_minmax_pool100 | 100.00% | 100.00% | 100.00% | 100.00% | 0.3733 | 0.4123 |
| question_plus_subquery_vector_centroid | 100.00% | 100.00% | 100.00% | 100.00% | 0.3733 | 0.4123 |
| subquery_vector_centroid | 100.00% | 100.00% | 100.00% | 100.00% | 0.3567 | 0.4512 |
| question_only | 80.00% | 100.00% | 100.00% | 100.00% | 0.3333 | 0.4016 |
| joined_plus_subquery_rrf_k60_pool100 | 100.00% | 100.00% | 100.00% | 100.00% | 0.3233 | 0.3776 |

Per-query top rankings are in per_query_rankings.jsonl.
