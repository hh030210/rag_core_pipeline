# Query composition experiment

Mapped queries: 5; excluded unmapped: 33; pool: 20.

Smoke only: True. Saved baseline prefix parity: {'checked': 5, 'exact_order': 5}.

| Strategy | Hit@1 | Hit@5 | Hit@10 | Hit@15 | Hit@20 | Gold Recall@15 | MRR@10 | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original_only | 0.0000 | 0.8000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.3333 | 0.4016 |
| subqueries_concat | 0.2000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.4467 | 0.5197 |
| original_plus_subqueries_concat | 0.2000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.4467 | 0.5251 |
| original_plus_subqueries_dedup | 0.2000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.4467 | 0.5251 |
| subqueries_comma | 0.0000 | 0.8000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.3733 | 0.3772 |
| original_subqueries_centroid_w2 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.3733 | 0.4123 |
| original_subqueries_rrf | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.3733 | 0.4123 |
RRF makes multiple searches. Results are exploratory, not an independent holdout.
