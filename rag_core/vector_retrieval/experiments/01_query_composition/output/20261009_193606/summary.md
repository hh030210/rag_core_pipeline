# Query composition experiment

Mapped queries: 402; excluded unmapped: 33; pool: 20.

Smoke only: False. Saved baseline prefix parity: {'checked': 402, 'exact_order': 402}.

| Strategy | Hit@1 | Hit@5 | Hit@10 | Hit@15 | Hit@20 | Gold Recall@15 | MRR@10 | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original_only | 0.5522 | 0.8284 | 0.9080 | 0.9403 | 0.9527 | 0.8977 | 0.6683 | 0.6676 |
| subqueries_concat | 0.5025 | 0.7836 | 0.8881 | 0.9129 | 0.9403 | 0.8771 | 0.6278 | 0.6161 |
| original_plus_subqueries_concat | 0.5000 | 0.7811 | 0.8831 | 0.9154 | 0.9303 | 0.8800 | 0.6273 | 0.6175 |
| original_plus_subqueries_dedup | 0.5025 | 0.7861 | 0.8856 | 0.9129 | 0.9303 | 0.8775 | 0.6299 | 0.6203 |
| subqueries_comma | 0.5025 | 0.7861 | 0.8905 | 0.9204 | 0.9428 | 0.8864 | 0.6272 | 0.6189 |
| original_subqueries_centroid_w2 | 0.5348 | 0.8433 | 0.9129 | 0.9453 | 0.9552 | 0.9023 | 0.6646 | 0.6646 |
| original_subqueries_rrf | 0.5050 | 0.8259 | 0.9055 | 0.9403 | 0.9577 | 0.8992 | 0.6427 | 0.6431 |
RRF makes multiple searches. Results are exploratory, not an independent holdout.
