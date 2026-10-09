# Adaptive centroid exact replay

- Queries: 402; Qdrant searches/query: 1; pool: 20; top-20
- Exact order parity with precomputed component retrievals: 402/402

| Strategy | Hit@1 | Hit@5 | Hit@10 | Hit@15 | Gold Recall@15 | MRR@10 | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive_threshold_39 | 54.98% | 84.33% | 91.54% | 94.78% | 90.48% | 0.6730 | 0.6711 |
| saved_concatenated_baseline | 50.25% | 78.36% | 88.81% | 91.29% | 87.71% | 0.6278 | 0.6161 |
| current_question_weight_2_centroid | 53.48% | 84.33% | 91.29% | 94.53% | 90.23% | 0.6646 | 0.6646 |
