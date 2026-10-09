# One/two-search dense vector strategies

- Queries: 402 mapped; per-search candidate pool: 50
- Baseline reproduction and single-centroid results are copied from the validated first sweep.
- Grouped 5-fold CV selected metrics: Hit@15 94.28%; MRR@10 0.6680; nDCG@5 0.6659
- CV selected methods by fold: {'0': 'dual_centroid_question_pool50_rrf_k20_equal', '1': 'dual_centroid_question_pool50_rrf_k60_equal', '2': 'centroid_one_search', '3': 'dual_centroid_question_pool50_rrf_k60_equal', '4': 'dual_centroid_question_pool50_rrf_k60_equal'}

| Search count | Strategy | Hit@5 | Hit@10 | Hit@15 | Gold Recall@15 | MRR@10 | nDCG@5 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | dual_centroid_question_pool50_rrf_k60_equal | 83.83% | 91.29% | 94.53% | 90.04% | 0.6688 | 0.6684 |
| 2 | dual_centroid_question_pool50_rrf_k20_equal | 83.58% | 91.29% | 94.53% | 90.04% | 0.6687 | 0.6679 |
| 1 | centroid_one_search | 84.33% | 91.29% | 94.53% | 90.23% | 0.6646 | 0.6646 |
| 1 | centroid_pool50_one_search | 84.33% | 91.29% | 94.53% | 90.23% | 0.6646 | 0.6646 |
| 2 | dual_centroid_question_pool50_rrf_k20_c2s1 | 83.83% | 90.80% | 94.28% | 90.10% | 0.6665 | 0.6676 |
| 2 | dual_centroid_question_pool50_rrf_k60_c2s1 | 83.83% | 90.80% | 94.28% | 90.10% | 0.6665 | 0.6676 |
| 1 | single_question_pool50 | 82.84% | 90.80% | 94.03% | 89.77% | 0.6683 | 0.6676 |
| 2 | dual_centroid_joined_pool50_rrf_k60_c2s1 | 83.33% | 90.80% | 94.03% | 90.00% | 0.6627 | 0.6617 |
| 2 | dual_centroid_joined_pool50_rrf_k20_c2s1 | 83.08% | 90.80% | 94.03% | 90.12% | 0.6625 | 0.6607 |
| 2 | dual_centroid_question_pool50_minmax_equal | 84.08% | 91.04% | 93.78% | 89.48% | 0.6733 | 0.6749 |
| 2 | dual_centroid_question_pool50_rrf_k60_s2 | 83.58% | 91.04% | 93.78% | 89.38% | 0.6672 | 0.6673 |
| 2 | dual_centroid_joined_pool50_rrf_k20_equal | 82.34% | 90.80% | 93.78% | 89.88% | 0.6635 | 0.6568 |
| 2 | dual_centroid_novel_pool50_rrf_k20_c2s1 | 82.84% | 91.04% | 93.78% | 89.48% | 0.6463 | 0.6442 |
| 2 | dual_centroid_novel_pool50_rrf_k60_c2s1 | 82.59% | 91.04% | 93.78% | 89.61% | 0.6458 | 0.6435 |
| 2 | dual_centroid_joined_pool50_minmax_equal | 82.34% | 91.29% | 93.53% | 89.65% | 0.6633 | 0.6580 |
| 2 | dual_centroid_novel_pool50_rrf_k20_equal | 80.85% | 90.30% | 93.53% | 89.03% | 0.6302 | 0.6275 |
| 2 | dual_centroid_joined_pool50_rrf_k60_equal | 82.34% | 90.80% | 93.28% | 89.50% | 0.6632 | 0.6560 |
| 2 | dual_centroid_novel_pool50_rrf_k60_equal | 80.60% | 90.30% | 93.28% | 88.71% | 0.6299 | 0.6267 |
| 2 | dual_centroid_joined_pool50_rrf_k60_s2 | 81.09% | 90.05% | 93.03% | 89.32% | 0.6463 | 0.6414 |
| 2 | dual_centroid_novel_pool50_minmax_equal | 80.60% | 90.05% | 92.79% | 88.42% | 0.6445 | 0.6396 |
| 2 | dual_centroid_novel_pool50_rrf_k60_s2 | 78.11% | 88.81% | 92.54% | 87.62% | 0.5872 | 0.5895 |
| 1 | single_joined_pool50 | 78.61% | 89.05% | 91.54% | 88.04% | 0.6284 | 0.6172 |
| 1 | saved_baseline_one_search | 78.36% | 88.81% | 91.29% | 87.71% | 0.6278 | 0.6161 |
| 1 | single_novel_pool50 | 74.63% | 87.56% | 91.04% | 85.98% | 0.5628 | 0.5581 |

Best full-set strategy: dual_centroid_question_pool50_rrf_k60_equal (2 searches). Hit@15 changes vs baseline: {'new_hit15': 16, 'lost_hit15': 3, 'same_hit15': 383}.
