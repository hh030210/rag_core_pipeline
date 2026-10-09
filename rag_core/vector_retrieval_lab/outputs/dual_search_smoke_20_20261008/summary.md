# One/two-search dense vector strategies

- Queries: 20 mapped; per-search candidate pool: 50
- Baseline reproduction and single-centroid results are copied from the validated first sweep.
- Grouped 5-fold CV selected metrics: Hit@15 100.00%; MRR@10 0.5658; nDCG@5 0.6454
- CV selected methods by fold: {'0': 'dual_centroid_novel_pool50_rrf_k20_c2s1', '1': 'dual_centroid_joined_pool50_rrf_k20_equal', '2': 'dual_centroid_novel_pool50_rrf_k20_c2s1', '3': 'dual_centroid_novel_pool50_rrf_k20_c2s1', '4': 'dual_centroid_novel_pool50_rrf_k20_c2s1'}

| Search count | Strategy | Hit@5 | Hit@10 | Hit@15 | Gold Recall@15 | MRR@10 | nDCG@5 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | dual_centroid_novel_pool50_rrf_k20_c2s1 | 100.00% | 100.00% | 100.00% | 100.00% | 0.6033 | 0.6707 |
| 2 | dual_centroid_novel_pool50_rrf_k60_c2s1 | 100.00% | 100.00% | 100.00% | 100.00% | 0.6033 | 0.6707 |
| 2 | dual_centroid_joined_pool50_rrf_k20_equal | 95.00% | 100.00% | 100.00% | 100.00% | 0.5896 | 0.6381 |
| 2 | dual_centroid_joined_pool50_rrf_k60_equal | 95.00% | 100.00% | 100.00% | 100.00% | 0.5896 | 0.6381 |
| 2 | dual_centroid_novel_pool50_rrf_k20_equal | 95.00% | 100.00% | 100.00% | 100.00% | 0.5738 | 0.6421 |
| 2 | dual_centroid_novel_pool50_rrf_k60_equal | 95.00% | 100.00% | 100.00% | 100.00% | 0.5738 | 0.6421 |
| 2 | dual_centroid_joined_pool50_rrf_k20_c2s1 | 95.00% | 100.00% | 100.00% | 100.00% | 0.5671 | 0.6330 |
| 2 | dual_centroid_joined_pool50_rrf_k60_c2s1 | 95.00% | 100.00% | 100.00% | 100.00% | 0.5671 | 0.6330 |
| 2 | dual_centroid_joined_pool50_minmax_equal | 95.00% | 100.00% | 100.00% | 100.00% | 0.5671 | 0.6207 |
| 2 | dual_centroid_joined_pool50_rrf_k60_s2 | 90.00% | 100.00% | 100.00% | 100.00% | 0.5656 | 0.6141 |
| 2 | dual_centroid_novel_pool50_minmax_equal | 95.00% | 100.00% | 100.00% | 100.00% | 0.5642 | 0.6295 |
| 2 | dual_centroid_question_pool50_rrf_k20_c2s1 | 90.00% | 100.00% | 100.00% | 100.00% | 0.5625 | 0.6320 |
| 2 | dual_centroid_question_pool50_rrf_k60_c2s1 | 90.00% | 100.00% | 100.00% | 100.00% | 0.5625 | 0.6320 |
| 2 | dual_centroid_question_pool50_rrf_k20_equal | 95.00% | 100.00% | 100.00% | 100.00% | 0.5600 | 0.6445 |
| 2 | dual_centroid_question_pool50_rrf_k60_equal | 95.00% | 100.00% | 100.00% | 100.00% | 0.5600 | 0.6445 |
| 2 | dual_centroid_question_pool50_minmax_equal | 95.00% | 100.00% | 100.00% | 100.00% | 0.5517 | 0.6418 |
| 1 | centroid_one_search | 95.00% | 100.00% | 100.00% | 100.00% | 0.5475 | 0.6201 |
| 1 | centroid_pool50_one_search | 95.00% | 100.00% | 100.00% | 100.00% | 0.5475 | 0.6201 |
| 2 | dual_centroid_question_pool50_rrf_k60_s2 | 95.00% | 100.00% | 100.00% | 100.00% | 0.4908 | 0.5961 |
| 1 | single_question_pool50 | 95.00% | 100.00% | 100.00% | 100.00% | 0.4783 | 0.5781 |
| 1 | saved_baseline_one_search | 85.00% | 100.00% | 100.00% | 100.00% | 0.4767 | 0.5350 |
| 1 | single_joined_pool50 | 85.00% | 100.00% | 100.00% | 100.00% | 0.4767 | 0.5350 |
| 2 | dual_centroid_novel_pool50_rrf_k60_s2 | 95.00% | 100.00% | 100.00% | 100.00% | 0.4671 | 0.5555 |
| 1 | single_novel_pool50 | 85.00% | 100.00% | 100.00% | 100.00% | 0.4167 | 0.4893 |

Best full-set strategy: dual_centroid_novel_pool50_rrf_k20_c2s1 (2 searches). Hit@15 changes vs baseline: {'new_hit15': 0, 'lost_hit15': 0, 'same_hit15': 20}.
