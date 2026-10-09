# Offline two-search rank-fusion refinement

- Queries: 402; fused inputs contain the top 50 candidates from each retrieval.
- Grouped 5-fold selected metrics: Hit@15 94.78%; MRR@10 0.6614; nDCG@5 0.6631
- CV selected by fold: {'0': 'dual_current_question_weight_2_centroid__fixed_question_mass_33_rrf_k0_left2', '1': 'dual_current_question_weight_2_centroid__fixed_question_mass_33_rrf_k0_left1', '2': 'dual_current_question_weight_2_centroid__fixed_question_mass_33_rrf_k0_left0.5', '3': 'dual_current_question_weight_2_centroid__fixed_question_mass_33_rrf_k0_left2', '4': 'dual_current_question_weight_2_centroid__fixed_question_mass_33_borda_left0.5'}

| Searches | Strategy | Hit@15 | MRR@10 | nDCG@5 |
|---:|---|---:|---:|---:|
| 2 | dual_current_question_weight_2_centroid__fixed_question_mass_33_rrf_k0_left2 | 94.78% | 0.6644 | 0.6645 |
| 2 | dual_current_question_weight_2_centroid__fixed_question_mass_33_rrf_k5_left2 | 94.78% | 0.6644 | 0.6645 |
| 2 | dual_current_question_weight_2_centroid__fixed_question_mass_33_rrf_k20_left2 | 94.78% | 0.6644 | 0.6645 |
| 2 | dual_current_question_weight_2_centroid__fixed_question_mass_33_rrf_k60_left2 | 94.78% | 0.6644 | 0.6645 |
| 2 | dual_current_question_weight_2_centroid__fixed_question_mass_33_borda_left0.5 | 94.78% | 0.6641 | 0.6633 |
| 2 | dual_current_question_weight_2_centroid__fixed_question_mass_33_rrf_k0_left0.5 | 94.78% | 0.6641 | 0.6633 |
| 2 | dual_current_question_weight_2_centroid__fixed_question_mass_33_rrf_k5_left0.5 | 94.78% | 0.6641 | 0.6633 |
| 2 | dual_current_question_weight_2_centroid__fixed_question_mass_33_rrf_k20_left0.5 | 94.78% | 0.6641 | 0.6633 |
| 2 | dual_current_question_weight_2_centroid__fixed_question_mass_33_rrf_k60_left0.5 | 94.78% | 0.6641 | 0.6633 |
| 2 | dual_current_question_weight_2_centroid__fixed_question_mass_33_rrf_k0_left1 | 94.78% | 0.6640 | 0.6643 |
| 2 | dual_current_question_weight_2_centroid__fixed_question_mass_33_rrf_k5_left1 | 94.78% | 0.6640 | 0.6643 |
| 2 | dual_current_question_weight_2_centroid__fixed_question_mass_33_rrf_k20_left1 | 94.78% | 0.6640 | 0.6643 |
| 2 | dual_current_question_weight_2_centroid__fixed_question_mass_33_rrf_k60_left1 | 94.78% | 0.6640 | 0.6643 |
| 2 | dual_current_question_weight_2_centroid__fixed_question_mass_33_borda_left1 | 94.78% | 0.6640 | 0.6643 |
| 2 | dual_current_question_weight_2_centroid__fixed_question_mass_33_borda_left2 | 94.53% | 0.6646 | 0.6646 |
| 2 | dual_fixed_question_mass_33__fixed_question_mass_60_rrf_k0_left0.5 | 94.28% | 0.6737 | 0.6737 |
| 2 | dual_fixed_question_mass_33__fixed_question_mass_60_rrf_k5_left0.5 | 94.28% | 0.6724 | 0.6737 |
| 2 | dual_fixed_question_mass_33__fixed_question_mass_60_rrf_k20_left0.5 | 94.28% | 0.6723 | 0.6737 |
| 2 | dual_fixed_question_mass_33__fixed_question_mass_60_rrf_k60_left0.5 | 94.28% | 0.6723 | 0.6737 |
| 2 | dual_fixed_question_mass_33__fixed_question_mass_60_borda_left0.5 | 94.28% | 0.6713 | 0.6736 |
