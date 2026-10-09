# One-search centroid aggregation sweep

- Queries: 15; Qdrant pool: 20; output depth: 20
- Grouped 5-fold selected metrics: Hit@15 100.00%; MRR@10 0.5167; nDCG@5 0.6094
- CV selected by fold: {'0': 'question50_subquery_diversity_weighted', '1': 'fixed_question_mass_25', '2': 'question50_subquery_diversity_weighted', '3': 'question50_subquery_diversity_weighted', '4': 'question50_subquery_diversity_weighted'}

| Strategy | Hit@15 | MRR@10 | nDCG@5 |
|---|---:|---:|---:|
| question50_subquery_diversity_weighted | 100.00% | 0.5278 | 0.6181 |
| fixed_question_mass_25 | 100.00% | 0.5189 | 0.5999 |
| fixed_question_mass_33 | 100.00% | 0.5189 | 0.5999 |
| fixed_question_mass_50 | 100.00% | 0.5167 | 0.6094 |
| fixed_question_mass_15 | 100.00% | 0.5133 | 0.5917 |
| fixed_question_mass_70 | 100.00% | 0.5056 | 0.5994 |
| fixed_question_mass_60 | 100.00% | 0.5000 | 0.5994 |
| fixed_question_mass_40 | 100.00% | 0.4856 | 0.6006 |
| fixed_question_mass_80 | 100.00% | 0.4722 | 0.5748 |
| question50_subquery_cosine_softmax_t25 | 100.00% | 0.4444 | 0.5193 |
| question50_subquery_cosine_softmax_t10 | 100.00% | 0.4333 | 0.5106 |
