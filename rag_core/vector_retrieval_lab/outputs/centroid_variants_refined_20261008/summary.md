# One-search centroid aggregation sweep

- Queries: 402; Qdrant pool: 20; output depth: 20
- Grouped 5-fold selected metrics: Hit@15 94.53%; MRR@10 0.6651; nDCG@5 0.6631
- CV selected by fold: {'0': 'fixed_question_mass_30', '1': 'fixed_question_mass_32', '2': 'fixed_question_mass_32', '3': 'fixed_question_mass_32', '4': 'fixed_question_mass_32'}

| Strategy | Hit@15 | MRR@10 | nDCG@5 |
|---|---:|---:|---:|
| fixed_question_mass_32 | 94.78% | 0.6639 | 0.6631 |
| fixed_question_mass_33 | 94.78% | 0.6638 | 0.6631 |
| fixed_question_mass_34 | 94.78% | 0.6624 | 0.6623 |
| fixed_question_mass_30 | 94.53% | 0.6651 | 0.6631 |
| fixed_question_mass_50 | 94.28% | 0.6697 | 0.6699 |
| fixed_question_mass_38 | 94.28% | 0.6689 | 0.6674 |
| fixed_question_mass_80 | 94.28% | 0.6687 | 0.6694 |
| fixed_question_mass_40 | 94.28% | 0.6685 | 0.6673 |
| fixed_question_mass_36 | 94.28% | 0.6625 | 0.6630 |
| fixed_question_mass_25 | 94.28% | 0.6620 | 0.6614 |
| fixed_question_mass_15 | 94.28% | 0.6587 | 0.6556 |
| fixed_question_mass_60 | 94.03% | 0.6743 | 0.6740 |
| question50_subquery_diversity_weighted | 94.03% | 0.6702 | 0.6694 |
| question50_subquery_cosine_softmax_t25 | 94.03% | 0.6688 | 0.6676 |
| fixed_question_mass_70 | 93.78% | 0.6704 | 0.6725 |
| question50_subquery_cosine_softmax_t10 | 93.78% | 0.6646 | 0.6641 |
