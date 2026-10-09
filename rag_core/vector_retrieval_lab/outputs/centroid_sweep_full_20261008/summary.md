# Query vector centroid weight sweep

- Mapped queries: 402; Qdrant pool: 20; folds: 5
- Saved baseline: Hit@15 91.29%, MRR@10 0.6278, nDCG@5 0.6161
- CV weight selections by fold: {'0': 2.0, '1': 2.0, '2': 2.0, '3': 2.0, '4': 2.0}
- CV selected-weight score: Hit@15 94.53%, MRR@10 0.6646, nDCG@5 0.6646

| Original-query weight | Hit@5 | Hit@10 | Hit@15 | Gold Recall@15 | MRR@10 | nDCG@5 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 78.61% | 86.82% | 90.05% | 86.00% | 0.6332 | 0.6260 |
| 0.5 | 82.59% | 90.80% | 94.03% | 89.85% | 0.6602 | 0.6562 |
| 1 | 83.58% | 91.29% | 94.28% | 90.10% | 0.6631 | 0.6617 |
| 2 | 84.33% | 91.29% | 94.53% | 90.23% | 0.6646 | 0.6646 |
| 4 | 83.33% | 91.29% | 94.03% | 89.73% | 0.6707 | 0.6707 |
| 8 | 83.58% | 90.80% | 93.78% | 89.48% | 0.6712 | 0.6727 |

Best full-set weight: 2; new Hit@15 cases: 14; regressed cases: 1.
