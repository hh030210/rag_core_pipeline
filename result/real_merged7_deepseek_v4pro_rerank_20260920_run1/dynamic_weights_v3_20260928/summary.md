# Retrieval Evaluation Summary

- Total: 435
- Answerable: 435
- Gold mapped: 402
- Fusion badcases: 51

| Route | MRR@5 | HitRate@5 | GoldenRecall@5 | nDCG@5 | Misses |
|---|---:|---:|---:|---:|---:|
| semantic | 0.6056 | 0.7609 | 0.7071 | 0.6112 | 53 |
| dimension | 0.3831 | 0.5908 | 0.5282 | 0.4043 | 133 |
| fusion | 0.6225 | 0.7793 | 0.7223 | 0.6266 | 45 |

## Route Classes

- `both_hit`: 291
- `dimension_only_hit`: 7
- `neither_hit`: 46
- `semantic_only_hit`: 91

## 动态权重诊断

- 平均维度权重：0.2194
- 双路均有候选时的维度权重范围：0.1504–0.3167
- 双路均有候选时的平均维度权重：0.2328
- 有强主/次维度规范标签证据的问题数：176 / 435
- 四指标均值：68.7677%

> 口径：对同一批 435 条缓存候选应用 dynamic_weights_v3 并重算检索指标；没有重新生成向量候选或调用问答模型。
