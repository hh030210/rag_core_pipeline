# 景区数据集最新检索结果：dynamic_weights_v3

本目录记录景区问答数据集 435 条问题的最新动态融合评测结果。逐问题明细压缩保存在 `results.jsonl.gz`，汇总指标见 `summary.md` 和 `summary.json`。

## 评测范围与口径

- 共 435 条问题，全部标记为可回答；其中 402 条问题的 Golden chunk 成功映射。
- 本实验将已保存的语义/维度候选重放并按 `dynamic_weights_v3` 重新计算融合排名；候选来自 `alpha_sweep_20260921/alpha_0.2/evaluation/results.jsonl`。
- 这是候选重放评测：没有重新生成向量候选，也没有重新调用问答模型。
- 每条记录包含语义、维度和融合候选，以及查询解析、Golden 编号、匹配分数和融合诊断；保留候选 chunk 正文。
- 三路候选明细总量：语义 7,547 条、维度 5,460 条、融合 9,955 条。

## Top-5 检索指标

| 路由 | HitRate@5 | GoldenRecall@5 | MRR@5 | nDCG@5 | 未命中数 |
|---|---:|---:|---:|---:|---:|
| 语义 | 76.09% | 70.71% | 60.56% | 61.12% | 53 |
| 维度 | 59.08% | 52.82% | 38.31% | 40.43% | 133 |
| 融合（dynamic_weights_v3） | 77.93% | 72.23% | 62.25% | 62.66% | 45 |

融合采用逐问题的动态权重，形式为：

```text
(1 - dimension_weight) * normalized_semantic
+ dimension_weight * normalized_dimension
```

本轮平均维度权重为 0.2194；两路均有候选的问题平均维度权重为 0.2328。完整权重诊断可查看 JSONL 中的 `fusion_strategy`。

## 明细读取

`results.jsonl.gz` 解压后每行是一个 JSON 对象。`gold.chunk_ids` 是 Golden chunk 编号；`retrieval.semantic_results`、`retrieval.dimension_results`、`retrieval.fusion_results` 分别是三路候选排名。候选项保留分数、排名、Golden 命中标志和维度匹配信息，融合项还保留两路原始/归一化分数及来源排名。
