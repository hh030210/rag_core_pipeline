# 离线融合：维度分数融合

从仓库根目录独立运行：

```bash
python -m rag_core.retrieval_fusion.experiments.06_dimension_score.code.run
```

默认读取 01_online_snapshots/output/dataset，保存完整输入候选、查询、Golden、正文以及维度分数融合结果。输出只有 output/dataset/index.json 和 details/*.json，可视化直接读取并计算指标。

成功后整批替换自己的 output，失败保留旧数据。不生成 comparison.jsonl 或 summary.json。不会修改或运行其他实验；后续线上更新不会改变已保存的本实验数据。
