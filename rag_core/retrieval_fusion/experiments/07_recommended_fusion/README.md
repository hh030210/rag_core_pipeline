# 推荐融合：维度角色校准

从仓库根目录独立运行：

```bash
python -m rag_core.retrieval_fusion.experiments.07_recommended_fusion.code.run
```

默认读取 06_dimension_score/output/dataset 的冻结输入及候选；--baseline-dataset 可切换基准，--input-dataset 可提供同批次兼容输入。

输出只有 output/dataset/index.json 和 details/*.json，保存输入、Golden、正文、候选及必要诊断，可视化直接计算指标。成功后整批覆盖自己的 output，失败保留旧数据。不生成 summary.json 或 badcases.json。不会修改其他实验，线上或离线重新运行不会改变本实验已保存的数据。
