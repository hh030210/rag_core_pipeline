# 景区融合主实验（本地冻结数据）

本实验复用主实验的三路明细格式、Golden 映射和指标口径。语义检索使用当前向量策略，融合使用当前融合策略。两路都在本轮重新执行。

## 数据来源

- 冻结快照来源提交为 `9e5dd32`。
- 语义检索和 Golden 映射使用冻结快照的本地 `local` 后端。索引文件是 `/home/humq/rag_core_runs/scenic_best_dimension_20261012_local_sync/result/hybrid_legacy_tags_sdu_entities_20261002_verified_facts/local_points.json`。
- 维度路由固定读取 `/home/humq/rag_core_runs/scenic_best_dimension_20261012_local_sync/dimension_retrieval_best_20261012/dimension_candidates_top20_for_fusion.jsonl`，按题目 ID 和原问题对齐，并保留候选顺序和分数。
- 默认测试集是冻结索引目录下的 `merged_7_rag_test_set_filtered.json`。按主实验口径评测全部 435 题。候选文件覆盖 402 题，另外 33 题的维度候选为空。

## 运行

从服务器项目根目录执行以下两条命令。示例使用归档查询解析，适合离线运行。

```bash
experiment=rag_core/retrieval_fusion/experiments/00_main_experiment_local_data
python3 "$experiment/code/export_retrieval_details.py" \
  --model-path /home/humq/hf_cache/bge-m3 --device cuda --replay-query-analysis
python3 "$experiment/code/calculate_metrics.py"
```

完整产物保存到 `output/runs/latest/`，固定报告入口是 `output/evaluation_report.md`。
