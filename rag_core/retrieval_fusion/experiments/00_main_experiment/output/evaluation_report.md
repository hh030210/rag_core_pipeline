# 景区数据集多路检索融合评测

- 评测时间：2026-11-04T03:37:14+00:00
- 融合入口：rag_core.retrieval_fusion.fusion.apply_fusion
- 当前策略：dimension_score.py
- 测试集：/home/humq/rag_core_runs/real_merged7_sdu_v4pro_full_20260921_run1/qa_dataset_435.json（435 条）
- 路由候选来源：/home/humq/rag_core_pipeline/rag_core/retrieval_fusion/experiments/00_method_comparison/output/dimension_score，批次 182f6bb9fe3247189ad1fd9b007aeb46
- 有效评测问题：402 条；无 golden chunk 排除 33 条；证据映射失败排除 0 条。
- Golden 文档映射回退次数：33（从 chunk ID 的来源文件前缀推断）。

## 评测范围

程序读取现有景区 QA 测试集对应的语义路由和维度路由候选，并调用当前 fusion.py 的 apply_fusion 重新执行融合排序。该实验评估融合排序本身；没有重新调用向量库、embedding 或 query parser。路由候选批次 ID 保存在 summary.json，便于复现和审计。

Golden chunk 使用路由快照中由 QA source 证据文本映射出的 gold.chunk_ids。Golden 文档优先取候选记录的 source_file；当 golden chunk 没有候选元数据时，从其 chunk ID 的来源文件前缀推断。

## 指标口径

所有指标只在有 golden chunk 且证据映射成功的问题上计算。k 表示融合排序 Top-k；若实际返回少于 k 条，则准确率分母为实际返回条数。

1. **Hit@k**：每题 Top-k 至少命中一个 golden chunk 的问题比例。
2. **Gold Recall@k**：每题命中的 golden chunk 数 / golden chunk 总数；报告 micro 汇总值与 macro 逐题平均值。
3. **Precision@k**：Top-k 中精确命中的 golden chunk 数 / k，保留原有 chunk 级精确率口径。
4. **MRR@k**：每题首个 golden chunk 排名倒数的平均值；同时保留全融合候选集的 RR、first_gold_rank、miss 和 result_count。
5. **NDCG@k**：按 golden chunk 二元相关性计算折损累计增益并按理想排序归一化。
6. **Chunk 完整覆盖率**：Top-k 命中该题全部 golden chunks 的问题数 / 有效问题数，用于检查关键证据是否齐全。
7. **Doc 召回率**：Top-k 命中的 golden 来源文档数 / golden 来源文档总数；报告 micro、macro 和逐题完整覆盖率。
8. **同源文档准确率**：Top-k 中属于该题 golden 来源文档的 chunk 数 / Top-k 实际返回 chunk 数；按用户要求，同一 golden 文档中的 chunk 均计为有效。

验收阈值按汇总 micro 指标判断：Chunk 召回率 ≥ 90%，Doc 召回率 ≥ 90%，准确率 ≥ 85%。逐题完整覆盖率作为额外诊断项展示。

## 结果

| K | Hit@k | Gold Recall@k micro / macro | Precision@k | MRR@k | NDCG@k | Chunk 完整覆盖率 | Doc Recall micro | Doc 完整覆盖率 | 同源文档准确率 micro | 阈值结果 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 62.94% | 43.55% / 53.41% | 62.94% | 62.94% | 62.94% | 45.52% | 82.60% | 81.84% | 88.56% | 未全部达标 |
| 5 | 88.81% | 74.70% / 82.55% | 21.59% | 73.09% | 73.09% | 76.62% | 96.98% | 97.01% | 74.46% | 未全部达标 |
| 10 | 95.27% | 85.37% / 90.49% | 12.34% | 73.96% | 76.14% | 85.57% | 98.84% | 98.76% | 69.45% | 未全部达标 |

## 达标结论

Top-10 汇总：Chunk 召回率 85.37%，Doc 召回率 98.84%，准确率 69.45%。
达到阈值：doc_recall_micro；未达到阈值：chunk_recall_micro, accuracy_micro。
Top-10 Hit 率为 95.27%；这表示至少命中一块 golden chunk，不表示每题所有证据均齐全。
Top-10 每题完整召回全部 golden chunks 的比例为 85.57%。
全候选集 MRR 为 74.06%，miss 率为 2.99%，首个 golden chunk 的平均排名（命中问题）为 2.32，平均候选数为 29.62。

## 结果解释

Chunk 级指标判断标准证据分块是否进入 Top-k；Doc 级指标判断这些证据所在的来源文档是否被覆盖。准确率依照本实验约定，统计 Top-k 中属于 golden 来源文档的所有 chunk，因此不要求 chunk 本身必须等于 golden chunk。

逐题明细见 per_query.jsonl，汇总指标和阈值布尔值见 summary.json。重跑命令：

    python3 rag_core/retrieval_fusion/experiments/00_main_experiment/code/evaluate_main.py --ks 1,5,10
