# 景区数据集三路实时检索评测

- 测试集：/home/humq/rag_core_runs/real_merged7_sdu_v4pro_full_20260921_run1/qa_dataset_435.json（共 435 条）
- Golden 映射：402 条成功，33 条未映射；未映射问题在 macro 指标中按 miss 计入。
- Run ID：`5e48889ab3bc4049aed872b241276b7f`；检索模式：`live_end_to_end`
- 当前集合：`rag_core_sdu_v4pro_full_20260921_run1`；融合策略：`{'name': 'dimension_score', 'file': 'dimension_score.py', 'configuration': {'semantic_weight': 1.0, 'dimension_weight': 0.05, 'lexical_weight': 0.1, 'missing_semantic_penalty': 0.025}}`
- 代码提交：`0005d5a2c9b68988de21add1756d268d8de026d3`；评测时间：2026-11-06T03:24:44+00:00

## 评测口径

export_retrieval_details.py 每轮直接读取原始 QA JSON/JSONL，使用当前 run_dir、当前 Qdrant/本地集合和当前向量检索策略完成查询解析、向量检索、维度检索与 fusion.py 融合。查询解析缓存限定在本轮临时目录。三路结果由同一个 run_id 标识；本报告只接受三份同轮产物，不读取旧实验目录、route snapshot 或先前查询解析结果。

向量检索衡量用户查询语义，使用本轮 semantic candidates；维度检索衡量关键信息匹配，使用 dimension candidates；混合检索使用 fusion.py 当前策略生成的 fusion candidates。Golden 证据按当前集合中的 chunk payload 重新映射。

## 指标定义

chunk 相关性采用二元判断：检索 chunk_id 属于该问题映射后的 golden_chunk_ids 即相关。所有指标按原始测试集问题计算；未映射到当前集合的 Golden 在 Hit、Recall macro、MRR、nDCG 和完整覆盖率中按 miss 计入。Recall micro 以可映射 golden chunk 总数为分母。

1. **Hit@k**：Top-k 命中至少一个 golden chunk 的问题占全部问题的比例。
2. **Recall@k macro**：逐题计算命中 golden chunk 数 / 该题 golden chunk 数，再对全部题目取平均；Golden 未映射题计 0。**Recall@k micro**：全体命中 golden chunk 数 / 全体已映射 golden chunk 总数。
3. **MRR@k**：每题第一个 golden chunk 的倒数排名，再对全部题目取平均；未命中计 0。
4. **nDCG@k**：以 chunk 二元相关性计算折损累计增益，并以理想排序归一化；未命中或无已映射 Golden 计 0。
5. **Doc Precision@k**：Top-k 返回 chunk 中，来源文档属于该题 golden 文档的比例。报告 micro（汇总相关 chunk / 汇总返回 chunk）和 macro（逐题平均）；分母使用实际返回数量。Golden 文档未映射的问题，其返回 chunk 均不计为相关。
6. **Chunk 完整覆盖率**：Top-k 命中该题全部已映射 golden chunks 的问题比例；无映射 Golden 题计 0。

Recall@k macro ≥ 90%、Doc Precision@k micro ≥ 85% 为原目标阈值；报告逐个 k 的判断。

## 结果

| 检索方式 | k | Hit@k | Recall@k macro / micro | MRR@k | MAP@k | nDCG@k | Doc Precision@k micro / macro | Chunk 完整覆盖率 | 阈值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 向量检索（用户查询语义） | 1 | 51.03% | 43.67% / 38.21% | 51.03% | 51.03% | 51.03% | 81.61% / 81.61% | 37.70% | 未达标 |
| 向量检索（用户查询语义） | 5 | 76.55% | 71.37% / 70.40% | 60.72% | 56.79% | 61.59% | 69.06% / 69.06% | 66.44% | 未达标 |
| 向量检索（用户查询语义） | 10 | 83.91% | 79.12% / 80.38% | 61.73% | 58.30% | 64.53% | 62.92% / 62.92% | 74.94% | 未达标 |
| 向量检索（用户查询语义） | 15 | 86.90% | 82.96% / 85.89% | 61.97% | 58.74% | 65.73% | 58.30% / 58.30% | 78.62% | 未达标 |
| 向量检索（用户查询语义） | 20 | 88.05% | 84.73% / 88.30% | 62.04% | 58.91% | 66.22% | 55.47% / 55.38% | 80.92% | 未达标 |
| 维度检索（关键信息匹配） | 1 | 25.29% | 21.17% / 18.93% | 25.29% | 25.29% | 25.29% | 68.86% / 65.06% | 17.47% | 未达标 |
| 维度检索（关键信息匹配） | 5 | 59.31% | 52.93% / 50.43% | 37.77% | 34.00% | 39.82% | 60.56% / 57.07% | 47.13% | 未达标 |
| 维度检索（关键信息匹配） | 10 | 65.06% | 59.50% / 60.24% | 38.61% | 35.41% | 42.45% | 56.63% / 53.46% | 53.79% | 未达标 |
| 维度检索（关键信息匹配） | 15 | 67.82% | 62.99% / 65.40% | 38.84% | 35.84% | 43.56% | 53.98% / 51.38% | 57.47% | 未达标 |
| 维度检索（关键信息匹配） | 20 | 69.89% | 65.00% / 68.50% | 38.96% | 36.01% | 44.13% | 51.50% / 50.01% | 59.77% | 未达标 |
| 混合检索（融合排序） | 1 | 57.70% | 48.70% / 43.20% | 57.70% | 57.70% | 57.70% | 81.38% / 81.38% | 41.15% | 未达标 |
| 混合检索（融合排序） | 5 | 82.30% | 76.13% / 74.35% | 67.40% | 62.20% | 67.12% | 68.51% / 68.51% | 70.11% | 未达标 |
| 混合检索（融合排序） | 10 | 85.98% | 81.77% / 83.13% | 67.86% | 63.46% | 69.31% | 63.10% / 63.10% | 77.47% | 未达标 |
| 混合检索（融合排序） | 15 | 88.28% | 85.02% / 88.30% | 68.04% | 63.88% | 70.36% | 58.22% / 58.22% | 81.15% | 未达标 |
| 混合检索（融合排序） | 20 | 89.43% | 86.61% / 90.88% | 68.10% | 64.03% | 70.82% | 55.11% / 55.04% | 82.76% | 未达标 |

## 文件

- 本轮完整输出目录：`/home/humq/rag_core_pipeline/rag_core/retrieval_fusion/experiments/00_main_experiment/output/runs/latest`
- vector_retrieval_details.jsonl：本轮向量检索 Top-20 详情。
- dimension_retrieval_details.jsonl：本轮维度检索 Top-20 详情。
- hybrid_retrieval_details.jsonl：本轮融合检索 Top-20 详情。
- per_query_metrics.jsonl：逐题、逐路由、逐 k 指标。
- summary.json：汇总指标及运行元数据。

## 重跑命令

    python3 rag_core/retrieval_fusion/experiments/00_main_experiment/code/export_retrieval_details.py
    python3 rag_core/retrieval_fusion/experiments/00_main_experiment/code/calculate_metrics.py --output-dir <export 命令打印的 output_dir>
