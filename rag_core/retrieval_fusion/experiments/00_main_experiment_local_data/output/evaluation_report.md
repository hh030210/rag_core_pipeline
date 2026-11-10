# 景区数据集三路检索评测（本地冻结数据）

- 测试集：/home/humq/rag_core_runs/scenic_best_dimension_20261012_local_sync/result/hybrid_legacy_tags_sdu_entities_20261002_verified_facts/merged_7_rag_test_set_filtered.json（共 435 条）
- Golden 映射：402 条成功，33 条未映射；未映射问题在 macro 指标中按 miss 计入。
- Run ID：`cfde51f92a9f4b01a3bf537c9e20eaa3`；检索模式：`live_semantic_frozen_dimension_live_fusion`
- 当前集合：`rag_core_chunks_v1`；融合策略：`{'name': 'dimension_score', 'file': 'dimension_score.py', 'configuration': {'semantic_weight': 1.0, 'dimension_weight': 0.05, 'lexical_weight': 0.1, 'missing_semantic_penalty': 0.025}}`
- 代码提交：`8da4df33450527b9525fdcd5b2652c8d403971c2`；评测时间：2026-11-06T08:55:49+00:00

## 本地冻结数据

- 来源提交：9e5dd32
- 语义检索和 Golden 映射使用的本地索引：/home/humq/rag_core_runs/scenic_best_dimension_20261012_local_sync/result/hybrid_legacy_tags_sdu_entities_20261002_verified_facts/local_points.json
- 维度检索固定读取的候选文件：/home/humq/rag_core_runs/scenic_best_dimension_20261012_local_sync/dimension_retrieval_best_20261012/dimension_candidates_top20_for_fusion.jsonl
- 候选文件覆盖 402 题。其余 33 题的维度候选为空，仍计入全量指标。
- 语义检索与融合排序在本轮重新执行。

## 查询解析来源

- 配置模式：llm
- 使用历史解析回放：是
- 实时 LLM：0 题；历史回放：435 题；规则解析：0 题。
- 回放原因：评测环境未配置 API 密钥
- 回放数据位置：`/home/humq/rag_core_pipeline/result/scenic_latest_gitmain_20261104/retrieval_results.jsonl.gz`

- 查询扩展：实时 LLM 0 题，历史回放 0 题，原问题 435 题。
- 查询扩展回放位置：`未使用`

## 评测口径

export_retrieval_details.py 每轮直接读取原始 QA JSON/JSONL，使用当前 run_dir、当前 Qdrant/本地集合和当前向量检索策略完成查询解析、向量检索、读取冻结维度候选与融合排序。查询解析缓存限定在本轮临时目录。三路结果由同一个 run_id 标识；本报告按同轮三路明细计算。查询解析回放仅替换查询解析输入。

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
| 向量检索（用户查询语义） | 1 | 54.25% | 45.74% / 40.34% | 54.25% | 54.25% | 54.25% | 82.76% / 82.76% | 38.85% | 未达标 |
| 向量检索（用户查询语义） | 5 | 76.78% | 71.13% / 70.77% | 62.90% | 58.48% | 62.90% | 68.46% / 68.46% | 65.98% | 未达标 |
| 向量检索（用户查询语义） | 10 | 84.14% | 79.51% / 81.54% | 63.93% | 60.17% | 66.11% | 62.55% / 62.55% | 74.94% | 未达标 |
| 向量检索（用户查询语义） | 15 | 86.67% | 82.81% / 85.98% | 64.13% | 60.55% | 67.12% | 57.81% / 57.81% | 78.62% | 未达标 |
| 向量检索（用户查询语义） | 20 | 88.74% | 85.46% / 89.23% | 64.25% | 60.78% | 67.84% | 54.80% / 54.96% | 81.61% | 未达标 |
| 维度检索（关键信息匹配） | 1 | 41.15% | 34.04% / 30.60% | 41.15% | 41.15% | 41.15% | 81.84% / 75.63% | 28.05% | 未达标 |
| 维度检索（关键信息匹配） | 5 | 65.52% | 58.00% / 56.07% | 50.31% | 44.39% | 49.31% | 69.70% / 64.91% | 51.26% | 未达标 |
| 维度检索（关键信息匹配） | 10 | 75.63% | 69.73% / 70.09% | 51.70% | 46.52% | 53.61% | 65.00% / 60.91% | 63.91% | 未达标 |
| 维度检索（关键信息匹配） | 15 | 78.85% | 72.99% / 75.38% | 51.96% | 46.92% | 54.68% | 60.39% / 57.65% | 66.67% | 未达标 |
| 维度检索（关键信息匹配） | 20 | 80.69% | 75.15% / 79.15% | 52.07% | 47.13% | 55.33% | 57.28% / 55.94% | 68.51% | 未达标 |
| 混合检索（融合排序） | 1 | 60.00% | 50.98% / 44.62% | 60.00% | 60.00% | 60.00% | 84.14% / 84.14% | 43.45% | 未达标 |
| 混合检索（融合排序） | 5 | 82.53% | 76.69% / 75.21% | 68.99% | 64.14% | 68.65% | 69.33% / 69.33% | 71.03% | 未达标 |
| 混合检索（融合排序） | 10 | 86.21% | 81.87% / 83.59% | 69.49% | 65.42% | 70.77% | 63.75% / 63.75% | 77.24% | 未达标 |
| 混合检索（融合排序） | 15 | 88.97% | 85.39% / 88.89% | 69.69% | 65.86% | 71.88% | 58.44% / 58.44% | 81.38% | 未达标 |
| 混合检索（融合排序） | 20 | 90.57% | 87.82% / 92.65% | 69.78% | 66.10% | 72.59% | 55.21% / 55.36% | 84.37% | 未达标 |

## 文件

- 本轮完整输出目录：`/home/humq/rag_core_pipeline/rag_core/retrieval_fusion/experiments/00_main_experiment_local_data/output/runs/latest`
- vector_retrieval_details.jsonl：本轮向量检索 Top-20 详情。
- dimension_retrieval_details.jsonl：本轮维度检索 Top-20 详情。
- hybrid_retrieval_details.jsonl：本轮融合检索 Top-20 详情。
- per_query_metrics.jsonl：逐题、逐路由、逐 k 指标。
- summary.json：汇总指标及运行元数据。

## 重跑命令

    python3 rag_core/retrieval_fusion/experiments/00_main_experiment_local_data/code/export_retrieval_details.py
    python3 rag_core/retrieval_fusion/experiments/00_main_experiment_local_data/code/calculate_metrics.py --output-dir <export 命令打印的 output_dir>
