# 景区数据集三路实时检索评测

- 测试集：/home/humq/rag_core_runs/real_merged7_sdu_v4pro_full_20260921_run1/qa_dataset_435.json（共 435 条）
- Golden 映射：402 条成功，33 条未映射；未映射问题在 macro 指标中按 miss 计入。
- Run ID：`d07e7e685a284e8fbd14b687e84ee4bc`；检索模式：`live_end_to_end`
- 当前集合：`rag_core_sdu_v4pro_full_20260921_run1`；融合策略：`{'name': 'dimension_score', 'file': 'dimension_score.py', 'configuration': {'semantic_weight': 1.0, 'dimension_weight': 0.05, 'lexical_weight': 0.1, 'missing_semantic_penalty': 0.025}}`
- 代码提交：`96ee915bdef45c05a92ae96bee5b5d46788b9ae8`；评测时间：2026-11-06T07:42:54+00:00

## 查询解析来源

- 配置模式：llm
- 使用历史解析回放：是
- 实时 LLM：0 题；历史回放：435 题；规则解析：0 题。
- 回放原因：本轮指定离线回放
- 回放数据位置：`/home/humq/rag_core_pipeline/result/scenic_latest_gitmain_20261104/retrieval_results.jsonl.gz`

## 评测口径

export_retrieval_details.py 每轮直接读取原始 QA JSON/JSONL，使用当前 run_dir、当前 Qdrant/本地集合和当前向量检索策略完成查询解析、向量检索、维度检索与 fusion.py 融合。查询解析缓存限定在本轮临时目录。三路结果由同一个 run_id 标识；本报告按同轮三路明细计算。启用回放时仅复用历史查询解析，三路候选及排序均在本轮重新计算。

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
| 维度检索（关键信息匹配） | 1 | 33.56% | 27.95% / 25.13% | 33.56% | 33.56% | 33.56% | 74.19% / 73.33% | 23.22% | 未达标 |
| 维度检索（关键信息匹配） | 5 | 64.60% | 57.32% / 55.94% | 44.86% | 40.05% | 45.71% | 61.86% / 61.01% | 50.80% | 未达标 |
| 维度检索（关键信息匹配） | 10 | 71.72% | 65.26% / 66.09% | 45.88% | 41.60% | 48.72% | 57.19% / 56.72% | 59.08% | 未达标 |
| 维度检索（关键信息匹配） | 15 | 75.17% | 69.26% / 71.60% | 46.16% | 42.04% | 49.95% | 54.06% / 54.49% | 62.76% | 未达标 |
| 维度检索（关键信息匹配） | 20 | 76.78% | 71.72% / 75.56% | 46.25% | 42.30% | 50.68% | 51.72% / 53.34% | 65.98% | 未达标 |
| 混合检索（融合排序） | 1 | 58.85% | 49.78% / 44.06% | 58.85% | 58.85% | 58.85% | 81.84% / 81.84% | 42.07% | 未达标 |
| 混合检索（融合排序） | 5 | 81.84% | 76.02% / 74.18% | 67.85% | 62.75% | 67.46% | 68.87% / 68.87% | 70.34% | 未达标 |
| 混合检索（融合排序） | 10 | 87.13% | 82.65% / 84.17% | 68.53% | 64.14% | 70.04% | 63.49% / 63.49% | 77.93% | 未达标 |
| 混合检索（融合排序） | 15 | 88.74% | 85.48% / 88.81% | 68.65% | 64.54% | 70.96% | 58.45% / 58.45% | 81.61% | 未达标 |
| 混合检索（融合排序） | 20 | 89.66% | 87.03% / 91.57% | 68.71% | 64.71% | 71.42% | 55.24% / 55.16% | 83.68% | 未达标 |

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
