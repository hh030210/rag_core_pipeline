# DuRetrieval MTEB dev三路检索评测（实时检索）

- 测试集：/home/humq/rag_core_pipeline/result/duretrieval_mteb_dev_deepseek_v4pro_20260923_run9/evaluation_freshschema_noexpansion_20260927/qa_dataset_gold_ids_only.json（共 2000 条）
- Golden 映射：2000 条成功，0 条未映射；未映射问题在 macro 指标中按 miss 计入。
- Run ID：`4d10ec50bc18462d861c6758d5a74f20`；检索模式：`live_end_to_end`
- 当前集合：`duretrieval_mteb_dev_qrels20k_dim_fresh_20260923`；融合策略：`{'name': 'dimension_score', 'file': 'dimension_score.py', 'configuration': {'semantic_weight': 1.0, 'dimension_weight': 0.05, 'lexical_weight': 0.1, 'missing_semantic_penalty': 0.025}}`
- 索引后端：qdrant；索引点数：32764。
- 代码提交：`8da4df33450527b9525fdcd5b2652c8d403971c2`；评测时间：2026-11-07T02:52:39+00:00

## 查询解析来源

- 配置模式：deterministic
- 使用历史解析回放：否
- 实时 LLM：0 题；历史回放：0 题；规则解析：2000 题。
- 回放原因：无
- 回放数据位置：`未使用`

- 查询扩展：实时 LLM 0 题，历史回放 0 题，原问题 2000 题。
- 查询扩展回放位置：`未使用`

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
| 向量检索（用户查询语义） | 1 | 92.40% | 25.28% / 16.26% | 92.40% | 92.40% | 92.40% | 92.40% / 92.40% | 5.85% | 未达标 |
| 向量检索（用户查询语义） | 5 | 97.25% | 73.53% / 61.19% | 94.39% | 84.19% | 88.18% | 69.56% / 69.56% | 37.30% | 未达标 |
| 向量检索（用户查询语义） | 10 | 97.95% | 87.40% / 80.12% | 94.49% | 83.60% | 88.69% | 45.53% / 45.54% | 64.95% | 未达标 |
| 向量检索（用户查询语义） | 15 | 98.15% | 90.90% / 85.97% | 94.51% | 83.99% | 89.40% | 32.57% / 32.57% | 73.50% | 未达标 |
| 向量检索（用户查询语义） | 20 | 98.35% | 92.57% / 88.90% | 94.52% | 84.25% | 89.84% | 25.26% / 25.26% | 77.10% | 未达标 |
| 维度检索（关键信息匹配） | 1 | 15.20% | 3.71% / 2.67% | 15.20% | 15.20% | 15.20% | 23.82% / 15.20% | 0.75% | 未达标 |
| 维度检索（关键信息匹配） | 5 | 34.55% | 14.16% / 12.46% | 22.98% | 12.80% | 16.96% | 22.59% / 14.38% | 3.35% | 未达标 |
| 维度检索（关键信息匹配） | 10 | 36.95% | 17.61% / 16.57% | 23.31% | 12.29% | 17.09% | 15.11% / 9.68% | 4.85% | 未达标 |
| 维度检索（关键信息匹配） | 15 | 38.10% | 18.84% / 18.01% | 23.40% | 12.29% | 17.37% | 10.98% / 7.10% | 5.50% | 未达标 |
| 维度检索（关键信息匹配） | 20 | 38.50% | 19.49% / 18.79% | 23.42% | 12.32% | 17.53% | 8.61% / 5.62% | 6.00% | 未达标 |
| 混合检索（融合排序） | 1 | 91.75% | 24.94% / 16.14% | 91.75% | 91.75% | 91.75% | 91.75% / 91.75% | 5.60% | 未达标 |
| 混合检索（融合排序） | 5 | 96.85% | 72.84% / 60.73% | 93.90% | 83.25% | 87.42% | 69.03% / 69.03% | 36.90% | 未达标 |
| 混合检索（融合排序） | 10 | 97.75% | 87.20% / 80.10% | 94.02% | 82.97% | 88.24% | 45.52% / 45.52% | 64.85% | 未达标 |
| 混合检索（融合排序） | 15 | 98.05% | 90.48% / 85.64% | 94.04% | 83.25% | 88.83% | 32.45% / 32.45% | 72.85% | 未达标 |
| 混合检索（融合排序） | 20 | 98.25% | 92.34% / 88.73% | 94.05% | 83.55% | 89.34% | 25.21% / 25.21% | 77.10% | 未达标 |

## 文件

- 本轮完整输出目录：`/home/humq/rag_core_pipeline/rag_core/retrieval_fusion/experiments/00_main_experiment/output/duretrieval_mteb/runs/latest`
- vector_retrieval_details.jsonl：本轮向量检索 Top-20 详情。
- dimension_retrieval_details.jsonl：本轮维度检索 Top-20 详情。
- hybrid_retrieval_details.jsonl：本轮融合检索 Top-20 详情。
- per_query_metrics.jsonl：逐题、逐路由、逐 k 指标。
- summary.json：汇总指标及运行元数据。

## 重跑命令

    python3 rag_core/retrieval_fusion/experiments/00_main_experiment/code/export_retrieval_details.py --dataset-profile duretrieval_mteb
    python3 rag_core/retrieval_fusion/experiments/00_main_experiment/code/calculate_metrics.py --dataset-profile duretrieval_mteb --output-dir <export 命令打印的 output_dir>
