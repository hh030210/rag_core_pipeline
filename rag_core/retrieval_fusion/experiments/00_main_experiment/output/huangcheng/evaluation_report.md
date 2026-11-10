# 皇城相府高精度扩展集三路检索评测（实时检索）

- 测试集：/home/humq/rag_core_runs/huangcheng_20261019/eval_source/gold_slot_audit_v1/retrieval_gold_slot_extended_auto_eval_v1.jsonl（共 126 条）
- Golden 映射：126 条成功，0 条未映射；未映射问题在 macro 指标中按 miss 计入。
- Run ID：`22b27e081cc54b5cbc4e28e8ca20d3e1`；检索模式：`live_end_to_end`
- 当前集合：`rag_core_huangcheng_20261019_v1`；融合策略：`{'name': 'dimension_score', 'file': 'dimension_score.py', 'configuration': {'semantic_weight': 1.0, 'dimension_weight': 0.05, 'lexical_weight': 0.1, 'missing_semantic_penalty': 0.025}}`
- 索引后端：qdrant；索引点数：290。
- 代码提交：`8da4df33450527b9525fdcd5b2652c8d403971c2`；评测时间：2026-11-07T02:17:43+00:00

## 查询解析来源

- 配置模式：deterministic
- 使用历史解析回放：否
- 实时 LLM：0 题；历史回放：0 题；规则解析：126 题。
- 回放原因：无
- 回放数据位置：`未使用`

- 查询扩展：实时 LLM 0 题，历史回放 0 题，原问题 126 题。
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
| 向量检索（用户查询语义） | 1 | 57.14% | 54.10% / 50.35% | 57.14% | 57.14% | 57.14% | 78.57% / 78.57% | 51.59% | 未达标 |
| 向量检索（用户查询语义） | 5 | 88.89% | 87.04% / 85.31% | 69.89% | 68.17% | 73.45% | 56.51% / 56.51% | 84.92% | 未达标 |
| 向量检索（用户查询语义） | 10 | 93.65% | 92.20% / 91.61% | 70.59% | 69.02% | 75.29% | 52.30% / 52.30% | 90.48% | 未达标 |
| 向量检索（用户查询语义） | 15 | 95.24% | 94.18% / 93.71% | 70.71% | 69.20% | 75.83% | 48.52% / 48.52% | 92.86% | 未达标 |
| 向量检索（用户查询语义） | 20 | 95.24% | 94.58% / 94.41% | 70.71% | 69.25% | 75.94% | 46.11% / 46.11% | 93.65% | 未达标 |
| 维度检索（关键信息匹配） | 1 | 30.16% | 28.97% / 26.57% | 30.16% | 30.16% | 30.16% | 66.00% / 52.38% | 27.78% | 未达标 |
| 维度检索（关键信息匹配） | 5 | 48.41% | 46.96% / 46.15% | 37.74% | 36.59% | 39.56% | 55.71% / 44.17% | 45.24% | 未达标 |
| 维度检索（关键信息匹配） | 10 | 53.17% | 52.12% / 51.75% | 38.43% | 37.45% | 41.38% | 50.96% / 40.35% | 50.79% | 未达标 |
| 维度检索（关键信息匹配） | 15 | 57.14% | 55.42% / 55.94% | 38.74% | 37.73% | 42.33% | 48.01% / 38.26% | 53.17% | 未达标 |
| 维度检索（关键信息匹配） | 20 | 57.94% | 56.61% / 57.34% | 38.79% | 37.82% | 42.64% | 46.85% / 37.50% | 54.76% | 未达标 |
| 混合检索（融合排序） | 1 | 70.63% | 67.33% / 62.24% | 70.63% | 70.63% | 70.63% | 80.95% / 80.95% | 64.29% | 未达标 |
| 混合检索（融合排序） | 5 | 92.86% | 91.01% / 89.51% | 80.05% | 78.38% | 82.10% | 59.37% / 59.37% | 88.89% | 未达标 |
| 混合检索（融合排序） | 10 | 94.44% | 93.39% / 93.01% | 80.27% | 78.88% | 83.00% | 51.90% / 51.90% | 92.06% | 未达标 |
| 混合检索（融合排序） | 15 | 96.83% | 95.77% / 95.10% | 80.46% | 79.07% | 83.63% | 48.89% / 48.89% | 94.44% | 未达标 |
| 混合检索（融合排序） | 20 | 96.83% | 96.16% / 95.80% | 80.46% | 79.11% | 83.74% | 45.83% / 45.83% | 95.24% | 未达标 |

## 文件

- 本轮完整输出目录：`/home/humq/rag_core_pipeline/rag_core/retrieval_fusion/experiments/00_main_experiment/output/huangcheng/runs/latest`
- vector_retrieval_details.jsonl：本轮向量检索 Top-20 详情。
- dimension_retrieval_details.jsonl：本轮维度检索 Top-20 详情。
- hybrid_retrieval_details.jsonl：本轮融合检索 Top-20 详情。
- per_query_metrics.jsonl：逐题、逐路由、逐 k 指标。
- summary.json：汇总指标及运行元数据。

## 重跑命令

    python3 rag_core/retrieval_fusion/experiments/00_main_experiment/code/export_retrieval_details.py --dataset-profile huangcheng
    python3 rag_core/retrieval_fusion/experiments/00_main_experiment/code/calculate_metrics.py --dataset-profile huangcheng --output-dir <export 命令打印的 output_dir>
