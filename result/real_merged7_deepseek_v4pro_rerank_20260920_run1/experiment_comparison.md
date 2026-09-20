# DeepSeek-V4-Pro 全流程实验对比

> 对比口径：两次实验使用同一份 435 条问答评测集，检索评测均关闭 query expansion，指标为 Top-5。逐条对齐采用 (id, occurrence_index)，因此保留了数据集中重复 ID 的全部样本。

## 运行配置

| 项目 | 之前版本 | 当前版本 |
|---|---:|---:|
| 模型 | DeepSeek-V4-Pro | DeepSeek-V4-Pro |
| 代码版本 | real_merged7_v4pro_full_20260917_run1 | real_merged7_deepseek_v4pro_rerank_20260920_run1 |
| 输入数据 | data_input/test_data | data_input/test_data |
| Chunk 数 | 323 | 316 |
| Qdrant 向量数 | 323 | 316 |
| 全部维度数 | 19 | 20 |
| 可索引叶子维度数 | 13 | 14 |
| Prompt 优化 | 3 轮 | 3 轮 |
| QA 生成 | 435/435 成功 | 435/435 成功 |

## Top-5 检索指标对比

| 检索路线 / 指标 | 之前版本 | 当前版本 | 变化 |
|---|---:|---:|---:|
| 语义检索 / MRR | 62.07% | 60.56% | -1.51pp |
| 语义检索 / Hit Rate | 76.32% | 76.09% | -0.23pp |
| 语义检索 / Gold Recall | 70.44% | 70.71% | +0.27pp |
| 语义检索 / nDCG | 62.04% | 61.12% | -0.92pp |
| 维度检索 / MRR | 14.56% | 38.95% | +24.39pp |
| 维度检索 / Hit Rate | 25.06% | 60.46% | +35.40pp |
| 维度检索 / Gold Recall | 19.88% | 53.89% | +34.01pp |
| 维度检索 / nDCG | 14.13% | 41.07% | +26.93pp |
| 融合检索 / MRR | 62.79% | 62.41% | -0.38pp |
| 融合检索 / Hit Rate | 76.32% | 77.70% | +1.38pp |
| 融合检索 / Gold Recall | 70.88% | 72.31% | +1.43pp |
| 融合检索 / nDCG | 62.65% | 62.85% | +0.20pp |

## 逐条 Top-5 命中状态变化

| 检索路线 | 新版新增命中 | 新版回退未命中 | 命中状态不变 | 平均 Gold Recall 变化 |
|---|---:|---:|---:|---:|
| 语义检索 | 7 | 8 | 420 | +0.27pp |
| 维度检索 | 183 | 29 | 223 | +34.01pp |
| 融合检索 | 20 | 14 | 401 | +1.43pp |

## 数据完整性

- 旧版评测结果：435 行；新版评测结果：435 行。
- 按 (id, occurrence_index) 成功对齐：435 行；题目文本不一致：0 行。
- 旧版 badcase：51；新版 badcase：50。
- 新版 QA：435/435 成功，失败 0 条。

## 结果文件

- 新版全流程目录：/home/humq/rag_core_runs/real_merged7_deepseek_v4pro_rerank_20260920_run1
- 新版检索评测：/home/humq/rag_core_runs/real_merged7_deepseek_v4pro_rerank_20260920_run1/evaluation_noexpansion
- 新版问答结果：/home/humq/rag_core_runs/real_merged7_deepseek_v4pro_rerank_20260920_run1/qa_results.jsonl
- 新版维度结构：/home/humq/rag_core_runs/real_merged7_deepseek_v4pro_rerank_20260920_run1/V_core_v2.json
- 新版标签结果：/home/humq/rag_core_runs/real_merged7_deepseek_v4pro_rerank_20260920_run1/tags_output_v2.json
- 对比 JSON：/home/humq/rag_core_runs/real_merged7_deepseek_v4pro_rerank_20260920_run1/experiment_comparison.json
