# Run9 全量检索明细

本目录保存 run9 最新一次全量检索评测的逐问题明细。评测共 2,000 条查询，数据来自新数据集 run9；检索结果由服务器上的 Qdrant 集合生成，查询解析模型为 DeepSeek-V4-Pro，向量模型为本地 BGE-M3，未启用 query expansion。

## 文件

- `retrieval_details.jsonl.gz`：逐问题的压缩 JSONL 明细，共 2,000 行。
- `summary.json`、`summary.md`：本次评测的汇总指标。
- `manifest.json`：数据来源、参数、记录数和候选统计。

明细保留问题、Golden chunk ID、查询解析结果、逐路检索指标，以及语义/维度/融合三路的候选排名和分数。语义与维度路各最多保留 100 个候选；融合路保留完整候选并集排序。候选项包含 chunk ID、rank、原始/归一化分数、是否 Golden、维度路径和匹配信息。为控制仓库体积，未包含 chunk 正文。

## 检索指标（@5）

| 路由 | HitRate@5 | GoldenRecall@5 | MRR@5 | nDCG@5 | 100 候选未命中数 |
|---|---:|---:|---:|---:|---:|
| 语义 | 97.30% | 73.66% | 94.44% | 88.31% | 15 |
| 维度 | 65.25% | 29.65% | 52.90% | 37.89% | 450 |
| 融合 | 97.45% | 71.78% | 92.78% | 85.52% | 14 |

## 读取

解压后每行是一个 JSON 对象。主要路径：

- `gold.chunk_ids`：该问题的 Golden chunk 编号。
- `retrieval.semantic_results`、`retrieval.dimension_results`、`retrieval.fusion_results`：对应检索路由的有序候选。
- `retrieval_metrics`：该问题逐路的 Hit、Golden Recall、MRR、nDCG 等指标。
- `query_analysis`：查询解析出的约束、活动维度、路由和诊断信息。

逐候选保留 `chunk_id`、`rank`、分数、`is_gold` 等字段；chunk 正文需根据 chunk ID 到原始索引/语料中查找。
