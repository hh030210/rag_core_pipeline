# real_merged7_v4pro_full_20260917_run1 实验结果

本目录保存服务器上完成的全量实验结果。

## 核心文件

- `qa_results.jsonl`：435 条全量生成式问答结果，失败数为 0。
- `qa_summary.json`：全量问答统计。
- `evaluation/results.jsonl`：逐问题检索评测结果。
- `evaluation/summary.json` / `evaluation/summary.md`：语义、维度和融合检索汇总指标。
- `optimized_prompt.json`：3 轮 Prompt 迭代优化结果。
- `dimension_miss_examples.md`：维度检索未命中案例整理。
- `qa_sample_complete.json`：完整单条问答样例。

## Top-5 结果格式

`qa_results.jsonl` 的每条记录都包含 `retrieval_top5` 字段：

```json
{
  "retrieval_top5": {
    "semantic_top5": [],
    "dimension_top5": [],
    "fusion_top5": []
  }
}
```

三个列表分别保留语义检索、维度检索和融合检索的独立 Top-5 结果；每个结果项包含完整的 `chunk_text_full`，没有截断为评测摘要长度。

## 数据范围

- 有效问答数：435
- 原始数据中 1 条空问题被过滤。
- 问答模型：`DeepSeek-V4-Pro`
- 向量模型：`bge-m3`
- Qdrant collection：`rag_core_v4pro_full_20260917_run1`
