# DeepSeek-V4-Pro 全流程实验结果

## 实验流程

1. 使用 `data_input/test_data` 进行动态分片。
2. 使用 BGE-M3 生成向量并写入 Qdrant。
3. 使用 DeepSeek-V4-Pro 进行维度结构生成、chunk 标签抽取和维度匹配。
4. 使用 20 条样例进行 3 轮 Prompt 迭代优化。
5. 使用 `merged_7_rag_test_set_filtered.json` 进行语义、维度和融合检索评测。
6. 使用 Top-5 检索结果和完整 chunk 内容生成问答结果。

## 关键结果

- Chunk 数：316
- Qdrant 向量数：316
- 维度总数：20
- 可索引叶子维度数：14
- 检索评测：435 条，402 条金标准成功映射
- 问答生成：435/435 成功，0 失败
- 问答结果中的 `semantic_top5`、`dimension_top5`、`fusion_top5` 按最多 5 条保留；如果某一路由去重后的候选不足 5 条，则保留实际可召回数量。所有实际返回项均带有对应的 `chunk_text_full`

## 主要文件

- `V_core_v2.json`：维度结构
- `tags_output_v2.json`：chunk 维度标签
- `inverted_index_v2.json`：维度倒排索引
- `evaluation_noexpansion/`：检索评测结果与汇总
- `qa_results.jsonl`：全量问答结果
- `qa_summary.json`：问答运行汇总
- `optimized_prompt.json`：迭代优化后的问答 Prompt
- `experiment_comparison.md`：与上一版本的 Top-5 对比表
- `experiment_comparison.json`：对比数据

检索评测使用了 `--no-query-expansion`，以便与上一版本保持同口径。
