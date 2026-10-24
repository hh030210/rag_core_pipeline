# 三数据集三路检索 Top10 Chunk 明细

此目录保存景区数据集、DuRetrieval-MTEB dev 和 db_qa 的语义、维度、融合三路检索结果。每个 `.jsonl.gz` 解压后每行对应一个问题及其三个检索通道；各通道数组保留实际返回的结果数量，少于 10 条时不补造结果。

每条 chunk 记录包含原始排名、分数、chunk ID、来源文件、标题、完整 chunk 正文和 `is_gold` 命中标记，并保留可用的通道分数字段。正文通过 `chunk_id` 回连到数据集切片文本。

## 文件

- `scenic_before_top10_chunks.jsonl.gz`：景区数据集，优化前，402 个问题。
- `scenic_after_top10_chunks.jsonl.gz`：景区数据集，优化后，402 个问题。
- `duretrieval_top10_chunks.jsonl.gz`：DuRetrieval-MTEB dev，2,000 个问题；来源为当前可恢复的完整三路归档评测 `evaluation_query_facet_evidence_v2_full_gpu0_20260928`。
- `dbqa_top10_chunks.jsonl.gz`：db_qa，全流程实验，3,199 个问题。
- `three_route_chunk_details.docx`：每套数据各展示一个问题的三路 Top10 实际 chunk 样例；所有问题的完整明细在上述压缩 JSONL 中。
- `manifest.json`：文件、数据版本和记录计数。

## 读取示例

```python
import gzip, json

path = "duretrieval_top10_chunks.jsonl.gz"
with gzip.open(path, "rt", encoding="utf-8") as f:
    record = json.loads(next(f))

print(record["question"])
for route in ("semantic", "dimension", "fusion"):
    print("\n", route)
    for hit in record["routes"][route]:
        print(hit["rank"], hit["score"], hit["chunk_id"], hit["source_file"])
        print(hit["chunk_text"])
```

