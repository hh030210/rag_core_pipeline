# Top-15 完整 chunk 正文检索明细

- 源文件：evaluation_noexpansion/results.jsonl
- chunk 全文来源：../chunks/chunks.json 的 chunk_text_full（缺省时回退到 chunk_text / doc_text）
- 问题数：435（源文件实际记录数；不是 347）
- 每条问题保留语义、维度、融合三路各自 rank ≤ 15 的候选；不改动原始排序和评测指标。
- gzip JSONL：每行一条问题记录。候选 chunk_text 是完整正文，text_truncated 固定为 false；original_preview_was_truncated 保留原文件的预览截断状态，full_text_chars 是全文字符数。
- 共有 300 个不同候选 chunk ID；所有候选均匹配到完整正文。

| 路由 | 导出候选出现次数 | 原预览曾被截断的出现次数 |
|---|---:|---:|
| 语义 | 6051 | 1580 |
| 维度 | 4227 | 1335 |
| 融合 | 7534 | 2050 |
