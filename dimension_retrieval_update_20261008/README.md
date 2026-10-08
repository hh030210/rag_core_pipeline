# 维度检索独立交付包（2026-10-08）

本目录用于多人协作时识别和复现实验，不修改既有 `rag_core/` 主流程的目录结构。

## 本轮代码基线

- Git 提交：`9e831d6`（`Add verified fact index and retrieval evaluation`）
- 主流程代码仍位于仓库根目录的正式模块中；本包通过下方代码清单锁定其文件与职责，避免复制出两套会漂移的可执行代码。
- 如需在个人分支提取本轮代码，可使用：`git checkout 9e831d6 -- <文件路径>`。

| 组件 | 正式代码位置 | 作用 |
| --- | --- | --- |
| 验证事实索引 | `rag_core/fact_index.py` | 三元组校验、主体/关系/属性倒排与精确匹配 |
| 检索接入 | `rag_core/retrieval.py` | 验证事实候选、Top10 后尾插、实体锚点和词法兜底 |
| 配置 | `rag_core/settings.py` | `fact_index_candidate_recall_enabled`、`fact_index_tail_insertion_rank` |
| LLM 抽取 | `code_jyx/llm_service.py` | 文档事实和查询事实约束抽取 |
| 索引构建 | `build_verified_fact_index.py` | 可断点抽取与 `fact_index_v1.json` 构建 |
| 查询缓存 | `build_query_fact_cache.py` | 查询事实约束缓存 |
| 维度评测 | `evaluate_dimension_only.py` | 单策略评测入口 |
| 严格 A/B | `evaluate_fact_tail_ab.py` | 单进程配对评测，避免缓存混杂 |

## 结果文件

- `results/fact_tail10_paired_summary.json`：本轮权威的同进程 @15 配对 A/B 指标。
- `results/fact_index_manifest.json`：全量事实索引覆盖与三元组统计。
- `results/top15_results_compact.jsonl`：402 条已映射 Golden 的逐 query Top15 检索结果；已去除原始 chunk 全文，保留 query、gold、候选 chunk id、来源、排序和指标。

完整的原始 `results.jsonl` 含大量重复 chunk 文本，约 40MB，不提交，以避免仓库膨胀；其紧凑导出内容足以复核检索命中与排序。

## 复现实验

```powershell
python evaluate_fact_tail_ab.py `
  --run-dir result/hybrid_legacy_tags_sdu_entities_20261002_verified_facts `
  --dataset result/hybrid_legacy_tags_sdu_entities_20261002_verified_facts/merged_7_rag_test_set_filtered.json `
  --query-fact-cache result/hybrid_legacy_tags_sdu_entities_20261002_verified_facts/query_fact_cache_rewritten_v1.json `
  --model-path <BGE-M3路径> `
  --tail-start 10 `
  --output result/evaluation_verified_fact_index_tail10_paired_bge/summary.json
```
