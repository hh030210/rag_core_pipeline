# 验证事实索引：实现与配对评测

## 目标

在不修改现有维度标签主索引的前提下，增加可追溯的“主体—关系—属性”事实索引，补齐由标签缺失造成的候选召回问题。

每一条文档事实均要求 `subject`、`object`、`evidence` 同时能在**同一个 chunk 原文**中定位；不满足该条件的 LLM 输出会被丢弃。因此，事实索引不是自由生成的新标签。

## 实现

- `code_jyx/llm_service.py`：抽取文档事实以及查询事实约束；关系受控为 `built_by`、`built_time`、`identity`、`located_in`、`feature`、`service_rule`、`price`、`transport_route`。
- `rag_core/fact_index.py`：校验三元组、建立 `subject/predicate/object -> chunk` 倒排索引，并以精确主体和受控关系进行匹配。
- `build_verified_fact_index.py`：可断点地抽取文档事实、构建 `fact_index_v1.json`。
- `build_query_fact_cache.py`：预解析查询事实，避免评测阶段调用 LLM；缓存按最终检索 query 而非样本 id 建立，以兼容重复 id。
- `rag_core/retrieval.py`：事实候选默认可关闭；启用时支持两种策略：按分数混排，或在维度候选头部之后尾插。
- `evaluate_fact_tail_ab.py`：在同一 Retriever、同一进程中依次运行对照与事实候选，消除跨进程缓存或随机性对 A/B 的干扰。

## 全量索引统计

| 项目 | 数量 |
| --- | ---: |
| 文档 chunk | 323 |
| 含验证事实的 chunk | 233 |
| 验证事实三元组 | 1,038 |
| 查询数 | 435 |
| 含事实约束的查询 | 376 |
| 查询事实约束 | 562 |

## @15 配对 A/B

评测口径：402 条已映射 Golden；33 条无法映射的 Golden 不进入指标分母。维度主索引、重写 query、高精度实体锚点与零候选词法兜底均保持一致。

事实候选不再按分数与维度候选直接竞争，而是在维度候选 Top10 之后插入。这样保留了已有前排排序，同时允许验证事实补齐 Top@15 的候选。

| 指标 | 对照组 | 验证事实尾插 Top10 后 | 变化 |
| --- | ---: | ---: | ---: |
| Hit@15 | 86.32% | 86.82% | +0.50pp |
| Gold Recall@15 | 79.66% | 80.26% | +0.60pp |
| MRR@15 | 0.5417 | 0.5422 | +0.0005 |
| nDCG@15 | 0.5795 | 0.5816 | +0.0021 |
| Top20 未命中 | 46 | 44 | -2 |

启用方式：

```powershell
python evaluate_dimension_only.py `
  --enable-verified-fact-index `
  --query-fact-cache result/<run>/query_fact_cache_rewritten_v1.json `
  --verified-fact-tail-start 10
```

`--verified-fact-tail-start 0` 保留旧的按分数混排行为；`10` 是本次验证有效的尾插位置。
