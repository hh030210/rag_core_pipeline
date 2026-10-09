# 查询组合对比实验

所有新增代码、日志、验证文件和结果均保存在本目录下。仅查询既有 Qdrant collection，不入库、不调用 LLM、不改变主检索策略。使用 PYTHONDONTWRITEBYTECODE=1，避免导入模块产生目录外的缓存。

## 输入与实验设计
输入为之前 evaluate 保存的 results.jsonl，包含 question、retrieval_query、query_analysis.spot_names、gold.chunk_ids。优先使用显式 subqueries 列表，否则从保存的 retrieval_query 按 | 重建子查询列表；该重建假设 | 是分隔符。原始查询优先 original_query，否则 question/query。空子查询回退为原始查询，并记录标记。未映射 Golden 的样本统计并排除，所有方案使用同一批样本。

| 方案 | 方法 | 检索次数 |
|---|---|---|
| original_only | 只编码原始查询 | 1 |
| subqueries_concat | 子查询用空格+竖线+空格拼接，当前策略对照 | 1 |
| original_plus_subqueries_concat | 原始查询在前，再拼接全部子查询，保留重复 | 1 |
| original_plus_subqueries_dedup | 上述文本精确去重后拼接 | 1 |
| subqueries_comma | 子查询用中文逗号拼接 | 1 |
| original_subqueries_centroid_w2 | 各唯一查询向量先归一化；原始查询权重2、其余权重1；质心归一化 | 1 |
| original_subqueries_rrf | 原始查询和各唯一子查询独立召回，等权 RRF(k=60)，截断到同一深度 | 唯一查询数 |

每次搜索复用 VectorRetriever.search，保持 chunk_text_vec、召回池大小和景区后置过滤一致。搜索后按同一 pool 截断；景区过滤可能使返回数少于 pool。RRF 每个分支也使用同一 pool，但合并前候选总数更多，应同时考虑检索成本。运行内重复文本搜索可复用，属于计算缓存。

## 指标与结果
直接复用 rag_core.evaluation.route_metrics，计算 K=1,5,10,15,20 的 Hit、Gold Recall、MRR、nDCG，以及命中时首个 Golden 排名和未命中数。兼容旧实验的 Hit@1/5/10/15/20、Gold Recall@15、MRR@10、nDCG@5。

- results.jsonl：原始查询、子查询、过滤范围、各方案实际输入、完整候选ID/分数/排名和逐题指标。
- summary.json / summary.md：配置、输入SHA256、编码器实际模式、样本排除数、指标汇总和旧排名前缀复现率。
- comparison.jsonl：相对 subqueries_concat 的逐题 Hit@15、MRR@10、nDCG@5 变化。
- smoke/full日志：运行进度与错误信息。

保存的旧结果可能被 top_k 截断，因此复现检查比较旧列表可用前缀，不声称验证不可见排名。--limit 是冒烟运行；正式结论必须使用全部样本。已有数据集上的结果属于探索，采用前需独立测试集验证。

## 运行
从 /home/humq/rag_core_pipeline 执行：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/humq/envs/denoise_qa/bin/python -m rag_core.vector_retrieval.experiments.query_composition.run \
  --run-dir /home/humq/rag_core_runs/real_merged7_sdu_v4pro_full_20260921_run1 \
  --input-results /home/humq/rag_core_runs/real_merged7_sdu_v4pro_full_20260921_run1/evaluation_after_pull_4896b80/results.jsonl \
  --model-path /home/humq/rag_db_silm/model/bge-m3 \
  --device cpu --pool 20
```

--run-dir、--input-results、--model-path 必填。--device 默认cpu；--pool 默认20，必须>=20；--limit 默认0（全量）；--output-dir 默认本目录 outputs/时间戳，仅接受本目录以内的新路径，已存在会拒绝覆盖。


## 配对诊断
完成后运行 `PYTHONDONTWRITEBYTECODE=1 /home/humq/envs/denoise_qa/bin/python -m rag_core.vector_retrieval.experiments.query_composition.analyze <本实验outputs目录>`，生成 analysis.md（相对对照组的指标差值、挽回/丢失问题数）和 diagnostics.json（景区与子查询数量分组）。

## 本次全量结果（2026-10-09）
输入435条，评估402条已映射样本，排除33条；pool=20，CPU，七方案比较阶段约228秒（不含模型加载）。编码器实际使用SentenceTransformer回退，与保存的旧语义结果可用前缀402/402一致。

正式结果：outputs/20261009_193606/；冒烟结果：outputs/20261009_193526/。

只用原始查询 Hit@15=94.03%、MRR@10=0.6683、nDCG@5=0.6676；当前子查询拼接分别为91.29%、0.6278、0.6161；原始加子查询直接拼接为91.54%、0.6273、0.6175；向量质心为94.53%、0.6646、0.6646。

原始查询独立检索的排序指标最好，向量质心的Hit@15和Gold Recall@15最好。单纯追加原始文本没有带来明显收益。质心相对对照挽回14题、丢失1题；原始查询独立检索挽回17题、丢失6题。探索结果尚未在独立测试集验证。完整表格见summary.md，逐题差异见analysis.md/comparison.jsonl。
