# Static fusion replay

`run.py` reads the saved JSON snapshots in `../output` and writes one new ranking per query to `../output_new`. It does not call retrieval, embedding, a language model, or the live RAG pipeline.

## Why this strategy

The 435 existing snapshots match 435 evaluation queries. In 21 queries, a golden chunk appeared in the top 5 of at least one route but dropped below top 5 after the original fusion. The original score-weighted branch is used in 404 queries; the dimension-rank-aware branch is used in 31. The median semantic top1-to-top2 raw score gap is 0.0306; when the semantic top1 is golden it is 0.0482. These observations motivated a query-dependent route weight rather than a fixed one.

For a candidate, `score = (1 - alpha) * normalized_semantic + alpha * normalized_dimension + 0.3 * lexical`, where `alpha = 0.2 + 0.1 * max(0, 1 - semantic_top1_gap / 0.05)`. The lexical term measures query character 2/3/4-gram coverage in the candidate's first 4,000 text characters, weighted by inverse document frequency among this query's candidates and normalized to the best lexical candidate. A large semantic top1 gap keeps the dimension weight at 0.2; a small gap raises it toward 0.3. Gold labels never enter this score.

## Run

From the repository root:

```bash
python3 rag_core/retrieval_fusion/fusion_engine/fusion_new/run.py \
  --evaluation-results /home/humq/rag_core_runs/real_merged7_deepseek_v4pro_rerank_20260920_run1/evaluation_once/results.jsonl
```

Omit `--evaluation-results` to rank without generating labeled comparisons. `--input-dir` and `--output-dir` override the default snapshot folders. Each output JSON records the original rank and score, the new rank and score, route ranks, the lexical score, and candidate text. `output_new/summary.json` and `comparison.jsonl` record the evaluation.

## Replay result

| Metric (435 queries; 402 mapped gold) | Original | New |
| --- | ---: | ---: |
| Hit@1 | 223 | 243 |
| Hit@5 | 339 | 355 |
| Hit@10 | 368 | 372 |
| MRR@10 | 0.624445 | 0.662748 |
| nDCG@5 | 0.622212 | 0.657942 |
| Single-route top-5 gold lost after fusion | 21 | 10 |

The new ranking rescues 17 former top-5 misses and causes one former top-5 hit to fall out. That query is `我想了解明代帝陵的建筑规制，比较长陵和定陵在地面建筑方面的异同。` (rank 5 to 7). Examples of rescued cases include `西湖有没有能看到钱塘江的地方？` (rank 12 to 4) and `我想买往返观光车票，但只坐单程可以吗？票价是分开的。` (rank 14 to 4).

For a SHA-256 query split used during parameter exploration, the 349-query development portion goes from 269 to 284 Hit@5, while the 86-query held-out portion goes from 70 to 71. The held-out MRR@10 goes from 0.620976 to 0.650175. This is a replay on the available snapshot set; the held-out scores were inspected while choosing the final configuration, so they should not be treated as a fully independent future-data estimate.
