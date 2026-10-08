# 最新离线融合策略

`fusion_new` 只保留最新离线策略。`run.py` 默认读取 `../output`，并将重排结果覆盖到 `../output_new`。稳定融合实现位于 `../fusion.py`。

## 当前公式

```text
g = max(s[i] - s[i+1]), i = 1,...,min(9, 语义候选数-1)
u = max(0, 1 - g / 0.1)
F = S* + (0.1 + 0.025*u) * L
```

`S*` 使用原始语义分；语义路未召回时估计为 `max(0, 最低已召回语义分 - 0.025)`，语义路为空时为 0。维度路提供候选，当前维度分数权重为 0。`L` 是查询字符 2/3/4-gram 在文本前 4,000 字符中的 IDF 加权覆盖率，再按候选最大值归一化。归一化检索分仅用于审计。

语义头部 `0.89, 0.88, 0.70` 的最大相邻断层为 0.18，位置在第二、第三名之间。

## 运行

从仓库根目录直接运行，无需选择版本：

```powershell
py -3.14 rag_core/retrieval_fusion/fusion_engine/fusion_new/run.py --evaluation-results result/real_merged7_deepseek_v4pro_rerank_20260920_run1/alpha_sweep_20260921/alpha_0.2/evaluation/results.jsonl
```

`py -3.14` 可替换为环境中的 `python`。不传标签也能排名；`--input-dir`、`--output-dir` 可更换目录，`--config` 可指定参数 JSON。默认参数位于 `selected_config.json`。

## 输出与结果

`output_new` 保存最新候选排序、`summary.json`、`comparison.jsonl`、`badcase_analysis.jsonl`。评测基准为输入快照保存的原融合结果。`evaluation_analysis.json` 保存本次研究的开发/保留结果，其中前次离线实验仅作历史对比。

| 指标（435 查询，402 条有映射标签） | 原融合快照 | 最新离线策略 |
| --- | ---: | ---: |
| Hit@1 | 223 | 247 |
| Hit@5 | 339 | 357 |
| Hit@10 | 368 | 374 |
| MRR@10 | 0.624445 | 0.674488 |
| nDCG@5 | 0.622212 | 0.668983 |

相对前次离线实验，Hit@5 从 355 到 357；94 条保留查询从 75 到 74。数据此前用于调参，第一轮保留结果也被查看过，这是回顾性回放，稳定性仍需新数据检验。详细公式、消融和 badcase 见 [EXPERIMENT.md](EXPERIMENT.md)。

检查命令：

```powershell
py -3.14 rag_core/retrieval_fusion/fusion_engine/fusion_new/test_strategies.py
```
