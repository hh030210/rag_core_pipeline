# 融合检索可视化

这是一个纯静态页面，不需要启动 Python 服务。双击 `index.html` 或 `start.bat` 即可查看仓库内生成的快照。

页面顶部展示四路检索的五项评估指标：Hit@1、Hit@5、Hit@10、MRR@10 与 nDCG@5。Hit 显示命中条数及有 Gold 映射的查询数；MRR@10 和 nDCG@5 显示 0–1 分数，并按全部快照数计算。页面同时注明 Gold 映射数和未映射数。下方可以按问题查看语义检索、维度检索、原融合与新融合候选，搜索和筛选 badcase，展开完整排名，并点击 chunk 编号查看保存的正文。

页面运行时只读取本目录下的 `snapshot_index.js` 和 `snapshot_details/` 静态数据，不调用检索、模型或外部服务。更新 `fusion_engine/output/`、`output_new/` 快照后，在仓库根目录重新生成静态数据包：

```bash
py -3 rag_core/retrieval_fusion/visualization/build_static_data.py
```

生成器会检查原融合和新融合的查询及候选集合是否一致。若 `output_new/comparison.jsonl` 不存在，页面仍可显示候选对比，但 Gold 排名与评估指标将为空。

## 当前融合策略

“原融合”展示 `output` 中实际保存的排名；页面不会用稳定融合函数重新计算历史排名。

“新融合”展示 `fusion_new` 当前离线策略生成的结果：

```text
g = max(s[i] - s[i+1]), i = 1,...,min(9, 语义候选数-1)
u = max(0, 1 - g / 0.1)
λ = 0.1 + 0.025 * u
F = S* + λ * L
```

`S*` 使用原始语义得分；语义路未召回的候选估计为 `max(0, 最低已召回语义分 - 0.025)`，语义路为空时为 0。`L` 是查询字符 2/3/4-gram 在候选前 4,000 字符中的 IDF 加权覆盖率，再按候选最大值归一化。当前维度分数权重为 0，维度路仍提供候选。
