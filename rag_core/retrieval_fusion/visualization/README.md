# 融合检索可视化

页面是纯静态 HTML，无需运行 Python 服务或选择数据目录。直接打开 `index.html` 即可查看当前快照。页面从 `snapshot_index.js` 读取问题列表，并在选中问题时加载 `snapshot_details/` 中对应的静态详情文件；不会触发检索、模型调用或新的融合计算。

快照更新后，在仓库根目录重新生成静态数据包：

```bash
py -3 rag_core/retrieval_fusion/visualization/build_static_data.py
```

`output_new/comparison.jsonl` 若存在，会为页面提供 Gold 数量及语义、维度、原融合、新融合的首个 Gold 名次，用于 badcase 筛选和统计；没有该文件时仍可查看四列候选，但不显示 Gold 指标。这个比较文件本身位于 `output_new`，页面不读取外部评测目录。

在 Windows 上可以双击 `index.html` 或 `start.bat`。数据包按需加载单题详情，打开页面时不会一次性读取全部正文。

页面默认显示各路 Top-5，可展开完整候选；点击 chunk 编号查看原始快照中的正文。原融合与新融合按相同文件名、query 和候选集合配对，发现不一致时会提示错误，避免错误对比。

## 当前融合策略的介绍

“原融合”列展示 `output` 中保存的实际融合排名，包含历史分支；稳定函数 `fusion_engine/fusion.py` 的定义是分别 Min–Max 归一化后计算 `(1 - α) * semantic_norm + α * dimension_norm`。页面不调用该函数重新排名。

“新融合”列展示 `fusion_new` 当前策略在 `output_new` 中生成的结果：

```text
g = max(s[i] - s[i+1]), i = 1,...,min(9, 语义候选数-1)
u = max(0, 1 - g / 0.1)
λ = 0.1 + 0.025 * u
F = S* + λ * L
```

`S*` 使用原始语义得分；语义路未召回的候选，估计为 `max(0, 最低已召回语义分 - 0.025)`，语义路为空时为 0。`L` 是查询字符 2/3/4-gram 在候选前 4,000 字符中的 IDF 加权覆盖率，再除以候选的最大覆盖率。当前维度分数权重为 0，维度路仍提供候选；归一化检索分仅用于审计。

每题展示快照中保存的头部原始分数、最大断层及位置、词面权重、维度权重和缺失语义估计值。新融合候选卡片区分原始语义分与估计分，并展示词面、维度的实际分数贡献。

badcase 统计与筛选统一以 `output` 原融合为基准，通过 `output_new/comparison.jsonl` 动态计算，不以之前的离线实验结果为基准。
