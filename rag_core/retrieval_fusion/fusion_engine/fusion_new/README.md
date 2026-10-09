# 离线融合

仅保留两个离线方案：`run.py` 的维度分数基准，以及 `recommended.py` 的推荐融合。线上融合保留在上一级 `fusion.py`。

## 维度分数基准

`F = S* + 0.05 × D_norm + 0.10 × L`。语义路缺失时 `S* = max(0, 最低语义分 − 0.025)`；维度路缺失时贡献为 0；`L` 为 IDF 加权字符 2/3/4-gram 覆盖率，按查询候选中的最高值归一化。

## 推荐融合

固定配置集中在 `recommended.py` 的 `CONFIG`，不再执行参数搜索，也不依赖旧实验脚本。

1. 使用原始维度分／最高维度分，维度权重 0.0375、词面权重 0.075；语义缺失时取本路最低原始分。
2. 两路首位一致，且至少一路 Top-5 有至少 3 个候选、相对 gap ≥ 2%、突出程度 ≥ 5 倍时，将得分与线性名次分混合 62.5%，并将维度和词面贡献乘 0.375。语义名次分映射回本路得分范围。大 gap 不代表正确，不直接删除首位。
3. 维度贡献乘 0.85；用 320 字、半窗口步长的局部文本覆盖替换 75% 原词面分；语义贡献再混合 15% 的线性名次分。缺失语义时不增加语义名次分。
4. 仅当约束属于 `entity_name`、`geo_admin`、`geo_position`，没有具体 POI，且语义相对 gap ≥ 1% 时，启用角色校准：维度贡献再乘 0.25，词面贡献乘 0.5，事实锚点加分权重 0.015。其他查询保留步骤 3 的维度贡献。

相对 gap 为 `(Top1−Top2)/abs(Top2)`；突出程度为 gap／Top2 至 Top5 的平均相邻间隔。局部文本和锚点覆盖使用当前查询候选池的 IDF 并归一化，文本最多取快照保存正文前 4000 字。事实锚点来自 `query_analysis.fact_anchor_terms` 和约束 `intent_terms`。接入生产需要提供相同的查询解析字段。

## 当前结果

| 策略 | Top-1 | Top-5 | Top-10 | MRR@10 | nDCG@5 |
| --- | --- | --- | --- | --- | --- |
| 维度分数基准 | 249/402（61.94%） | 354/402（88.06%） | 374/402（93.03%） | 67.73% | 66.76% |
| 推荐融合 | 259/402（64.43%） | 362/402（90.05%） | 374/402（93.03%） | 68.76% | 68.15% |

推荐相对基准救回 8 条、掉出 0 条。剩余 40 条 Top-5 badcase：10 条候选池缺少 Golden、30 条已有 Golden 但排名靠后。

Hit 分母为 402 条 Gold 已映射查询；MRR 和 nDCG 以全部 435 条快照为分母。Gold 仅用于报告，不参与排序。数字描述当前历史样本的表现。

## 重建

在仓库根目录依次运行：

```powershell
py -3 rag_core/retrieval_fusion/fusion_engine/fusion_new/run.py
py -3 rag_core/retrieval_fusion/visualization/build_static_data.py
py -3 rag_core/retrieval_fusion/fusion_engine/fusion_new/recommended.py
```

基准结果写入 `output_new`，推荐摘要和 badcase 写入 `output_recommended`。推荐可视化数据使用 `recommended_index.js` 和 `recommended_details/`；只读取已有快照，不请求模型或检索服务。
