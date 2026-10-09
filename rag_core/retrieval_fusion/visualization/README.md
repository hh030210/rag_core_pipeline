# 检索融合可视化

打开 `index.html` 查看语义检索、维度检索、线上融合、维度分数离线基准和推荐融合。默认选中推荐策略；选择框只提供推荐融合和离线基准，控制问答列表、筛选及命中关系图。逐题详情保留五列，便于直接比较。

首表全部指标采用百分制。Hit 分母为 402 条 Gold 已映射查询；MRR@10 和 nDCG@5 按全部 435 条快照计算。33 条未映射问题有独立标记和筛选，不纳入命中统计。

命中关系图按仅维度命中、仅语义命中、都命中、都未命中分组，条块总长度体现查询数，颜色表示线上与当前离线融合的命中关系。推荐策略详情显示原始 gap、同首位保护、维度角色校准是否触发、局部证据与事实锚点。点击 chunk 编号可查看正文。

推荐策略 Top-5 为 362/402（90.05%），相对维度分数基准救回 8 条、掉出 0 条。使用“当前离线 Top-5 未命中”查看 40 条 badcase；筛选也支持相对线上／离线基准救回或掉出的查询。

重建静态数据（仓库根目录）：

```powershell
py -3 rag_core/retrieval_fusion/fusion_engine/fusion_new/run.py
py -3 rag_core/retrieval_fusion/visualization/build_static_data.py
py -3 rag_core/retrieval_fusion/fusion_engine/fusion_new/recommended.py
```

基准数据为 `snapshot_index.js`、`snapshot_details/`；推荐数据为 `recommended_index.js`、`recommended_details/`。详情按题懒加载。方法与结果口径见 `../fusion_engine/fusion_new/README.md`。
