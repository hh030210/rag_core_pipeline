# 可视化

从仓库根目录启动静态 HTTP 服务，再访问 `/rag_core/retrieval_fusion/visualization/`。不要用 file:// 打开。

融合数据直接读取 `../experiments/00_method_comparison/output/<策略名>/index.json` 和同目录逐查询 JSON；三个方法为 `online_adaptive`、`dimension_score`、`recommended_fusion`。主线参照使用 `fusion.py` 当前指定的方法目录，不再创建 `output/online`。

语义数据直接读取 `../../vector_retrieval/experiments/00_method_comparison/output/<策略名>/index.json` 和同目录逐查询 JSON，覆盖全部五种保留方法。

指标均由各方法的候选排序和 Gold 现场计算，不读取预存汇总指标。界面读取 `search.py`、`fusion.py` 的配置标注当前主线策略，并定时检查批次变化。更换策略后应重新评测以更新数据。

可视化目录只存界面代码与说明，不生成或保存评测 JSON。方法组织规则见 [../README.md](../README.md)。
