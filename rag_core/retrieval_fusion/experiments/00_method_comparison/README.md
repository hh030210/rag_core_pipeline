# Fusion method comparison

所有保留策略集中在 `code/`，每个 Python 文件都可用于线上流程：

|文件名|策略|
|---|---|
|online_adaptive.py|保留的自适应线上基准，按候选数和两路 Top-5 重叠选择分数分支或维度名次分支|
|dimension_score.py|维度分数基准，默认：语义原分 + 0.05×维度归一分 + 0.10×词面分；缺失语义估计为 max(0, min(S)−0.025)|
|recommended_fusion.py|维度角色校准：两路 gap、角色约束、局部窗口和事实锚点校准|

在 `rag_core/retrieval_fusion/fusion.py` 开头设置 `FUSION_STRATEGY = "dimension_score.py"`。运行中的服务应重启；下一次 evaluate 才产生新策略结果。每个文件提供统一 `fuse(...)` 接口。线上与离线调用同一评分代码，不使用 Golden 排序。

目录：
```
00_method_comparison/
  code/                         # 只有三种可选策略 .py
  runtime.py                    # 候选准备与完整 payload 恢复
  run.py                        # 一键评测及对比
  replay_adaptive.py
  replay_dimension.py
  replay_recommended.py
  verify.py
  output/
    online_adaptive/dataset/     # 独立自适应基准
    online/dataset/              # 当前配置的线上策略结果
    dimension_score/dataset/     # 独立维度分数基准结果
    recommended_fusion/dataset/  # 独立推荐策略结果
```

从仓库根目录执行：
```bash
PYTHONDONTWRITEBYTECODE=1 /home/humq/envs/denoise_qa/bin/python -m rag_core.retrieval_fusion.experiments.00_method_comparison.run
# 等价旧入口
PYTHONDONTWRITEBYTECODE=1 /home/humq/envs/denoise_qa/bin/python rag_core/retrieval_fusion/run_all.py
# 单独重排，各自覆盖自己的 output
PYTHONDONTWRITEBYTECODE=1 /home/humq/envs/denoise_qa/bin/python -m rag_core.retrieval_fusion.experiments.00_method_comparison.replay_dimension
PYTHONDONTWRITEBYTECODE=1 /home/humq/envs/denoise_qa/bin/python -m rag_core.retrieval_fusion.experiments.00_method_comparison.replay_recommended
# 保存排名、线上接口、空路及非法文件名校验
PYTHONDONTWRITEBYTECODE=1 /home/humq/envs/denoise_qa/bin/python -m rag_core.retrieval_fusion.experiments.00_method_comparison.verify
```

`run.py evaluate` 只覆盖 `output/online/dataset`，不会更新其他策略的数据。两种重排都从原始语义/维度候选准备固定的自适应基准并列顺序，避免当前线上配置影响各方法公式或并列排序。独立重排不调用模型或数据库。所有数据只在各自 dataset 中，不生成额外汇总文件。可视化直接读取这三个路径，并标出已保存批次实际采用的线上策略。

一键入口现在依次执行 evaluate、自适应基准重排、维度分数基准重排、推荐策略重排。新增自适应基准读取主线候选并仅覆盖 output/online_adaptive/dataset，不使用当前主线融合排序来替代该策略。
