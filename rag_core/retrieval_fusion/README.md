# Retrieval fusion

策略入口：`fusion.py`；默认 `FUSION_STRATEGY = "dimension_score.py"`。

所有三种保留策略与评测结果均在 `experiments/00_method_comparison`。详见该目录 README。

一键运行：
```bash
/home/humq/envs/denoise_qa/bin/python rag_core/retrieval_fusion/run_all.py
```

依次执行当前线上策略的 `run.py evaluate`、自适应融合基准、维度分数基准和推荐策略重排；每步成功覆盖自己的 output/dataset，失败停止后续步骤。旧 01/06/07 目录已迁移，不再使用。

可视化目录与 experiments 平级，通过 HTTP 服务从实验目录读取数据；不保存数据副本。
