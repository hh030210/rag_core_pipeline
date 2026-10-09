# 向量检索查询组合实验

对比原始查询、子查询拼接、原始加子查询拼接、去重拼接、逗号拼接、加权向量中心及 RRF。保存逐条候选及 Hit@5/10/15 等指标，RRF 调用成本可由唯一 component_texts 数量计算，其他策略各检索一次。

从仓库根目录运行：

```bash
python -m rag_core.vector_retrieval.experiments.01_query_composition.code.run --run-dir <已有索引目录> --input-dataset rag_core/retrieval_fusion/experiments/01_online_snapshots/output/dataset --model-path <模型目录> --device cpu
```

--input-results 作为 --input-dataset 的旧名称兼容，但参数值必须是 dataset 目录或 index.json。--limit 0 默认评测所有可映射查询，正数限制查询数量用于冒烟验证。

默认成功后整批覆盖本实验 output，失败保留旧数据。仅生成 dataset/index.json 和 details/*.json，不创建时间戳目录、JSONL、汇总或日志文件。现有 402 条正式结果已迁移至 output/dataset。

查看配对及分组分析：

```bash
python -m rag_core.vector_retrieval.experiments.01_query_composition.code.analyze rag_core/vector_retrieval/experiments/01_query_composition/output
```

分析直接读取 dataset，计算结果输出到终端，不生成文件。
