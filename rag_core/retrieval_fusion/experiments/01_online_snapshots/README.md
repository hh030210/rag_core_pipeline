# 线上融合评测数据

执行 run.py evaluate，成功后整批覆盖本实验 output/dataset；失败保留旧数据。主流程不会触发 06 或 07。

仅生成 dataset/index.json 和 details/*.json。详情包括查询、检索查询、query_analysis、Golden、语义/维度/融合候选和正文；evaluation 字段保存完整逐条评测数据。可视化、向量实验和评测对比直接读取 dataset。不再生成 results.jsonl、summary.json、summary.md 或日志。
