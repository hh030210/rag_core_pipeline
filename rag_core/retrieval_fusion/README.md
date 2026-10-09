# 融合策略与实验

根目录 fusion.py 为线上融合策略。experiments 包含三个独立实验：

- 01_online_snapshots：run.py evaluate 成功后整批覆盖 output/dataset。
- 06_dimension_score：独立运行 code/run.py，读取运行时的 01/output/dataset，在自己的 dataset 保存输入和离线候选。
- 07_recommended_fusion：独立运行 code/run.py，默认读取 06 保存的输入和候选，在自己的 dataset 保存推荐候选。

每个 output 仅保留 dataset/index.json 和 dataset/details/*.json。index 保存批次、来源、配置及详情索引；details 保存查询、Golden、候选、正文及必要诊断。线上 details 中的 evaluation 保存完整逐条评测字段，供向量实验和评测对比读取。不会生成独立 JSONL、汇总、坏例或日志文件。运行失败保留旧输出；各实验不会自动运行或修改其他实验。

visualization 与 experiments 平级，只保存界面代码，直接读取各实验 dataset 并计算指标。
