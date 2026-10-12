# 最佳维度检索结果：融合测试交付

来源实验：`evaluation_overall_slot_coverage_rerank_bge`。

评测口径为 402 条可映射 Golden：Hit@5 为 **70.90%**，Gold Recall@5 为 **62.76%**；Hit@10 为 **81.84%**，Gold Recall@10 为 **75.46%**。

## 文件说明

- `dimension_candidates_top20_for_fusion.jsonl`：融合测试的唯一输入。每行一个 query，包含维度检索 Top20 候选及其维度、实体、事实和重排特征；**不包含 Golden**。
- `dimension_candidates_top20_with_gold_for_eval.jsonl`：仅供离线评测融合结果，含 `gold` 和 `reference_answer`，不得作为融合模型输入。
- `best_dimension_retrieval_summary.json`：本次维度检索完整评测指标与开关配置。
- `manifest.json`：导出格式、来源和文件清单。

## 一键复现最佳维度检索

完整仓库 clone 后，在 Windows 双击根目录的 `run_best_dimension_retrieval.bat`，或在 PowerShell 中执行：

```powershell
./run_best_dimension_retrieval.ps1
```

脚本会创建虚拟环境、安装 `requirements.txt`、自动下载一次公开的 BAAI/bge-m3（约 4GB；已存在时跳过），并使用本仓库提交的索引资产与 query 事实缓存运行全量评测。检索评测不调用 LLM。若已有共享模型，可传入路径：

```powershell
./run_best_dimension_retrieval.ps1 -ModelPath D:\models\bge-m3
```

输出写入 `result/evaluation_best_dimension_repro/summary.json`。

## 融合建议

以 `id` 对齐其他召回路的 query；以 `chunk_id` 进行候选去重。`score` 是当前维度路最终排序分数，`dimension_score` 是维度匹配原始分；`matched_dimensions`、`matches` 和各类 `*_rerank_bonus` 可作为融合特征。保留 `chunk_text_full` 以便下游重排。
