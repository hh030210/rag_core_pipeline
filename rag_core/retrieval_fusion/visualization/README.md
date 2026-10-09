# 检索融合可视化

本目录只保存页面和指标计算代码，不保存实验结果，不生成数据副本。
页面通过 HTTP 直接读取以下三个实验的 JSON：

- `../experiments/01_online_snapshots/output/dataset/index.json` 及 `details/`：语义、维度、线上融合候选，正文、查询分析和 Golden。
- `../experiments/06_dimension_score/output/dataset/index.json` 及 `details/`：独立输入候选、Golden、正文及维度分数融合候选。
- `../experiments/07_recommended_fusion/output/dataset/index.json` 及 `details/`：独立输入候选、Golden、正文及维度角色校准推荐融合候选与诊断。

页面不读取预存指标，始终从各实验自己的候选和 Golden 现算 Hit@1/5/10/15、MRR@10、nDCG@5。每组默认以自己的 Golden 已映射查询为分母，可切换全部查询。线上、离线、推荐分别有独立批次，不要求它们相同。

06、07 的逐题 JSON 同时保存自己的输入、Golden、正文和融合结果。更新主流程不会改变它们的数据。命中图、筛选和详情使用所选策略的独立数据；线上组的语义成绩始终来自 01 最新评测。

## 打开页面

在服务器仓库根目录执行：

```bash
python3 -m http.server 8000 --bind 127.0.0.1
```

通过 SSH 端口转发或远程开发工具转发 8000 端口，然后访问：
`http://127.0.0.1:8000/rag_core/retrieval_fusion/visualization/`。
Windows 完整仓库可运行 `start.bat`。必须通过 HTTP 打开，直接双击 index.html 无法读取 JSON。

## 更新实验结果

- run.py evaluate：只更新 01 线上评测。
- experiments/06_dimension_score/code/run.py：单独更新离线维度分数实验。
- experiments/07_recommended_fusion/code/run.py：单独更新推荐融合实验，默认使用 06 保存的输入。

三者都只写自己的输出。运行主流程后无需也不会自动重跑实验。刷新或“重新读取并计算”只重读已有 JSON，不运行检索或实验。

若某个实验尚未运行，页面禁用相应策略。单个实验内部批次不完整会报错；不会从其他实验补齐候选或 Golden。缺少整个实验目录时仍可显示线上结果。

“导入评测结果”支持 results.jsonl。导入仅存在页面内存，并禁用文件中不存在的离线、推荐策略。刷新或“返回最新评测”重新读取各实验自己的结果。
