# 检索融合可视化

本目录只保存页面和指标计算代码，不保存实验结果，不生成数据副本。
页面通过 HTTP 直接读取以下三个实验的 JSON：

- `../experiments/00_method_comparison/output/online/dataset/index.json` 及 `details/`：语义、维度、线上融合候选，正文、查询分析和 Golden。
- `../experiments/00_method_comparison/output/dimension_score/dataset/index.json` 及 `details/`：独立输入候选、Golden、正文及维度分数融合候选。
- `../experiments/00_method_comparison/output/recommended_fusion/dataset/index.json` 及 `details/`：独立输入候选、Golden、正文及维度角色校准推荐融合候选与诊断。

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


## 指标总表与更新

指标区仅显示一张五行总表：语义检索、维度检索、线上融合来自01，离线融合来自06，推荐融合来自07。每行使用对应实验保存的候选与Golden，按当前分母选项计算。界面显示实际评测记录中的向量策略文件名，以及三组实验各自的生成时间；批次链不一致时提示运行run_all.py。

页面每30秒检查实验index批次，切回页面时也检查；发现新批次会重新读取候选并计算指标。手动“重新读取并计算”仍可立即更新。JSON请求使用no-store及时间戳参数；页面脚本带版本参数。正在运行的实验若未成功发布新批次，界面不会把旧候选冒充新结果。

数据由当前HTTP服务提供。如果打开的是本地复制的仓库，需同步experiments数据；界面不能从服务器自动更新另一份本地文件。


## 语义检索策略对比

页面顶部仅用一张表展示 `vector_retrieval/experiments/00_method_comparison/output/dataset/index.json` 的全部 `methods`。`semantic_comparison.js` 读取对应详情中的候选排序、Golden 和 `cost.logical_dense_searches`，现场计算 Hit@1/5/10/15、MRR@10、nDCG@5 及平均逻辑子检索次数，不读取预存指标、不写入数据或汇总文件。方法对比与融合实验分别使用各自的样本，表上注明分母和未映射排除数；逻辑子检索次数不等同于数据库调用次数（纯向量集成可在本地检索）。

刷新页面或点击“重新读取并计算”会更新表格；页面每 30 秒检查实验清单变化，窗口重新获得焦点也会检查。数据缺失或更新时批次不一致会明确报错，不回退旧分数。语义检索效果、融合介绍、融合对比、融合分析和详情均有独立边框。


## 全策略展示与当前主线标记

语义表展示方法清单中的全部五种策略，融合表展示 online_adaptive.py、dimension_score.py、recommended_fusion.py 三种独立策略，另保留语义/维度路线作为参照。自适应基准读取 experiments/00_method_comparison/output/online_adaptive/dataset；不再用当前线上融合行重复代替一项策略。分析和详情下拉框也可以选择三种策略，以及单独查看最近的主线评测批次。

`pipeline_strategies.js` 通过同一 HTTP 服务读取 vector_retrieval/search.py 的 VECTOR_RETRIEVAL_STRATEGY 和 retrieval_fusion/fusion.py 的 FUSION_STRATEGY，为介绍、表格及融合下拉框标注“当前主线”。每 30 秒、窗口焦点变化及手动刷新均重新读取。配置标记与已保存批次实际策略分别展示，修改配置不意味着评测结果已更新。
