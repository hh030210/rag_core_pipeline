# 融合检索

根部只保留当前上线 fusion.py：
- fusion.py：adaptive_fusion 是当前线上 adaptive_report_v1 策略；Retriever 直接调用。fuse_retrieval_results 保留通用加权融合接口；融合快照保存函数也在此。
- experiments/06_dimension_score/code/run.py：维度分数离线基准。
- experiments/07_recommended_fusion/code/run.py：离线推荐融合实验，未上线。
- visualization/：页面、动态指标计算模块与详情数据；与 experiments/ 平级。

## 目录
所有实验采用 experiments/序号_名称/code 与 output。线上策略实现只在根部 fusion.py；离线策略的实现放在对应实验 code/。
| 目录 | 内容 |
|---|---|
| 01_online_snapshots | 线上融合快照与迁移后排名一致性验证 |
| 06_dimension_score | 维度分数离线基准的重放代码与输出 |
| 07_recommended_fusion | 推荐融合的重放代码与输出 |

当前仅保留线上融合基准、维度分数离线融合和维度角色校准推荐融合。各自代码与结果位于 code/ 和 output/。

## 入口与路径
从仓库根目录执行：
~~~bash
PYTHONDONTWRITEBYTECODE=1 /home/humq/envs/denoise_qa/bin/python -m rag_core.retrieval_fusion.experiments.06_dimension_score.code.run
PYTHONDONTWRITEBYTECODE=1 /home/humq/envs/denoise_qa/bin/python rag_core/retrieval_fusion/visualization/build_static_data.py
PYTHONDONTWRITEBYTECODE=1 /home/humq/envs/denoise_qa/bin/python -m rag_core.retrieval_fusion.experiments.07_recommended_fusion.code.run
~~~
也可使用 experiments.06_dimension_score.code.run 与 experiments.07_recommended_fusion.code.run 的模块入口。

线上融合快照默认写到 experiments/01_online_snapshots/output/。设置 RAG_FUSION_OUTPUT_DIR 可把某次实验快照写入该实验自己的 output/，避免冲突。
维度基准、推荐重放分别默认写到 experiments/06_dimension_score/output 和 experiments/07_recommended_fusion/output。
历史静态页面仍直接打开 visualization/index.html；数据构建脚本默认读取新目录。页面支持导入 evaluate results.jsonl 并从候选与Golden现算指标。

