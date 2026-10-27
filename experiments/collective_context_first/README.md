# 群智完整流程：上下文优先与内部知识补全

本目录发布最后完成的 **run33** 实现；不是旧三维评价或双调用复核实验。`*.py`（除便携启动器）及提示文本逐字节复制自实际运行的 `code_snapshot`，`config.original.json` 是原配置，仅有历史机器路径、没有凭据。没有改动仓库已有代码。原始完整运行见 [结果目录](../../results/scenic_context_first_knowledge_deepseek_v4_pro_20261026_run33/README.md)。

## 完整流程

1. meta仅生成一次 QA、评价、更新、群智融合四阶段通用指令；QA指令直接接收问题与top5生成答案，不继续生成提示。
2. 23道独立训练题：初始QA作答→统一五指标评价→根据问题、参考、实际答案、上下文、当前提示及各项评分理由更新QA。某主维下降、可评价范围变化或主维平均增幅不足1分即停止，保留上一接受版本；辅助参考相似性不参与接受。30轮是安全上限。
3. 问题与最终案例提示的归一化BGE向量等权组合，候选K=2…10，按余弦silhouette选K（可行时最小簇2，平局选较小K）；仅训练题选K。测试题按问题向量路由。本轮实际 **8簇**，不是全局单提示。
4. 每簇以训练Q、参考A、最终案例QA及统一标准融合针对共同问题类型的群智QA。
5. 群智提示在原训练题回测，质性比较案例级与群智答案：寻找融合丢失、过度泛化、冲突或数据/评价问题，有依据才更新。候选重新回测并核对改善/退步；不使用固定五分来驱动此环节，不读取测试答案。最多3轮。本轮没有候选被接受，全部如实保留。
6. 冻结后378题，每题只由所属簇提示＋原问题＋原顺序top5进行**一次逻辑答案调用**。网络/空响应重试单独保留；不做多答案融合，不读取参考作答。
7. 最终五指标与案例迭代同口径；另计算原chapter3的13项机器指标。

## 评价范围

五项0–100、越高越好：事实正确性、上下文忠实度、问题响应完整性、回答准确性（需求契合/去无关冗余）、参考语义相似性（辅助）。具体规范见 `knowledge_fallback_rubric.txt`。

物理隔离3个请求：事实/忠实仅看Q+上下文+实际A；完整性/回答准确性仅看Q+实际A；参考相似性仅看参考A+实际A。**每指标一次有效评价，不是重复评分3次**；适用性null不填0，不计算五项混合总分。上下文优先，缺失部分允许可靠模型内部知识补全；忠实度只评价材料对应部分。裁判的知识判断不是外部事实认证。

## 文件说明

- `knowledge_fallback_workflow.py`：唯一新版实验入口，绑定五指标/知识补全/视图隔离。
- `five_metrics_workflow.py`、`restored_meta_runner.py`：底层案例迭代、API首份有效缓存、动态聚类、单调用推理；不要单独运行旧底层入口。
- `collective_feedback*.py`、`knowledge_feedback_prompts.py`、`audit_collective_feedback.py`：训练回放、质性归因、修订接受与来源核验。
- `knowledge_view_format.py`：有限JSON格式恢复，不改变分值、理由；`test_knowledge_fallback.py`含16项离线测试。
- `embedding_runtime.py`：本地BGE；`machine_protocol.py`只复用原13项机器评分函数，旧CLI/文档不代表新版LLM评价。
- `five_metrics_report.py`：结果/提示导出；`rejudge_five_runner.py`仅为底层导入依赖，**不使用其旧独立重评入口**。
- `launch_collective.py`：新增便携初始化、审核和续跑工具；不含SSH、不设定时任务、不复制历史答案或分数。
- `verify_results.py`：只读核验归档全部字节/冻结哈希、原始请求数、逐题汇总分数与失败数；不会调用模型。
- `config.original.json`、`user_meta.md`、`evaluation_rubric.txt`：实际配置、meta、标准。

## 在自己的服务器重新运行

需要Python 3.10+，可用的OpenAI兼容本地/学校接口，以及已下载的中文BGE模型（原实验为bge-large-zh-v1.5，CPU编码）。原接口接受`Bearer EMPTY`且模型名`DeepSeek-V4-Pro`必须映射到非联网模型；该发布不包含或修改代理。若服务要求密钥，请自行通过安全的本地代理配置，不把密钥写进仓库。

```bash
cd experiments/collective_context_first
python -m pip install -r requirements.txt
python -m unittest test_knowledge_fallback -v
python verify_results.py
python launch_collective.py init --run /your/new/run34 --embedding /your/models/bge-large-zh-v1.5 --base-url http://127.0.0.1:8910/v1 --workers 8
python launch_collective.py run --run /your/new/run34
```

初始化从归档只取23/378题及top5，不复制旧提示、答案或评分；在新运行目录保存独立runtime/config与源码/输入哈希。接口、依赖版本和模型服务变化可能导致不同结果，不能保证逐字复现。便携启动器仅离线验证，本次发布没有再次调用模型。

保留实际流程的四个语义审核闸：`awaiting_stage_review`、`awaiting_cluster_review`、`awaiting_feedback_stage_review`、`awaiting_feedback_review`。每次先查看新运行目录的 `stages/`、`clusters/`、`collective_feedback/stages/` 或 `collective_feedback/final_clusters/`，确认生成层级、证据策略、评价边界、问题类型均符合要求后执行对应命令，再`run`续跑：

```bash
python launch_collective.py status --run /your/new/run34
python launch_collective.py approve --run /your/new/run34 --gate stage --reviewed
python launch_collective.py run --run /your/new/run34
# 后续三个闸依次使用 --gate cluster / feedback_stage / feedback
```

不自动批准不合规提示；生成格式/语义失败应先核查原始输出，不能以追求高分为由丢弃有效低分或反复重评。不要同时启动同一运行目录的多个进程。报告生成在最后自动完成，保留失败与缺失记录。

## 本轮结果与限制

23训练题42版本；378计划测试、377有效答案、376完整评价；1834次全部阶段原始请求（含失败/重试）。事实97.9782、忠实99.0362、完整94.5612、回答准确93.3830、参考相似81.2207，F1=0.536649。不同指标适用题数见完整报告。

本轮**没有新裸基线**，没有接受群智反馈修订，不宣称已优于裸q+top5。历史多次检视的数据属于探索性，不能冒充独立显著性证据。两条服务空返回失败、裁判分数与理由不一致、有限格式恢复及全部原始记录都保留在结果目录/归档中，不改分、不填零、不隐藏。

`audit_archive.zip`保留原字节（仅排除pycache），其中绝对路径是历史来源信息，不是新运行依赖。`archive_inventory.json`列出所有SHA256；解压到新目录后可按清单校验。GitHub预览HTML不执行脚本，下载`旅游群智_完整流程.htm`后直接打开即可逐题浏览实际提示、答案、top5和最终评分理由。
