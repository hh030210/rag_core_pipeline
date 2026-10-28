# 群智完整流程：四项案例评价、六项最终评价（run34）

本目录为新版本，不修改同分支中 run33 的代码或结果。实验运行源码与提示逐字节复制自本轮 `code_snapshot`；`config.original.json` 是原配置，含历史机器路径但不含凭据。`launch_ten_point.py` 是新增便携启动器，初始化时只重新定位路径，不复制旧答案、评分、提示或审批。

完整结果：[run34结果目录](../../results/scenic_ten_point_four_train_six_final_deepseek_v4_pro_20261027_run34/README.md)。

## 完整流程

1. meta各生成一次QA、评价、QA更新、群智融合四阶段通用指令；QA指令直接用问题和top5生成答案，不再次生成提示。
2. 23道独立训练题进行案例迭代；更新可看到问题、参考答案、实际答案、top5、当前提示、前四项评分与理由及统一评价规范。初次作答不读取参考答案。任一主维下降、可评价范围改变或主维均分增长不足0.1分时停止，保留上一接受版本；四项均满10分时停止，30轮仅安全上限。参考相似性和答案优劣不参与训练选择。
3. 用问题与最终案例提示的归一化BGE向量等权组合，训练题上按余弦silhouette从候选K=2…10选簇数；可行时最小簇2，平局优先较小K。测试按问题向量路由。本轮算法实际选出4簇，不合并为全局提示。
4. 各簇输入训练Q、参考A、最终案例提示与统一要求，融合共同问题类型的针对性QA方法。
5. 在原训练题上回放群智QA，对照案例级答案，质性寻找融合丢失、过度泛化、冲突或数据/评价问题；不按固定分数驱动反馈，不用测试题反馈。修订回放后有退步或证据不足则保留旧版，各簇独立。本轮没有接受反馈修订，2条诊断不可核验也保留记录。
6. 冻结后378题，每题按所属簇QA＋问题＋原顺序top5进行一次逻辑答案生成。传输/空响应的有界重试另计，不使用参考作答、不做多答案融合。
7. 前四项同口径用于案例与最终评价；最终增加参考比较两项，并计算chapter3的13项机械指标。

作答上下文优先；内部按问题重排全部材料，仅用有用可靠内容，不固定保留数量，允许全部保留或舍弃。材料未包含的部分可用可靠内部知识补全，不确定则说明，不把补充知识冒称材料原文。不限定具体字数，完整回答所需内容后尽可能简洁。

## 评价规范

前五项为1–10整数，未添加锚点；逐题输出均保留 `{score, reasoning}`：

- 事实正确性：已陈述内容对不对；有上下文对应内容时核对使用，缺失部分用模型知识判断，不接收参考答案。
- 上下文忠实度：只核对材料对应内容，知识补充不纳入此项。
- 问题响应完整性：只看Q与实际A，逐问回应，不把参考额外内容当必答项。
- 回答精简性：只看Q与实际A，去无关扩展、重复和套话，不因必要条件或解释扣分。
- 参考语义相似性（补充）：只在最终比较中同时给Q、top5、参考A和实际A，比较两份答案在前四项下的相似程度。
- 答案优劣比较：最终单列1=生成答案更优、0=无法判断、-1=参考答案更优；前端显示“生成答案 / 无法判断 / 参考答案”，不是数值。与裸q+top5的优劣无关。

物理隔离：案例每答案两个评价请求（事实/忠实度、完整性/精简性）；最终增加一个参考比较请求。每指标只取第一次有效评价，不是重复三次评分。不适用null与服务缺失保留N/A，0是有效“无法判断”，不与缺失混淆。不计算六项混合总分。完整规范见 `ten_point_rubric.txt` 与结果的实际评价提示。

## 文件与运行入口

- `ten_point_workflow.py`：本轮唯一实验入口，绑定四项训练、六项最终、1–10整数和新视图。
- `ten_point_format.py`、`test_ten_point_workflow.py`：有界JSON结构恢复与17项离线测试；不改分、不删改理由字词。
- `knowledge_fallback_workflow.py`、`five_metrics_workflow.py`、`restored_meta_runner.py`：被新入口绑定的复用组件；其中历史文档或旧常量不是本轮运行口径，不单独运行旧入口。
- `collective_feedback*.py`、`knowledge_feedback_prompts.py`：训练题质性回放和反馈修订。
- `machine_protocol.py`、`embedding_runtime.py`、`five_metrics_report.py`：13项机械指标、BGE、原始运行报告。
- `audit_five_metrics_workflow.py`：独立核对精确请求、统一裁判、训练选择、单调用、评分与无损结构恢复。
- `frontend_source/`：实际前端构建源码和CSS，含文字优劣标签、最终理由、较差案例优先和可拖动上下区域；路径布局保留，以便原构建脚本直接运行。
- `config.original.json`、`user_meta.md`、`evaluation_rubric.txt`：实际配置与meta/评价规范。

Python 3.10+，本地中文BGE（原实验bge-large-zh-v1.5，CPU编码），可用的OpenAI兼容接口；模型名`DeepSeek-V4-Pro`需在自己的代理正确映射为非联网服务。发布不包含密钥、SSH脚本或代理修改，原协议使用`Bearer EMPTY`。依赖版本和远端模型变化可能导致不同结果，不保证逐字复现。

```bash
cd experiments/collective_ten_point_run34
python -m pip install -r requirements.txt
python -m unittest test_ten_point_workflow -v
python launch_ten_point.py init --run /your/new/run35 --embedding /your/models/bge-large-zh-v1.5 --base-url http://127.0.0.1:8910/v1 --workers 8
python launch_ten_point.py run --run /your/new/run35
```

四个语义审核闸均保留，不能自动批准不合规提示：阶段、簇、反馈阶段、最终反馈。先查看新目录的 `stages/`、`clusters/`、`collective_feedback/stages/`、`collective_feedback/final_clusters/` 及相应review文件，再按闸批准并续跑：

```bash
python launch_ten_point.py status --run /your/new/run35
python launch_ten_point.py approve --run /your/new/run35 --gate stage --reviewed
python launch_ten_point.py run --run /your/new/run35
# 后续依次 --gate cluster / feedback_stage / feedback
```

不要同时运行同一目录的多个进程。生成不合规时先查原始记录，不为追求高分重评或丢弃有效低分。启动器仅做离线初始化测试，本次发布未再次请求模型。

原结果复核/重建前端：将归档解压至一个新目录（保留全部层级），在本代码目录执行：

```bash
python audit_five_metrics_workflow.py /your/extracted/run34
python frontend_source/analysis/evidence_explainer/five_metrics_page.py /your/extracted/run34
```

## 本轮结果和限制

23训练题、42版本、算法4簇；378测试题中377份有效答案、375份完整评价。事实正确性9.807、忠实度9.863、完整性9.699、精简性8.784、参考相似性7.890；各项有效题数见报告，F1=0.506037。最终裁判认为生成答案更优224题、无法判断115题、参考答案更优36题。

12份已存在评分仅做无损结构恢复，原评分、理由、所有答案和机械指标保持不变，仅补5个从未取得的评价视图；3条服务空返回保留失败。所有1784次阶段请求（含失败）、冻结源码、输入、案例、群智和反馈、技术恢复证明均在归档中。

本轮只跑群智＋top5，没有重新跑裸基线，不宣称优于裸q+top5；新10分与旧100分不可直接比较。测试数据此前多次检视，结果为探索性。模型自身知识判断并非外部事实认证，已有评分理由矛盾已单独披露，未人工改分或据此重选提示。
