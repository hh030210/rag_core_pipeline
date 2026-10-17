# 当前新流程（2026-10-17）

本分支基于上游 `main` 的 `af2dabb46ecd338c2db6ee3abd5e7e1a8c70ec6a`，单独保存新流程，不修改上游已有分支或历史实验结果。

## 方法与入口

1. 从固定五模块 **meta-prompt** 为每个训练案例单独生成初始提示，不读取预写的成品初始提示。
2. 在独立训练案例的 **golden 上下文** 下优化。三个训练维度为 AQ、Accuracy、Completeness，均为 0–100 分；总分是算术平均。任一维度下降或总分提升不足 1 个绝对分值点，拒绝该候选并停止；最多 30 次改写作为保护上限。
3. 按训练问题与参考答案的向量做 QA 聚类（默认 6 簇），各簇合成群智提示。推理路由仅使用测试问题，不使用测试答案。
4. 根据缓存的检索上下文生成答案，不把 golden 插入测试上下文。
5. 最终沿用 chapter3 机械指标与 Qwen 单裁判的 AQ、Accuracy、Completeness **1–5 分**，不混入训练期的 0–100 分。该最终三维评价不是 EM。

最新双组入口 `rag_core.scenic_ablation` 对两组分别执行全流程：

- `with_policy`：初始、迭代和群智提示都必须保留“上下文可能含噪音”与“对全部上下文 rerank，仅保留有用上下文、不固定保留数”两项要求。
- `without_policy`：不加入上述两项要求，逐阶段检查，避免被旧默认模板补回；仍要求基于上下文核验事实。

两组共享数据、向量、聚类设置及评价口径，但各自生成和优化提示。它不是同一个最终提示只增删两句话；rerank 是提示层面的要求，不是独立部署的重排序器。

## 最新景区 top5 双组运行

已有服务器部署目录为 `/pub/yanglw/rag_core_pipeline`。先确认当前 checkout 是本分支，并准备下述外部输入与运行环境，再运行：

```bash
cd /pub/yanglw/rag_core_pipeline
bash run_scenic_ablation.sh --workers 80
```

更换运行目录或参数：

```bash
bash run_scenic_ablation.sh \
  --source inputs/scenic_after_top10_chunks.jsonl.gz \
  --run-dir runs/my_scenic_top5 \
  --workers 80 --max-rounds 30 --min-gain 1 \
  --base-url http://127.0.0.1:8000/v1 \
  --llm-model /pub/yanglw/models/Qwen/Qwen3-8B \
  --model-path /pub/yanglw/models/bge-large-zh-v1.5
```

该双组入口固定取 `routes.fusion` 前 5 条；旧的可选 1–10 条预算入口是 `run_noise_aware.sh`，不能把它的旧上下文与新双组数据混为同批实验。

断开 SSH 后继续运行的监督器：

```bash
/pub/yanglw/miniconda3/bin/python -m rag_core.scenic_ablation_job start --workers 80
/pub/yanglw/miniconda3/bin/python -m rag_core.scenic_ablation_job status
```

默认输出目录 `runs/scenic_new_top5_ablation_20261017/`。如已有完成记录，监督器会直接返回状态，不重复启动。不同实验参数请使用新目录，配置指纹不匹配会拒绝复用。

## 外部输入与环境：不随此次代码提交上传

这次提交只包含源代码、模板、文档和离线测试，不上传问答输入、检索正文、模型权重、逐题实验结果、运行日志、账号密码或本地包。

运行前需要保留/自行准备：

| 路径或服务 | 用途 |
|---|---|
| `inputs/iteration_questions.json` | 原 chapter3 独立训练案例（当前 24 条），包含 golden 证据 |
| `inputs/eval_questions.json` | 原 chapter3 测试问题与参考答案，用于匹配与最终评价 |
| `inputs/scenic_after_top10_chunks.jsonl.gz` | 新景区三路检索缓存，实际仅使用 `routes.fusion` 前 5 条 |
| `/pub/yanglw/models/bge-large-zh-v1.5` | BGE 编码模型，可通过 `--model-path` 修改 |
| 本地 Qwen3-8B vLLM `/v1` 服务 | 提示生成、案例优化、群智合成、推理与最终裁判；脚本不负责启停 vLLM |
| `/pub/yanglw/tourist_three_experiments/run_three_experiments.py` | 原 chapter3 机械指标及 LLM 评价协议；保留原文件，记录 SHA256，避免改变评价口径 |

最新已完成实验从 402 个源问题中排除 23 个训练重合题和 1 个不足 5 条 fusion 文档的题，保留相同的 378 道测试题。数量由输入决定，不应在数据改变时宣称仍是同一批 378 题。

启动脚本默认使用服务器已有 `code_generation` Python 环境；最终评价由 `scenic_ablation.py` 调用已有 base Python。另一个服务器需要调整这些服务器专用路径。`RAG_PYTHON` 可指定启动脚本的 Python，但不会自动改写双组入口内的最终评价 Python 路径。

编码依赖沿用现有 torch/transformers；`requirements-runtime.txt` 记录仓库局部 `runtime_deps/` 所需 accelerate 版本。最终评价需要原协议使用的 jieba、nltk、rouge-chinese。不要在正在提供 vLLM 服务的环境中盲目升级依赖。

## 代码索引

| 文件 | 职责 |
|---|---|
| `rag_core/prompts/*.txt` | 五模块初始生成的 meta-prompt |
| `rag_core/prompt_policy.py` | 两条说明、五字段校验、分组约束和 JSON 格式修复 |
| `rag_core/adaptive_prompting.py` | 案例级自适应迭代、回退与恢复 |
| `rag_core/cluster_prompting.py` | QA 聚类、簇内群智提示与问题路由 |
| `rag_core/cached_context.py` | 缓存检索文本读取、预算选择、训练排重 |
| `rag_core/case_collective_experiment.py` | 原缓存上下文的完整实验 CLI |
| `rag_core/scenic_ablation.py` | 新景区共同数据、双组训练/推理/评价与汇总 |
| `rag_core/final_qwen_evaluation.py` | 复用 chapter3 协议的机械指标和单 Qwen 三维评分 |
| `rag_core/audit_scenic_ablation.py` | 双组数据、上下文、配置、提示政策与结果一致性审计 |
| `rag_core/*_job.py` | 后台运行、日志及有限状态查询 |

## 离线验证

以下测试使用模拟客户端，不请求模型，不启动实验：

```bash
python -m unittest discover -s tests -p 'test_noise_aware.py' -v
python -m unittest discover -s tests -p 'test_scenic_ablation.py' -v
python -m unittest discover -s tests -p 'test_final_qwen_evaluation.py' -v
python -m compileall -q rag_core
```

本次提交前以上 29 个针对新流程的测试通过。完整运行说明分别见 `USAGE_NOISE_AWARE.md`（早期部署/旧检索上下文入口）与 `SCENIC_ABLATION_README.md`（最新上下文双组实验）。
