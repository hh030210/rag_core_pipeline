# 案例优化 + 群智提示 + 检索证据筛选实验

部署目录：`/pub/yanglw/rag_core_pipeline`。
上游快照：`af2dabb46ecd338c2db6ee3abd5e7e1a8c70ec6a`。
本次只部署、准备数据和离线验收，不启动真实模型实验，也不修改已有 vLLM 服务。

## 数据和模型

- `inputs/iteration_questions.json`：chapter3 原案例优化训练题，24 题。`source` 是 golden 证据。
- `inputs/eval_questions.json`：chapter3 原测试题，412 题。
- 检索文件：`result/real_merged7_deepseek_v4pro_rerank_20260920_run1/evaluation_noexpansion/results_top15_full_text.jsonl.gz`。
- 默认从原测试题中匹配具有完整 top10 融合检索正文的 397 题。
- 训练与全部原测试题按 NFKC、大小写、去标点空白后的问题文本检查重合，删除重合训练题并保存清单；当前重合数为 0。
- `--min-retrieval-depth 15` 可选择之前有完整 top15 的 347 题，但仍只使用其前 top10。
- `--context-budget 5` 严格取同一批题 top10 的前 5 条；`10` 使用全部十条；可选整数 1–10。
- 推理不混入、不插入、不额外检索 golden；候选文本及顺序直接来自仓库融合检索结果，不把 `is_gold` 等 oracle 字段提供给模型。
- BGE 使用 chapter3 原实验的 `bge-large-zh-v1.5`，部署在 `/pub/yanglw/models/bge-large-zh-v1.5`；CLS 池化、512 token 上限、归一化、CPU 编码。QA K-means 聚类方法不变。也可以用 `--model-path` 指定其他 BGE 模型，但训练和推理必须保持一致，不混用旧簇中心。
- LLM 默认使用服务器现有的 Qwen3-8B vLLM：`http://127.0.0.1:8000/v1`，模型名 `/pub/yanglw/models/Qwen/Qwen3-8B`。可以用 CLI 或环境变量改模型和端点。
- 不安装/升级现有 vLLM 环境的包。BGE 在无 FlagEmbedding/SentenceTransformer 时复用已安装的 torch + transformers。

## 提示生成与停止规则

固定的是 `rag_core/prompts/case_initial_meta.txt` 这个 meta-prompt 以及五字段结构，不是预先写好的五模块初始提示。
每个训练案例单独调用 LLM 生成新的初始提示，输入问题和 golden 证据，不输入参考答案。
独立 Prompt 优化接口仅接受一条真实训练案例；多案例数据集必须使用 `cluster-prompts` 或本实验入口，以免误用一个共享初始提示。
参考答案只用于后续评价。模板和改写要求禁止把案例具体答案硬编码到通用提示。

`context_strategy` 必须逐字包含 `rag_core/prompt_policy.py` 的两条不可变约束：

1. 给定上下文可能包含噪音、无关信息或相互矛盾的信息。
2. 无论多少条上下文，都对全部上下文按当前问题的相关性和证据价值 rerank，仅使用被判断有用的上下文，不固定保留数量；可以全部保留或全部舍弃，证据不足时明确说明。

初始生成、每轮改写、群智归纳都执行严格字段与约束校验；缺失时要求重新生成一次，仍不合格则初始/群智生成报错，改写阶段保留上一已接受提示并停止。不会用静态默认模板冒充真实优化结果。
这是提示层面的模型证据使用策略，不是增加一个独立 reranker，也不声称能保证实际去噪效果。推理结果保存实际提供的上下文；不将模型未输出的内部排序伪造成外部排序记录。

每个案例先评价初始提示（iteration=0），再自适应改写、生成答案、评价：

- 三个维度：AQ (`answer_quality`)、Accuracy、Completeness，均为 0–100 分。
- 总分为三个维度的算术平均值，由程序计算。
- 任一维度下降，或总分下降，或相对上一已接受轮的总分提升小于 **1 个绝对分值点**，停止并丢弃该候选，保留上一已接受提示。
- 恰好提升 1 分且所有维度均不下降，可以接受。
- 最多 30 次改写是保护上限，不是固定跑满 30 轮（另有初始评价一次）。
- API 失败、空输出、评分非法等作为错误报告，不伪造 0 分、不当作正常收敛。
- 案例优化使用完整 golden，不再静默截成 6000 字符；LLM 输入超限会显式报错。
- 已完成的案例在数据、meta-prompt、模型及参数指纹匹配时复用；中断中的案例重新开始，其他已完成案例不重复计费。

聚类仍按“问题 + 参考答案”的向量执行确定性 K-means，默认 6 类，可用 `--clusters` 调整；不是聚类提示文本，也不自动选择最优 K。
群智提示汇总各簇所有训练案例的最终提示和迭代评分反馈，同样强制保留两条约束。
推理仅使用新问题向量匹配一个簇，不使用测试参考答案路由或生成答案。

## 运行命令

```bash
cd /pub/yanglw/rag_core_pipeline

# 完整真实流程：准备数据 -> golden 案例优化 -> QA 聚类与群智 -> top10 推理
bash run_noise_aware.sh all --context-budget 10

# 已训练后，仅以相同 top10 的前5条再推理，不重新优化案例
bash run_noise_aware.sh infer --context-budget 5

# 参数示例
bash run_noise_aware.sh all --run-dir runs/custom --context-budget 10 --workers 80 --max-rounds 30 --min-gain 1 --clusters 6

# 与以前 top15 实验严格同批的347题，需要独立运行目录
bash run_noise_aware.sh all --run-dir runs/cohort347 --min-retrieval-depth 15 --context-budget 10
```

默认 workers=80，独立训练案例可并行优化，每个案例内部仍逐轮进行；推理并发同样为 80。不修改 vLLM 的 `--max-num-seqs`，超出服务端同时处理能力的请求会排队。
脚本不会启动或重启 vLLM；需要现有服务可用。

## 产物

默认运行目录 `runs/noise_aware_v1/`：

- `prepared/train_golden.json`：独立训练题与 golden。
- `prepared/inference_top10.json`：固定 top10 测试题。
- `prepared/data_manifest.json`：数据来源 SHA256、匹配数和排除清单。
- `cluster_prompting/case_iterations/<case_id>.json`：meta-prompt、初始提示、每轮提示与候选答案、三维评分、接受/拒绝标记、最终轮和停止原因。
- `cluster_prompting.json` 与 `cluster_prompting/cluster_prompts/cluster_*.json`：QA 簇中心、归属及群智提示。
- `answers_top10.jsonl` / `answers_top5.jsonl`：可恢复推理结果，包含完整实际输入上下文、使用的提示和路由信息；参考答案仅作为后续评价标签保存在结果中，不进入生成输入。
- `inference_topK_errors.json`：失败请求清单；再次执行 infer 只补缺失题。

本次修改的三维 0–100 分用于**训练期的自适应停止**，不应直接混入以前 1–5 分的最终问答评价表。
最终机器指标与交叉模型评价可继续使用既有 chapter3 评价流程；本次没有更改它的评价口径，也不会自动调用两个裁判。

## 离线验收

```bash
/pub/yanglw/miniconda3/envs/code_generation/bin/python -m unittest discover -s tests -p 'test_noise_aware.py' -v
/pub/yanglw/miniconda3/envs/code_generation/bin/python -m compileall -q rag_core
bash -n run_noise_aware.sh
```

`--mock` 只能用于名称带 mock 的独立运行目录，输出明确标记 MOCK_ONLY，真实推理拒绝读取 mock/fallback 群智提示。离线验收不代表实际模型效果已经得到验证。

部署保留上游 Git 快照及全部常规仓库文件。历史上由 Git LFS 管理的其他实验大文件保留 LFS 指针，未额外下载；本实验所需完整 top15 检索文件、训练/测试输入、BGE 模型均已实际部署，不依赖这些历史 LFS 文件。

## Top5 全流程和 Qwen 最终评价（2026-10-17）

`bash run_top5_qwen_full.sh` 执行案例优化、群智生成、top5 推理，再执行最终机器指标与 Qwen 单裁判三维 1–5 分评价。
默认运行目录：`runs/noise_aware_top5_qwen_20261017`；默认 `WORKERS=80`。
训练/推理使用现有 `code_generation` 环境，最终评价使用已有 jieba/nltk/rouge-chinese 的 base 环境，不安装/升级服务端包。
最终评价直接复用 `/pub/yanglw/tourist_three_experiments/run_three_experiments.py` 的机械指标函数和三维评价提示，记录其 SHA256，不改原口径，不调用 DeepSeek。
实际首次加载发现 Transformers 缺少 accelerate，已将 `accelerate==1.6.0` 离线安装到本仓库 `runtime_deps/`，由启动脚本加入 PYTHONPATH，不修改 vLLM 环境。
最终结果：`evaluation_qwen/summary.json`、`summary.md`、`question_scores.jsonl`、`question_scores.csv`。
训练期间的 0–100 分用于停止判定；最终问答评价仍是原来的三个 1–5 分，二者不混用。

断开 SSH 后继续运行：

```bash
cd /pub/yanglw/rag_core_pipeline
/pub/yanglw/miniconda3/bin/python -m rag_core.top5_job start
/pub/yanglw/miniconda3/bin/python -m rag_core.top5_job status
```

监督器记录 `job_status.json` 和 `job.log`。完成后只有全部题目均得到最终评分才标记 complete；失败不当作完整实验。
