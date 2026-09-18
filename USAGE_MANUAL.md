# RAG Core Pipeline 全流程使用说明书

本文档面向服务器部署和日常使用，说明从原始数据到答案生成的完整流程。项目默认不接触原仓库中的旧索引、旧数据库和旧实验结果；每次运行都使用独立的运行目录和 collection 名称。

## 1. 项目完成什么工作

项目只有一条主流程：

~~~text
原始 txt / md / json
  → 三阶段语义分片
  → 机械去噪（可关闭）
  → 第三阶段长度收口
  → v2 两层维度 Schema
  → 叶子维度多标签抽取
  → Qdrant 向量和数组标签入库
  → 查询解析与 Prompt 扩展
  → 语义检索 + 维度检索融合
  → 完整 chunk 传给生成模型
  → 输出答案和检索审计记录
~~~

主要目录如下：

- run.py：统一命令行入口。
- rag_core/：分片、去噪、嵌入、存储、检索、Prompt 和问答流程，并通过适配器调用维度生产实现。
- code_jyx/：从原项目完整迁移的维度生产实现，也是当前唯一实现来源；保留 v2 两层 Schema、叶子维度多标签抽取、倒排索引、v4 查询解析和层次化计分。
- core/：历史兼容代码，当前主流程不从这里导入维度生产实现。
- tests/：不依赖真实 API 的本地测试。
- requirements.txt：基础依赖和可选模型依赖。

## 2. 服务器首次部署

以下说明以服务器0为例。服务器上的 Qdrant 已使用非 Docker 方式运行，地址为：

~~~text
http://127.0.0.1:6333
~~~

服务器项目目录建议使用：

~~~text
/home/humq/rag_core_pipeline
~~~

### 2.1 上传或克隆项目

项目仓库是公开仓库，可以直接克隆：

~~~bash
cd /home/humq
git clone https://github.com/hh030210/rag_core_pipeline.git
cd /home/humq/rag_core_pipeline
~~~

以后更新项目：

~~~bash
cd /home/humq/rag_core_pipeline
git pull --ff-only origin main
~~~

如果目录已经存在，不要重复克隆；先执行：

~~~bash
cd /home/humq/rag_core_pipeline
git status
git pull --ff-only origin main
~~~

### 2.2 创建 Python 环境

服务器已有 Python 3.11 环境时，可直接使用：

~~~bash
/home/humq/envs/denoise_qa/bin/python --version
/home/humq/envs/denoise_qa/bin/python -m pip install -r requirements.txt
~~~

requirements.txt 已包含真实 BGE-M3 所需的 FlagEmbedding、Sentence Transformers 和兼容版本的 Transformers 依赖。安装后检查导入：

~~~bash
/home/humq/envs/denoise_qa/bin/python -c "import qdrant_client, numpy, yaml; print('DEPENDENCIES_OK')"
~~~

## 3. 配置服务地址、模型和密钥

不要把 API 密钥写入代码、README 或 Git。建议在服务器终端中临时设置环境变量：

~~~bash
export QDRANT_URL='http://127.0.0.1:6333'
export BGE_MODEL_PATH='/home/humq/models/bge-m3'
export LLM_OPENAI_COMPAT=1
export LLM_API_KEY='替换为实际密钥'
export LLM_BASE_URL='替换为兼容 OpenAI 接口的地址'
export LLM_MODEL='替换为实际模型名'
export LLM_API_INTERVAL=3
~~~

如果使用 SiliconFlow 等兼容接口，LLM_BASE_URL 应填写到 /v1，例如：

~~~bash
export LLM_BASE_URL='https://api.siliconflow.cn/v1'
export LLM_MODEL='Qwen/Qwen3-8B'
~~~

如果使用 DashScope 的兼容接口，填写该服务实际提供的兼容地址和模型名。不同接口的模型名、限流和余额状态可能不同，项目不会自行替换模型。

确认 Qdrant：

~~~bash
curl --max-time 5 http://127.0.0.1:6333/
~~~

返回 HTTP 200 或 Qdrant 的 JSON 响应，说明服务可用。

## 4. 全量入库流程

### 4.1 准备输入数据

--input 可以指向一个文件或目录。目录中的 txt、md 和 json 文件会被读取。建议把本次数据放在独立目录，例如：

~~~text
/home/humq/data/scenic/
~~~

### 4.2 执行分片、维度抽取和 Qdrant 入库

每次实验使用新的 collection 和新的 run-dir，例如：

~~~bash
cd /home/humq/rag_core_pipeline

/home/humq/envs/denoise_qa/bin/python -u run.py ingest \
  --input /home/humq/data/scenic \
  --run-dir /home/humq/rag_core_runs/scenic_v1 \
  --backend qdrant \
  --qdrant-url "$QDRANT_URL" \
  --collection rag_core_scenic_v1 \
  --model-path "$BGE_MODEL_PATH" \
  --llm-api-key "$LLM_API_KEY" \
  --llm-base-url "$LLM_BASE_URL" \
  --llm-model "$LLM_MODEL" \
  --llm-interval "$LLM_API_INTERVAL" \
  --denoise-method mechanical
~~~

入库阶段依次完成：

1. 读取原始文档并执行三阶段分片。
2. 使用机械去噪清理明显噪声。
3. 在第三阶段依据推荐长度进行语义合并和长度收口。
4. 生成或读取 v2 两层维度 Schema。
5. 只对叶子维度抽取 0 到多个原子标签，并保存 evidence。
6. 生成 chunk、标题和文档标题向量。
7. 创建新的 Qdrant collection 并批量写入。

--denoise-method 可选：

- mechanical：默认机械去噪。
- none：关闭去噪，用于对照实验。
- ppl：使用困惑度去噪，要求相关模型依赖和配置完整。

### 4.3 需要一次入库后立即测试时

可以使用 run，它等价于入库后再回答一条问题：

~~~bash
/home/humq/envs/denoise_qa/bin/python -u run.py run \
  --input /home/humq/data/scenic \
  --run-dir /home/humq/rag_core_runs/scenic_v1 \
  --backend qdrant \
  --qdrant-url "$QDRANT_URL" \
  --collection rag_core_scenic_v1 \
  --model-path "$BGE_MODEL_PATH" \
  --llm-api-key "$LLM_API_KEY" \
  --llm-base-url "$LLM_BASE_URL" \
  --llm-model "$LLM_MODEL" \
  --denoise-method mechanical \
  --query '景区的开放时间和门票是多少？'
~~~

## 5. 问答和检索

入库完成后，使用同一个 run-dir 和 collection：

~~~bash
cd /home/humq/rag_core_pipeline

/home/humq/envs/denoise_qa/bin/python -u run.py ask \
  --run-dir /home/humq/rag_core_runs/scenic_v1 \
  --backend qdrant \
  --qdrant-url "$QDRANT_URL" \
  --collection rag_core_scenic_v1 \
  --model-path "$BGE_MODEL_PATH" \
  --llm-api-key "$LLM_API_KEY" \
  --llm-base-url "$LLM_BASE_URL" \
  --llm-model "$LLM_MODEL" \
  --query '景区的开放时间和门票是多少？' \
  --top-k 5
~~~

默认行为是把检索到的每个 chunk 的完整文本传给生成模型，不再默认截取前 500 个字符。只有明确添加下面参数时才会限制上下文：

~~~bash
--context-chars 1000
~~~

不设置 --context-chars 或设置为 0，表示传入完整 chunk。

输出 JSON 中重点查看：

- answer：生成答案。
- retrieval.semantic_top：语义检索结果。
- retrieval.dimension_top：维度检索结果。
- retrieval.top_chunks：归一化融合后的最终 Top-K。
- 每条结果中的 matched_dimensions、matched_labels、dimension_paths 和 score。
- context_audit：实际传给生成模型的上下文长度。

## 6. 离线检索评测

`evaluate` 使用带 Golden 证据的 JSON/JSONL 测试集，分别评估语义、维度和融合三路检索，输出 First Golden Rank、MRR、Hit@K、Golden Recall@K、nDCG、融合救回/伤害和逐问题 badcase。

测试集推荐使用以下字段：

~~~json
[
  {
    "id": "q_0001",
    "question": "景区的开放时间是什么？",
    "gold_evidence_texts": ["景区每日八点开放。"],
    "reference_answer": "景区每日八点开放。",
    "question_type": "开放时间",
    "spot": "测试景区",
    "answerable": true
  }
]
~~~

也兼容现有 `query`、`answer`、`source`、`attraction` 字段。切片版本变化时，优先使用证据文本或字符区间映射到新的 chunk；不要只依赖旧 chunk ID。

在已有运行目录上执行：

~~~bash
cd /home/humq/rag_core_pipeline

/home/humq/envs/denoise_qa/bin/python -u run.py evaluate \
  --run-dir /home/humq/rag_core_runs/scenic_v1 \
  --dataset /home/humq/data/eval_gold.json \
  --output /home/humq/rag_core_runs/scenic_v1/evaluation \
  --top-k 10 \
  --eval-depth 20 \
  --ks 1,3,5,10,20
~~~

评测输出为 `results.jsonl`、`summary.json` 和 `summary.md`。评测只检查检索，不生成答案；答案质量应单独记录。

比较 baseline 与优化版本：

~~~bash
/home/humq/envs/denoise_qa/bin/python -u run.py compare \
  --baseline /home/humq/rag_core_runs/baseline/evaluation/results.jsonl \
  --candidate /home/humq/rag_core_runs/optimized/evaluation/results.jsonl \
  --output /home/humq/rag_core_runs/compare
~~~

`compare` 会输出逐问题改善/退化、三路 MRR/Hit/Recall/nDCG 差值和融合 badcase 变化。

## 7. Prompt 迭代优化

Prompt 迭代需要一个 JSON 样本文件，格式如下：

~~~json
[
  {
    "question": "景区什么时候开放？",
    "context": "景区全年开放，具体时段以公告为准。",
    "reference_answer": "景区全年开放，具体时段以公告为准。"
  }
]
~~~

执行案例级迭代：

~~~bash
/home/humq/envs/denoise_qa/bin/python -u run.py optimize-prompt \
  --run-dir /home/humq/rag_core_runs/scenic_v1 \
  --examples /home/humq/data/prompt_examples.json \
  --prompt-iterations 3 \
  --llm-api-key "$LLM_API_KEY" \
  --llm-base-url "$LLM_BASE_URL" \
  --llm-model "$LLM_MODEL" \
  --llm-interval "$LLM_API_INTERVAL"
~~~

每轮包括“生成答案 → LLM 评测 → 根据问题改写 Prompt”。结果写入：

~~~text
/home/humq/rag_core_runs/scenic_v1/optimized_prompt.json
~~~

后续执行 ask 时会自动读取优化后的 Prompt。没有执行案例级迭代时，项目使用默认的严格上下文问答 Prompt；在线查询扩展仍然保留原问题作为失败兜底。

### 7.1 独立输入输出模块

如果只想优化 Prompt，不希望启动分片、维度抽取或 Qdrant，可以直接运行独立模块：

~~~bash
cd /home/humq/rag_core_pipeline

/home/humq/envs/denoise_qa/bin/python -m rag_core.prompt_module \
  --input /home/humq/data/prompt_input.json \
  --output /home/humq/data/prompt_output.json \
  --iterations 3 \
  --llm-api-key "$LLM_API_KEY" \
  --llm-base-url "$LLM_BASE_URL" \
  --llm-model "$LLM_MODEL" \
  --llm-interval "$LLM_API_INTERVAL"
~~~

输入文件可以是问答样本数组：

~~~json
[
  {
    "question": "景区几点开放？",
    "context": "景区每日八点开放。",
    "reference_answer": "景区每日八点开放。"
  }
]
~~~

也可以使用带基础 Prompt 和轮数配置的对象：

~~~json
{
  "base_prompt": {
    "system_prompt": "你是严格的知识库问答助手。"
  },
  "config": {"iterations": 3},
  "examples": []
}
~~~

输出文件会保存最终 Prompt、每一轮的答案评分和反馈，并保存输入路径、输出路径、有效样本数和迭代轮数。输出文件可以直接放入某个运行目录，后续 ask 会自动读取其中的 system_prompt 等字段。无 API 或调试时添加 --mock，模块会输出默认 Prompt 和空的迭代历史。

统一入口也支持同样的功能：

~~~bash
/home/humq/envs/denoise_qa/bin/python run.py prompt-module \
  --input /home/humq/data/prompt_input.json \
  --output /home/humq/data/prompt_output.json \
  --prompt-iterations 3
~~~

### 7.2 QA 聚类与聚类级群智 Prompt

新项目已迁移原项目中的“Prompt 迭代后聚类 QA，再由新查询匹配聚类 Prompt”的流程。执行时，每条 QA 先使用统一的 Prompt 优化器迭代；随后使用“问题 + 参考答案”向量进行确定性 KMeans 聚类；最后将同一聚类内的多条迭代记录交给 LLM，合成为该聚类的 Prompt。

在已有 run-dir 上单独执行：

~~~bash
cd /home/humq/rag_core_pipeline

/home/humq/envs/denoise_qa/bin/python -u run.py cluster-prompts \
  --run-dir /home/humq/rag_core_runs/scenic_v1 \
  --examples /home/humq/data/prompt_examples.json \
  --cluster-count 6 \
  --cluster-iterations 3 \
  --llm-api-key "$LLM_API_KEY" \
  --llm-base-url "$LLM_BASE_URL" \
  --llm-model "$LLM_MODEL" \
  --llm-interval "$LLM_API_INTERVAL"
~~~

问答时显式启用聚类路由：

~~~bash
/home/humq/envs/denoise_qa/bin/python -u run.py ask \
  --run-dir /home/humq/rag_core_runs/scenic_v1 \
  --backend qdrant \
  --qdrant-url "$QDRANT_URL" \
  --collection rag_core_scenic_v1 \
  --model-path "$BGE_MODEL_PATH" \
  --cluster-prompts \
  --cluster-top-k 1 \
  --llm-api-key "$LLM_API_KEY" \
  --llm-base-url "$LLM_BASE_URL" \
  --llm-model "$LLM_MODEL" \
  --query '景区几点开门？'
~~~

`--cluster-top-k 1` 表示只使用最相近的聚类 Prompt；设为 2 或更大时，会分别生成多个聚类答案，再用全局 Prompt 融合。`cluster_prompting.json` 不存在、路由未启用或聚类流程失败时，问答自动回退到原有全局 Prompt。也可以在 `run` 或 `ingest` 命令中增加 `--cluster-examples`，让入库结束后自动生成聚类 Prompt。

## 8. 运行产物和检查方法

每个 run-dir 都是独立的，主要文件如下：

| 文件 | 内容 |
|---|---|
| chunks/chunks.json | 三阶段分片明细 |
| chunks/chunk_summary.json | chunk 数量、长度和去噪统计 |
| chunks.json | 统一 chunk 记录 |
| V_cand_v2.json | 候选层次化 Schema |
| V_core_v2.json | 实际使用的层次化 Schema |
| tags_output_v2.json | 叶子维度数组标签、证据和置信度 |
| inverted_index_v2.json | 叶子维度到标签到 chunk 的倒排索引 |
| dimension_metadata_v2.json | 维度元数据 |
| dimension_hierarchy_v2.json | 父子层次关系 |
| store_manifest.json | Qdrant 地址、collection 和 point 数量 |
| run_manifest.json | 本次运行总清单 |
| optimized_prompt.json | Prompt 迭代记录 |
| cluster_prompting.json | QA 聚类、聚类中心和聚类级 Prompt |
| cluster_prompting/case_iterations/ | 每条 QA 的 Prompt 迭代审计 |
| cluster_prompting/cluster_prompts/ | 每个聚类的群智 Prompt |
| last_answer.json | 最近一次答案、检索结果和上下文审计 |

快速检查入库结果：

~~~bash
cd /home/humq/rag_core_runs/scenic_v1
grep -E '"point_count"|"collection"|"backend"' store_manifest.json
grep -E '"schema_version"|"documents"' tags_output_v2.json | head
~~~

检查 Qdrant collection：

~~~bash
curl -sS "$QDRANT_URL/collections/rag_core_scenic_v1"
~~~

Qdrant payload 中的 dim_<leaf_id> 必须是 JSON 数组，例如：

~~~json
"dim_content.location": ["山门", "寺院"]
~~~

不能是：

~~~text
"山门; 寺院"
~~~

## 9. 后台运行

适合较长的入库或问答任务：

~~~bash
mkdir -p /home/humq/rag_core_logs
nohup /home/humq/envs/denoise_qa/bin/python -u run.py ingest \
  --input /home/humq/data/scenic \
  --run-dir /home/humq/rag_core_runs/scenic_v1 \
  --backend qdrant \
  --qdrant-url "$QDRANT_URL" \
  --collection rag_core_scenic_v1 \
  --model-path "$BGE_MODEL_PATH" \
  --llm-api-key "$LLM_API_KEY" \
  --llm-base-url "$LLM_BASE_URL" \
  --llm-model "$LLM_MODEL" \
  --denoise-method mechanical \
  > /home/humq/rag_core_logs/scenic_v1_ingest.log 2>&1 &
echo $!
~~~

查看进度：

~~~bash
tail -f /home/humq/rag_core_logs/scenic_v1_ingest.log
~~~

检查是否结束：

~~~bash
test -f /home/humq/rag_core_runs/scenic_v1/run_manifest.json && echo INGEST_DONE || echo INGEST_RUNNING_OR_FAILED
~~~

如果需要停止自己启动的任务，先通过进程列表确认 PID，再结束对应 PID；不要停止 Qdrant 主进程。

## 10. 无模型、无 API 的本地验收

在首次部署或排查代码问题时，使用 --mock。此模式不访问 Qdrant、BGE 和 LLM：

~~~bash
cd /home/humq/rag_core_pipeline
/home/humq/envs/denoise_qa/bin/python -u run.py run \
  --input tests/fixtures \
  --run-dir /home/humq/rag_core_runs/mock_smoke \
  --mock \
  --query '景区的开放时间和门票是多少？'
~~~

运行测试：

~~~bash
/home/humq/envs/denoise_qa/bin/python -m unittest discover -s tests -v
~~~

测试至少验证：确定性嵌入、多标签数组、Prompt 无 LLM 回退和两层 Schema 合法性。

## 11. 常见问题

### Qdrant 无法连接

先执行：

~~~bash
curl --max-time 5 http://127.0.0.1:6333/
pgrep -af qdrant
~~~

如果 Qdrant 没有运行，应先恢复 Qdrant 服务，再启动入库。不要用同名 collection 覆盖旧实验。

### 找不到模型

确认：

~~~bash
test -d "$BGE_MODEL_PATH" && echo MODEL_PATH_OK || echo MODEL_PATH_ERROR
~~~

真实入库需要本地 BGE 模型；模型下载完成后只把模型路径写入 BGE_MODEL_PATH，不要把模型提交到 Git。

### LLM 请求频繁受限

增大：

~~~bash
export LLM_API_INTERVAL=5
~~~

并降低并发。维度抽取和 Prompt 迭代都依赖 LLM，API 限流时应先看日志中的失败原因，不要重复启动同一批入库任务。

### 维度标签为空

检查 V_core_v2.json 是否包含叶子维度，再检查 tags_output_v2.json 中的 documents。如果只有一级维度，说明 Schema 没有通过 v2 校验；如果标签存在但检索为空，检查查询中的维度名、别名和 dim_<leaf_id> payload。

### 答案不完整

确认 ask 没有使用 --context-chars，并查看 last_answer.json 的 context_audit。默认流程会把完整 chunk 传入生成模型；如果上游 chunk 本身过短，应先检查 chunks/chunk_summary.json 和第三阶段长度收口结果。

## 12. 数据和安全约定

- 每次实验使用新的 run-dir 和 collection 名称。
- 不删除旧 collection，不覆盖生产索引。
- 不把 API 密钥写入项目文件、日志或 Git。
- 不把模型文件和大规模运行结果上传 GitHub。
- Gold Source 只用于评测对照；问答模型实际使用的是融合检索返回的 Top-K chunk。
- 生产运行前先用 --mock 完成代码验收，再使用小规模输入验证 Qdrant 和 LLM 配置，最后再进行全量入库。
