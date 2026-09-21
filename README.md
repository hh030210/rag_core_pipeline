# RAG Core Pipeline

这是从原仓库抽出的最小可运行项目，只保留一条完整闭环：

```text
原始 txt/md/json
    ↓
三阶段语义分片（第三阶段执行推荐长度收口）
    ↓
v2 层次化维度发现 + 叶子多标签抽取
    ↓
Qdrant 入库（chunk 向量 + 标题向量 + 数组标签 payload）
    ↓
查询扩展 + 维度检索 + 语义检索 + 归一化融合
    ↓
Prompt 优化
    ↓
QA 聚类 → 聚类级群智 Prompt → 新查询匹配聚类（可选）
    ↓
答案生成
```

原项目中的实验报告、badcase 格式化、Milvus 流程、Web 前端、MySQL 兼容写入和历史版本脚本没有复制进来。维度信息在新项目中以 `V_core_v2.json`、`tags_output_v2.json`、`inverted_index_v2.json` 和 Qdrant payload 为准。真实景区原始文档位于 `data_input/test_data/`，问答数据位于 `data/`；模型和运行产物不放入项目数据目录。

完整的服务器部署、入库、检索、问答、Prompt 迭代和故障排查说明见 [USAGE_MANUAL.md](USAGE_MANUAL.md)。

## 0. 代码来源和叶子维度实现

`code_jyx/` 是从原项目完整迁移的维度相关生产代码，当前由它负责：

- v2 两层 Schema：一级维度只分组，二级叶子维度才可索引；
- 叶子维度的多标签抽取，标签保存为数组，并保留 `raw_label`、`evidence` 和 `confidence`；
- v2 倒排索引、父子层次、`QueryParserV4` 查询解析和层次化维度计分。

`rag_core/` 只负责把三阶段分片、嵌入、Qdrant、问答和 Prompt 流程串起来，并通过适配器调用 `code_jyx`。这样保留了原来的维度逻辑和叶子维度方案，同时避免继续维护两份生产实现。`core/` 目录中的旧实现仅作为兼容代码保留，当前主流程不使用它。

## 1. 安装

```bash
cd rag_core_pipeline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

服务器上设置真实模型和接口：

```bash
export BGE_MODEL_PATH=/path/to/bge-m3
export LLM_OPENAI_COMPAT=1
export LLM_API_KEY='你的密钥'
export LLM_BASE_URL='https://api.siliconflow.cn/v1'
export LLM_MODEL='Qwen/Qwen3-8B'
```

不要把密钥写入代码或提交到 Git。

## 2. 本地无网络验收

`--mock` 会使用确定性哈希向量、规则多标签抽取和本地 JSON 向量库，不需要 Qdrant、BGE 或 LLM：

```bash
python run.py run \
  --input ./data_input/test_data \
  --run-dir ./runs/smoke \
  --mock \
  --query '少林寺的开放时间和门票是多少？'
```

也可以分开执行：

```bash
python run.py ingest --input ./data_input/test_data --run-dir ./runs/demo --mock
python run.py ask --run-dir ./runs/demo --mock --query '西湖什么时候开放？'
```

## 3. 使用服务器 Qdrant

先确认服务器 Qdrant 地址可访问，然后使用一个全新的 collection 名称：

```bash
python run.py ingest \
  --input /path/to/data \
  --run-dir ./runs/scenic_v1 \
  --backend qdrant \
  --qdrant-url http://127.0.0.1:6333 \
  --collection rag_core_scenic_v1 \
  --model-path /path/to/bge-m3 \
  --llm-api-key "$LLM_API_KEY" \
  --llm-base-url "$LLM_BASE_URL" \
  --llm-model "$LLM_MODEL" \
  --denoise-method mechanical
```

入库后问答：

```bash
python run.py ask \
  --run-dir ./runs/scenic_v1 \
  --backend qdrant \
  --qdrant-url http://127.0.0.1:6333 \
  --collection rag_core_scenic_v1 \
  --model-path /path/to/bge-m3 \
  --llm-api-key "$LLM_API_KEY" \
  --llm-base-url "$LLM_BASE_URL" \
  --llm-model "$LLM_MODEL" \
  --query '颐和园几点开门？'
```

默认每个检索 chunk 将完整文本传给生成模型；只有显式设置 `--context-chars` 才会截断。

## 4. Prompt 迭代

案例样本格式可以是数组，也可以是 `{"examples": [...]}`：

```json
[
  {
    "question": "西湖的开放时间是什么？",
    "context": "西湖全年开放，具体开放时段以公告为准。",
    "reference_answer": "西湖全年开放，具体时段以公告为准。"
  }
]
```

在入库后执行案例级“生成答案 → LLM 评测 → Prompt 改写”的迭代：

```bash
python run.py optimize-prompt \
  --run-dir ./runs/scenic_v1 \
  --examples ./examples.json \
  --prompt-iterations 3 \
  --llm-api-key "$LLM_API_KEY" \
  --llm-base-url "$LLM_BASE_URL" \
  --llm-model "$LLM_MODEL"
```

生成的 `optimized_prompt.json` 会被后续 `ask` 自动读取。没有样本时会使用严格的上下文问答 Prompt；在线问答中的查询扩展仍会保留原问题作为失败兜底。

Prompt 迭代也可以完全脱离入库和问答流程，作为独立的 JSON 输入输出模块运行：

```bash
python -m rag_core.prompt_module \
  --input ./prompt_input.json \
  --output ./prompt_output.json \
  --iterations 3 \
  --llm-api-key "$LLM_API_KEY" \
  --llm-base-url "$LLM_BASE_URL" \
  --llm-model "$LLM_MODEL"
```

输入可以是问答样本数组，也可以是包含 `examples`、可选 `base_prompt` 和 `config.iterations` 的 JSON 对象；输出包含最终 Prompt、每轮评分反馈以及 `io` 输入输出信息。统一入口也提供等价命令：`python run.py prompt-module ...`。

## 5. QA 聚类与聚类级群智 Prompt

新项目已迁移原流程中的聚类 Prompt 逻辑。它先对每条 QA 样本执行案例级 Prompt 迭代，再使用“问题 + 参考答案”向量做确定性 KMeans 聚类，最后汇总每个聚类中的多条迭代记录，由 LLM 生成一个聚类级 Prompt。新查询问答时可按余弦相似度选择 Top-K 聚类，并使用对应 Prompt；选择多个聚类时会再融合候选答案。

单独执行聚类流程：

```bash
python run.py cluster-prompts \
  --run-dir ./runs/scenic_v1 \
  --examples ./examples.json \
  --cluster-count 6 \
  --cluster-iterations 3 \
  --llm-api-key "$LLM_API_KEY" \
  --llm-base-url "$LLM_BASE_URL" \
  --llm-model "$LLM_MODEL"
```

产物包括 `cluster_prompting.json`、`cluster_prompting/case_iterations/` 和 `cluster_prompting/cluster_prompts/`。已有入库运行目录问答时启用路由：

```bash
python run.py ask \
  --run-dir ./runs/scenic_v1 \
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
```

也可以在 `run` 或 `ingest` 时增加 `--cluster-examples`，在入库后自动生成聚类 Prompt。默认 `--cluster-top-k 1`，设置为大于 1 时会生成多个聚类答案并进行答案融合。未启用或找不到 `cluster_prompting.json` 时，自动使用原有全局 Prompt，不影响原流程。

## 6. 离线检索评测

`evaluate` 使用带 Golden 证据的 JSON/JSONL 测试集，分别评估语义、维度和融合三路结果，输出 First Golden Rank、MRR、Hit@K、Golden Recall@K、nDCG、融合救回/伤害和逐问题 badcase。

测试集可以使用规范字段：

```json
[
  {
    "id": "q_0001",
    "question": "南孔庙的开放时间是什么？",
    "gold_evidence_texts": ["南孔庙开放时间为……"],
    "reference_answer": "……",
    "question_type": "开放时间",
    "spot": "南孔庙",
    "answerable": true
  }
]
```

也兼容现有 `query`、`answer`、`source`、`attraction` 字段。Golden 优先使用当前索引的 `gold_chunk_ids`；切片版本变化时，应使用证据文本或字符区间重新映射，避免把 chunk ID 变化误判为检索失败。

在已有运行目录上评测：

```bash
python run.py evaluate \
  --run-dir ./runs/scenic_v1 \
  --dataset ./eval/gold.json \
  --output ./runs/scenic_v1/evaluation \
  --top-k 10 \
  --eval-depth 20 \
  --ks 1,3,5,10,20
```

输出文件：

- `results.jsonl`：逐问题三路检索结果、Golden 映射和指标
- `summary.json`：全量及按景区/问题类型分组汇总
- `summary.md`：便于人工查看的摘要

对比 baseline 与优化版本：

```bash
python run.py compare \
  --baseline ./runs/baseline/evaluation/results.jsonl \
  --candidate ./runs/optimized/evaluation/results.jsonl \
  --output ./runs/compare
```

对比结果包含逐问题改善/退化、MRR/Hit@K/Recall/nDCG 差值以及新增/修复 badcase。评测默认不生成答案；答案质量应作为独立实验记录。

## 6.1 问答答案质量评价

批量问答完成后，可使用 `rag_core.qa_evaluation` 对每条答案进行 LLM 评价。评价维度包括正确性、完整性、相关性、证据支撑和无依据陈述，并可关联同一顺序的检索评测结果：

```bash
python -m rag_core.qa_evaluation \
  --input ./runs/scenic_v1/qa_results.jsonl \
  --output-dir ./runs/scenic_v1/qa_evaluation \
  --retrieval-results ./runs/scenic_v1/evaluation/results.jsonl \
  --base-url "$LLM_BASE_URL" \
  --model "$LLM_MODEL" \
  --api-key "$LLM_API_KEY" \
  --concurrency 4
```

输出 `qa_answer_evaluation.jsonl`、`qa_answer_evaluation_summary.json` 和 `qa_answer_evaluation_summary.md`。该评价与检索指标分开统计，便于区分“检索证据不足”和“答案生成质量不足”。

同时会为每条问答计算不调用模型的机械指标：归一化 Exact Match、字符级 Precision/Recall/F1、ROUGE-L F1、数字事实 Precision/Recall/F1、遗漏/多答数字数量，以及答案与融合 Top5 证据的字符覆盖率。字符重叠和证据覆盖率是 lexical proxy，适合做版本间稳定对比，不能单独替代事实正确性评价。数字指标只在参考答案包含阿拉伯数字时纳入总体均值；每条明细指标保存在 `qa_answer_evaluation.jsonl` 的 `mechanical_metrics` 字段中。

## 7. 运行产物

每次运行都写入独立的 `run-dir`，不会删除或覆盖其他运行：

- `chunks/chunks.json`：三阶段分片原始产物
- `chunks/chunk_summary.json`：长度和去噪审计
- `chunks.json`：统一 chunk 记录
- `V_cand_v2.json` / `V_core_v2.json`：层次化 Schema
- `tags_output_v2.json`：每个叶子维度的数组标签、证据和置信度
- `inverted_index_v2.json`：`leaf_dimension -> label -> chunk_ids`
- `dimension_metadata_v2.json` / `dimension_hierarchy_v2.json`
- `store_manifest.json`：Qdrant 或本地后端信息
- `optimized_prompt.json`：Prompt 优化结果和每轮评测记录
- `cluster_prompting.json`：QA 聚类、聚类中心和聚类级 Prompt 路由产物
- `cluster_prompting/case_iterations/`：逐条 QA 的 Prompt 迭代审计
- `cluster_prompting/cluster_prompts/`：每个聚类的群智 Prompt
- `last_answer.json`：检索、上下文长度、Prompt 和答案审计记录
