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
答案生成
```

原项目中的实验报告、badcase 格式化、Milvus 流程、Web 前端、MySQL 兼容写入和历史版本脚本没有复制进来。维度信息在新项目中以 `V_core_v2.json`、`tags_output_v2.json`、`inverted_index_v2.json` 和 Qdrant payload 为准。

完整的服务器部署、入库、检索、问答、Prompt 迭代和故障排查说明见 [USAGE_MANUAL.md](USAGE_MANUAL.md)。

## 1. 安装

```bash
cd rag_core_pipeline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install FlagEmbedding torch
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
  --input ../data_input/test_data \
  --run-dir ./runs/smoke \
  --mock \
  --query '少林寺的开放时间和门票是多少？'
```

也可以分开执行：

```bash
python run.py ingest --input ../data_input/test_data --run-dir ./runs/demo --mock
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

## 5. 运行产物

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
- `last_answer.json`：检索、上下文长度、Prompt 和答案审计记录
