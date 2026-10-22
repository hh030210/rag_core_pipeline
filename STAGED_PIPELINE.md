# 分阶段运行 RAG 流程

`staged_run.py` 将文件处理、检索和问答拆成独立命令。每一步都写入指定的 `--run-dir`，阶段状态保存在 `pipeline_state.json`；已经完成的产物可供后续阶段直接使用。原有 `run.py run/ingest/ask/evaluate/batch-ask` 命令仍然可用。

## 阶段与中间产物

| 阶段 | 命令 | 主要产物 |
|---|---|---|
| 分片 | `chunk` | `chunks.json`、`chunks/`、分片统计、推荐参数 |
| 数据集维度 schema | `schema` | `V_cand_v2.json`、`V_core_v2.json` |
| 维度标签 | `tag` | `tags_output_v2.json`、`tags_checkpoint.jsonl`、倒排索引、维度诊断 |
| 向量入库 | `index` | `store_manifest.json`、`run_manifest.json`、本地向量或 Qdrant collection |
| 单条检索 | `retrieve` | 三路语义/维度/融合检索 JSON |
| Golden 检索评测 | `evaluate` | 现有评测详情和 summary |
| 问答 | `qa` | 单条 JSON 或可续跑的批量 JSONL |

维度 schema 阶段会基于当前 run 中的 chunks 新生成 schema，不会读取其他景区/数据集的维度。真实模式要求配置 LLM；只有显式 `--mock` 时才使用测试 schema、mock 标签和本地哈希向量。

## 服务器环境

在项目目录、已激活项目 Python 环境后设置所需变量。真实入库还需 `QDRANT_URL`、`BGE_MODEL_PATH` 和专用 collection；schema/标签/问答需要可用的 LLM API 配置。不要把 API key 写入命令历史或提交到仓库，建议用安全的环境变量注入方式。

```bash
export QDRANT_URL='http://127.0.0.1:6333'
export BGE_MODEL_PATH='/path/to/bge-m3'
export LLM_OPENAI_COMPAT=1
export LLM_API_KEY='由部署环境安全注入'
export LLM_BASE_URL='https://your-openai-compatible-endpoint/v1'
export LLM_MODEL='your-model'
export LLM_API_INTERVAL=3
```

## 从零开始

每个数据集使用独立的 run 目录和新的 Qdrant collection，避免与已有实验互相覆盖：

```bash
RUN=/home/your-user/rag_runs/dataset_20261022

python -u staged_run.py chunk \
  --input /path/to/documents \
  --run-dir "$RUN" \
  --denoise-method mechanical

python -u staged_run.py schema --run-dir "$RUN"
python -u staged_run.py tag --run-dir "$RUN"
python -u staged_run.py index --run-dir "$RUN" \
  --backend qdrant --collection dataset_20261022_chunks

python -u staged_run.py retrieve --run-dir "$RUN" \
  --query '测试问题' --top-k 5

python -u staged_run.py evaluate --run-dir "$RUN" \
  --dataset /path/to/golden.json --top-k 10 --eval-depth 20

python -u staged_run.py qa --run-dir "$RUN" \
  --dataset /path/to/questions.json --output "$RUN/qa_results.jsonl" \
  --top-k 5 --concurrency 1
```

也可以单独传入 `--question '问题'` 执行一条问答，或将 `--dataset` 指向 JSON/JSONL 批量问答集。批量问答默认按已完成行续跑；需要强制重做时使用 `--no-resume`。

## 从中间结果继续

后续阶段直接对同一个 `--run-dir` 执行。例如已有 `chunks.json` 时可从 schema 开始；已有 schema 时可从 `tag` 开始。维度标签阶段默认从 `tags_checkpoint.jsonl` 续跑。较早版本运行目录若已有兼容 `chunks.json`，schema 阶段也可直接读取；执行 `chunk` 时会登记/采用已有分片，不会先覆盖它。

```bash
python staged_run.py status --run-dir "$RUN"
python staged_run.py inspect --run-dir "$RUN" --stage all

# 标签请求中断后，在同一个 run-dir 继续：
python -u staged_run.py tag --run-dir "$RUN"
```

若要对改变后的输入重新分片，请新建 `--run-dir`，保留旧运行的中间结果和日志。状态命令会区分 `not_started`、`running`、`completed`、`failed` 以及“找到旧产物但尚未登记”的阶段。

## 离线逐环节验收

```bash
RUN=/tmp/rag_staged_smoke
python staged_run.py chunk --input /path/to/small_fixture.md --run-dir "$RUN" --mock
python staged_run.py schema --run-dir "$RUN" --mock
python staged_run.py tag --run-dir "$RUN" --mock
python staged_run.py index --run-dir "$RUN" --mock
python staged_run.py retrieve --run-dir "$RUN" --mock --no-query-expansion --query '样例问题'
python staged_run.py qa --run-dir "$RUN" --mock --question '样例问题'
python staged_run.py status --run-dir "$RUN"
```

`--mock` 只验证流程、格式、状态记录和中间产物，不代表线上 LLM 维度归纳/标签质量、BGE 编码或 Qdrant 连接已经通过；这些仍需在目标服务器配置真实服务后做小样本验收。
