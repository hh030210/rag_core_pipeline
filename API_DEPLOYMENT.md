# RAG Core Pipeline API 部署与调用说明

本服务提供三个完整能力：文件处理、检索、问答。文件处理复用项目流水线，依次完成分片、基于本批语料归纳维度并提取标签、向量入库；检索返回语义、维度和融合三路结果；问答在检索后调用大模型生成答案。

## 认证与大模型密钥

所有 `/api/v1/*` 接口都需要 API 服务访问令牌：

```http
Authorization: Bearer <RAG_API_TOKEN>
```

此令牌只用于访问本 RAG API，和大模型服务商的密钥不是同一个东西。

每次文件处理、检索或问答请求，还必须显式提供调用方自己的大模型配置：

- `llm_api_key`：调用方的大模型 API Key。
- `llm_base_url`：服务商提供的 OpenAI 兼容 HTTPS 接口地址，例如 `https://api.example.com/v1`。
- `llm_model`：该服务商支持的模型名称。

API 不再使用服务器上的 SDU 登录态、Cookie 或 SDU 代理。大模型密钥只在请求处理期间使用：文件处理的密钥仅保存在 API 进程内存中，不写入任务状态文件或返回结果；检索/问答的密钥也不会写入 API 响应。请勿把密钥放进 URL、代码仓库或日志。生产调用应通过 HTTPS；本机访问服务器时可使用 SSH 隧道。

示例中用环境变量传递密钥，避免把实际密钥写进命令历史：

```bash
export RAG_API_TOKEN='你的RAG_API访问令牌'
export LLM_API_KEY='你的大模型服务商API Key'
export LLM_BASE_URL='https://api.example.com/v1'
export LLM_MODEL='服务商提供的模型名'
```

## 1. 文件处理

### 提交任务

`POST /api/v1/file-processing/jobs` 使用 `multipart/form-data`：

- `files`：一个或多个 `.txt`、`.md`、`.json` 文件。
- `collection`：可选的 Qdrant collection 名；不填则自动生成。
- `denoise_method`：`mechanical`（默认）、`none` 或 `ppl`。
- `llm_api_key`、`llm_base_url`、`llm_model`：本次任务使用的大模型配置，必填。

```bash
curl -X POST 'http://127.0.0.1:8010/api/v1/file-processing/jobs' \
  -H "Authorization: Bearer ${RAG_API_TOKEN}" \
  -F 'files=@./docs/example.md' \
  -F "llm_api_key=${LLM_API_KEY}" \
  -F "llm_base_url=${LLM_BASE_URL}" \
  -F "llm_model=${LLM_MODEL}" \
  -F 'denoise_method=mechanical'
```

成功后返回 `202 Accepted` 和 `job_id`、`run_id`。处理异步执行，按返回的 `status_url` 查询进度；到达 `completed` 后再访问 `summary_url` 获取 collection、分片统计及维度信息。新上传的语料会重新归纳自己的维度，不复用其它数据集的维度体系。

任务完成后，summary 还会返回三个可观查链接；它们分别用于查看真实分片、维度标签和入库后的向量库状态：

```http
GET /api/v1/file-processing/jobs/<job_id>/chunks?offset=0&limit=20&include_text=false
GET /api/v1/file-processing/jobs/<job_id>/dimensions?offset=0&limit=20&include_evidence=false
GET /api/v1/file-processing/jobs/<job_id>/ingestion?sample_limit=5&include_text=false
```

- `chunks` 返回 chunk ID、来源文档、长度、分片统计；`include_text=true` 时额外返回 chunk 正文。
- `dimensions` 返回本数据集新归纳的维度结构、各维度覆盖 chunk 数/标签数、诊断统计，以及按页返回的 chunk 标签；`include_evidence=true` 时附上标签证据。
- `ingestion` 会读取入库 manifest，并检查 local 文件后端的点数，或实时查询 Qdrant collection 的点数及少量 payload 样例；不会返回向量值。若多个任务共用 collection，实时点数是整个 collection 的数量，不一定等于本次任务的写入数。

三个接口都要求相同的 Bearer API 访问令牌。列表接口采用分页，单页最多 100 条；正文和标签证据默认不返回。

### 查询进度

```bash
curl 'http://127.0.0.1:8010/api/v1/file-processing/jobs/<job_id>' \
  -H "Authorization: Bearer ${RAG_API_TOKEN}"

```

如果服务在文件处理任务运行期间重启，出于不将密钥落盘的考虑，任务不会自动恢复；请重新提交该任务。

## 2. 检索

`POST /api/v1/retrieval/search` 请求 JSON：

```json
{
  "run_id": "12位十六进制任务ID",
  "query": "问题文本",
  "llm_api_key": "调用方的大模型API Key",
  "llm_base_url": "https://api.example.com/v1",
  "llm_model": "服务商提供的模型名",
  "top_k": 5,
  "semantic_pool": 100,
  "dimension_pool": 20,
  "include_text": false
}
```

返回语义、维度、加权融合三路排名及查询解析信息。`include_text` 默认为 `false`；只有确实需要原始 chunk 内容时才设为 `true`。

## 3. 问答

`POST /api/v1/qa/ask` 请求 JSON：

```json
{
  "run_id": "12位十六进制任务ID",
  "query": "问题文本",
  "llm_api_key": "调用方的大模型API Key",
  "llm_base_url": "https://api.example.com/v1",
  "llm_model": "服务商提供的模型名",
  "top_k": 5,
  "context_chars": 0,
  "include_context_text": false
}
```

返回生成答案、实际检索查询、三路检索结果和上下文统计。`context_chars: 0` 表示将完整 chunk 提供给答案生成；`include_context_text` 控制是否在 API 响应中返回 chunk 正文，不影响模型使用上下文。

\
