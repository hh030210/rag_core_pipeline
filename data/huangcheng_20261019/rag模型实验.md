# RAG 知识库：召回与生成模型实验
---

## 1. 项目已接入模型

### 1.1 生成 LLM（回答主力）

| 短名 | model_id |
|------|----------|----------|
| Qwen Plus | `qwen-inference-plus-online`  |
| Qwen3.7 Plus | `qwen-inference-3.7-plus-online` |
| DeepSeek-V3 | `deepseek-inference-online` | 
| 豆包 Seed 2.0 Lite | `doubao-inference-lite-online` |

### 1.2 召回相关（Embedding / Rerank）

| 类型 | 短名 | model_id |
|------|------|----------|
| Embedding | text-embedding-v3 | `qwen-embedding-online` |
| Embedding | bge-large-zh-v1.5 | `deepseek-embedding-online` |
| Rerank | qwen-rerank | `qwen-rerank-online` |
| Rerank | deepseek-rerank | `deepseek-rerank-online` |

## 2. 知识库与测试集

景区：**皇城相府**（`scenic_id: 1398285657515941888`）。各组实验共用同一知识库、同一测试集，benchmark 为 `travel_scenic_qa_huangcheng`。

### 2.1 知识库源文件

路径：`agents/travel/app/scenic_data/scenic/1398285657515941888/sources/`

| 源文件 | 说明 |
|--------|------|
| `皇城相府故事.txt` | 人物、典故、家族史（主知识，约 64KB） |
| `景点介绍.txt` | 各 POI 介绍（河山楼、藏兵洞等） |
| `讲解词.txt` | 导览讲解词 |
| `景区全景方位说明.txt` | 全景方位 |
| `推荐路线.txt` | 游览路线推荐 |
| `运营信息.txt` | 开放时间、门票、简介 |
| `运营提问.txt` | 餐饮、停车等运营 FAQ |
| `游客高频问题清单.txt` | 高频问答 |
| `节目清单.txt` | 演出节目 |


主流对标时，将上述 `sources/` 或 `knowledge_base.jsonl` 导入各平台，保持内容一致。

### 2.2 测试集

| 集 | 路径 | 规模 | 用途 |
|----|------|------|------|
| **多跳草稿** | `core/eval_service/configs/tasks/base/_scenic_qa_huangcheng_multihop_draft - Sheet1.csv` | 239 条 | 多跳 / 跨段推理；列 `query`、`参考` |

CSV 示例：「陈廷敬当官那些年，跟步步高那五十三级是怎么对上的呀？」——参考答案须同时引用「为官五十三年」与「五十三级台阶」两段原文。

**建议用法**：

- 召回 / 生成 / 主流对标：**主集全量**（191 条），各组一致。
- 多跳专项：在主集达标后，用 CSV 239 条补测；入库前需转为 task yaml（带 `rag_recall_keywords`、`answer_hit` 等字段）。

---

## 3. 主流知识库问答系统对标

### 3.1 对标对象

| 系统 | 定位 | 与我们的关系 |
|------|------|-------------|
| **FastGPT** | 专用知识库问答 | **最直接对标** |
| **RAGFlow** | 文档解析 + RAG 引擎 | 复杂 PDF / 引用溯源 |
| **MaxKB** | 低代码私有化知识库 | 部署快、调优深度一般 |
| 阿里云百炼知识库 | 云厂商托管 | 省心、定制弱 |

### 3.2 六维对比表

| 维度 | 外部 RAG + Agent | FastGPT | RAGFlow | MaxKB |
|------|------------------|---------|------|---------|
| Recall@5 | | | | | 
| answer_keywords 命中 | | | | | 
| answer_quality_llm 均值 | | | | | 
| 混合检索 | | | | | 
| Rerank | | | | | 
| 领域增强（地点/标签） | 有 | 无 | 无 | 无 |

