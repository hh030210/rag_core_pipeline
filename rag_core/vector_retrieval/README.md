# 向量检索封装

本目录封装语义向量召回，供 rag_core.retrieval.Retriever 调用。它与维度检索、事实补充和结果融合分开，后续可以在本目录独立调整向量召回策略。

## 接口

VectorRetriever 初始化时注入：

- embeddings：文本向量编码器。
- vector_store：项目当前的 VectorStore 实例。
- vector_name：向量字段名，默认 chunk_text_vec。

每次调用 search() 时传入：

- subqueries：子查询列表。封装内部用 | 拼接为本次检索文本。
- semantic_pool：语义向量召回的候选数。
- original_query：可选；子查询为空时作为回退文本。
- query_vector：可选的预计算向量。上层传入时可继续与维度查询向量批量编码。
- spot_names：可选的景区或 POI 范围，用于保留当前的候选过滤。

返回值是按语义分数排序的候选列表，每项保留 chunk_id、score、source、rank 和原始 payload 字段，与原 Retriever 的语义候选格式一致。

## 当前调用流程

上层先将查询子句解析、编码，并把已经编码的主查询向量传入 search()；向量封装使用 semantic_pool 调用向量存储，再执行现有景区范围过滤和排名编号。若独立调用时没有传 query_vector，封装会自行编码拼接后的子查询文本。

top_k 仍由上层用于最终融合结果截断；它不是语义向量召回池大小。维度检索与融合逻辑仍由 Retriever 和 retrieval_fusion 负责。


## 实验目录

`experiments/00_method_comparison/` 集中保存五种方法：子查询拼接、仅原始查询、原始查询加子查询拼接、查询向量加权平均、纯向量集成。

- `code/`：五个可选策略文件，每种方法一个 `.py`。
- 实验根目录 `run.py`、`verify.py`、`build_index.py`：公共运行、验证和索引构建入口。
- `output/dataset/`：五种方法的逐条结果、指标及必要向量索引。

详细运行说明见 `experiments/00_method_comparison/code/README.md`。从仓库根目录执行：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/humq/envs/denoise_qa/bin/python -m rag_core.vector_retrieval.experiments.00_method_comparison.run --help
```

生产检索由根目录 `search.py` 分派到所选策略。在文件开头修改：

```python
VECTOR_RETRIEVAL_STRATEGY = "subqueries_concat.py"
```

可填写 `code` 目录中的任意策略文件名（包含 `.py`）。修改后重启运行或新建VectorRetriever实例。当前默认子查询拼接，主流程会按策略所需文本进行批量编码，再执行语义检索。纯向量集成从实验output/dataset读取三组索引，返回完整payload供融合流程使用。

