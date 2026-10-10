# Vector retrieval method comparison

仅保留五种方法：

|方法键|方法|向量子检索次数|Hit@5|
|---|---|---:|---:|
|subqueries_concat|子查询用 ` \| ` 拼接后编码|1|78.36%（315/402）|
|original_only|只编码原始查询|1|82.84%（333/402）|
|original_plus_subqueries_concat|原始查询与子查询用 ` \| ` 拼接后编码|1|78.11%（314/402）|
|original_subqueries_centroid_w2|原始查询权重2，子查询权重1；文本去重、向量加权求和后归一化|1|84.33%（339/402）|
|dense_ensemble|0.25整块余弦＋0.375独立短段最大余弦＋0.375上下文短段最大余弦|3|90.30%（363/402）|

当前保存同一批402条有映射Golden的比较样本；原输入435条，33条未映射样本不在这个对比数据集中。没有修改保留样本的Golden。四个查询组合方案沿用Qdrant候选池20、召回后景区过滤；纯向量集成沿用内存精确余弦、景区过滤后排序。现有数据为迁移保留的验证结果，生产search.py已支持策略分派，当前默认仍为子查询拼接。

短段窗口240字符、步长160字符。上下文短段先补充文档标题和前块末尾160字符。索引向量归一化，代码中的点积等于余弦。纯向量集成不使用词项匹配、维度标签或Golden打分。

code目录只包含五个策略.py和本说明；公共run.py、verify.py、build_index.py、test_strategy_switching.py在实验根目录，不属于可选策略。output只包含dataset：index.json、details逐条五种方法结果及指标、corpus.json和三组必要向量与短段元数据。不生成results.jsonl、summary或额外日志。

从服务器仓库根目录 `/home/humq/rag_core_pipeline` 执行：

```bash
# 重跑五种方法；默认复用本目录已保存的纯向量索引，成功后覆盖本目录output
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 EMBEDDING_BATCH_SIZE=32 OPENBLAS_NUM_THREADS=1 /home/humq/envs/denoise_qa/bin/python -m rag_core.vector_retrieval.experiments.00_method_comparison.run

# 同时从当前Qdrant重建纯向量集成索引并重新评测
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 EMBEDDING_BATCH_SIZE=32 OPENBLAS_NUM_THREADS=1 /home/humq/envs/denoise_qa/bin/python -m rag_core.vector_retrieval.experiments.00_method_comparison.run --rebuild-index

# 复算保存排名的指标；--fresh额外重新编码并检索三条代表查询，验证五种方法排名
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 /home/humq/envs/denoise_qa/bin/python -m rag_core.vector_retrieval.experiments.00_method_comparison.verify --fresh
```

可以用--input-dataset指定线上评测dataset，读取其中原始查询、子查询、解析信息和Golden；不修改来源文件。模型、整块向量和短段索引必须使用同一个BGE-M3模型，语料变更后应使用--rebuild-index。失败保留旧output，成功只替换本实验output。无旧实验目录依赖。

成本按单个方法使用计：前三种只编码一个文本；加权平均编码去重后的多个文本，但最终仅检索一次；纯向量集成只编码原始查询一次，检索三个向量索引。前三种及加权平均每种调用Qdrant一次；当前纯向量集成在内存计算，Qdrant请求为0，但逻辑子检索次数仍为3。运行整个对比会复用批量编码，不能把编码调用次数视为子检索次数。


## 策略切换

在 `rag_core/vector_retrieval/search.py` 开头配置文件名：

```python
VECTOR_RETRIEVAL_STRATEGY = "dense_ensemble.py"
```

可选文件：
- `subqueries_concat.py`：子查询拼接。
- `original_only.py`：仅原始查询。
- `original_plus_subqueries_concat.py`：原始查询加子查询拼接。
- `original_subqueries_centroid_w2.py`：查询向量加权平均。
- `dense_ensemble.py`：纯向量集成。

默认 `subqueries_concat.py`。配置在新建VectorRetriever时读取；修改后应重启当前运行。对比入口run.py始终运行全部五种方法，不受默认策略配置影响。

每个策略文件定义Strategy类，提供query_texts(parts, original)和search(parts, original, vectors, limit, spot_names)。公共search.py负责加载策略、补齐编码和旧接口兼容。主流程为所选策略追加需要的编码文本，共用批量编码，维度查询向量保持原逻辑。

旧调用若只传query_vector，默认它对应子查询拼接文本；切换策略时不会错误复用它。可通过query_vector_text明确对应文本，或通过query_vectors传入文本到向量的映射。纯向量集成索引首次使用时加载并缓存在实例中；读取自己的output/dataset，校验集合名称，返回原始块payload。

回归检查：
```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 /home/humq/envs/denoise_qa/bin/python -m rag_core.vector_retrieval.experiments.00_method_comparison.test_strategy_switching
```
