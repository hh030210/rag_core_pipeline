# 向量检索：目录、策略与运行说明

本文说明当前目录规则、策略入口、输出格式及结果版本。每种方法的逻辑保存在自己的策略文件中，纯向量集成使用即时编码。

## 当前目录

```text
vector_retrieval/
├── README.md
├── __init__.py
├── search.py
└── experiments/
    └── 00_method_comparison/
        ├── code/
        │   ├── subqueries_concat.py
        │   ├── original_only.py
        │   ├── original_plus_subqueries_concat.py
        │   ├── original_subqueries_centroid_w2.py
        │   ├── dense_ensemble.py
        │   └── run_comparison_methods.py
        └── output/
            ├── subqueries_concat/
            ├── original_only/
            ├── original_plus_subqueries_concat/
            ├── original_subqueries_centroid_w2/
            └── dense_ensemble/
```

`00_method_comparison` 仅包含 `code` 与 `output`。每种策略一个 `.py`，另有唯一批量入口 `run_comparison_methods.py`，该入口不属于可选策略。

各策略文件包含自身所需的查询组合、归一化、评分和候选处理逻辑，不依赖其他策略文件或 `tools`，公共函数允许重复。策略提供 `Strategy` 接口供 `search.py` 调用，没有独立评测命令行入口。

`output/<方法名>` 与策略文件名去掉 `.py` 后一致，直接包含 `index.json` 和 `query_*.json`，不嵌套 `dataset`、`details`。索引保存查询文件清单与批次元数据，`detail_files` 的值是同目录 JSON 文件名。每份逐查询 JSON 只保存自身方法的候选排序、Gold、查询和成本等数据。

一次完整对比成功后统一覆盖五种方法输出，失败保留上一批，不积累历史批次，不生成额外 summary 文件。

## 主线选择与策略说明

在 `search.py` 开头设置：

```python
VECTOR_RETRIEVAL_STRATEGY = "original_subqueries_centroid_w2.py"
```

| 文件 | 查询处理 |
| --- | --- |
| `subqueries_concat.py` | 子查询使用分隔符拼接，编码为一个查询向量 |
| `original_only.py` | 只编码原始查询 |
| `original_plus_subqueries_concat.py` | 原始查询加子查询拼接后整体编码 |
| `original_subqueries_centroid_w2.py` | 原始查询与子查询去重、分别编码并单位化；原始权重 2，其余权重 1，加权和再次单位化 |
| `dense_ensemble.py` | 原始查询向量与完整 chunk、独立短段、上下文短段三组向量计算分数后融合 |

可填写上表策略文件名，不能填写批量入口。修改后重启主流程或新建 `VectorRetriever`；已有结果不会自动更新。

加权平均公式为 `v = U(2U(E(q0)) + sum(U(E(qi))))`，子查询求和不包含与原始查询相同的文本。前四种方法各做一次数据库向量检索，沿用查询分析中的景区过滤。

`VectorRetriever` 接收编码器、向量库和可选向量字段；`query_texts` 返回策略所需文本，`search` 支持原始查询、子查询、候选池和文本到向量的映射。只有对应文本一致时才复用传入向量，不把子查询拼接向量误作原始查询向量。候选保留完整 payload，供融合使用。

## 对比运行入口

在仓库根目录执行：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/humq/envs/denoise_qa/bin/python \
  rag_core/vector_retrieval/experiments/00_method_comparison/code/run_comparison_methods.py
```

此入口独立执行五种方法，不调用 `run.py evaluate`。它读取输入 JSON 中的原始查询、子查询、查询分析和 Gold，重新编码查询并检索当前数据库；旧候选排序不作为新结果，Gold 只用于评测。它不会重新生成子查询。

| 参数 | 默认值与含义 |
| --- | --- |
| `--input-dataset` | `experiments/00_method_comparison/output/original_only`，提供查询与标注 |
| `--run-dir` | `/home/humq/rag_core_runs/real_merged7_sdu_v4pro_full_20260921_run1`，读取向量库清单 |
| `--model-path` | `/home/humq/rag_db_silm/model/bge-m3` |
| `--device` | `cuda:0` |
| `--pool` | `20`，前四种方法的候选池大小，必须至少为 20 |

该入口没有 `--methods` 或 `--output-dir` 参数；每次执行全部五种方法，写入固定的五个输出目录。内部统计 `@1/5/10/15/20`，终端汇报 `Hit@5/10/15`。所有方法成功后整体替换输出，失败则保留原有结果。

```bash
/home/humq/envs/denoise_qa/bin/python \
  rag_core/vector_retrieval/experiments/00_method_comparison/code/run_comparison_methods.py --help
```

若更换数据集或查询改写结果，应通过 `--input-dataset` 指向新的查询 JSON 目录。编码器和数据库沿用项目已有接口，不需要先执行主流程。

## 纯向量集成的当前实现

每次查询都通过 Qdrant `scroll` 读取当前集合的完整 chunk 向量和 payload，每页 256 条；完整 chunk 向量直接使用数据库中的 `chunk_text_vec`。独立短段与上下文短段在本次查询中即时编码，不预先保存索引，也不跨查询缓存。

短段窗口为 240 字符，步长 160。上下文由文档标题、同文档前一 chunk 的末尾 160 字符与当前 chunk 组成；前一 chunk 根据 `::chunk_####` 编号定位。

评分为 `0.25 * chunk_cosine + 0.375 * max(independent_cosine) + 0.375 * max(context_cosine)`。对全库候选评分，批量评测保留前 100 条结果。因此它的成本包含多页数据库读取和候选短段编码，不能把三组向量得分理解为三次数据库检索。本方法只读数据库。

## 新尝试的组织方式

当前没有正在尝试的编号实验。以后新方法与 `00_method_comparison` 平级，按 `01_english_name`、`02_english_name` 顺序命名：

```text
experiments/01_english_name/
├── code/
│   └── english_name.py
└── output/
    ├── index.json
    └── query_*.json
```

每个尝试目录的 `code` 只放一个方法文件，包含运行所需逻辑；`output` 直接存 JSON，不再套方法名或 `dataset`。纳入正式对比时，将方法文件放入 `00_method_comparison/code`，输出放入同名子目录，并更新批量运行入口与可视化。

## 可视化与结果版本

融合模块的 `visualization` 直接读取本模块五个方法目录中的 JSON，按候选与 Gold 计算指标，并从 `search.py` 读取主线策略标记。修改配置不会改变已有评测数据，必须重新执行相应评测。

当前向量对比各保存 402 条查询，融合对比各保存 435 条查询，跨模块比较时需核对输入批次与 Gold 分母。现存输出包含先前版本的评测结果。纯向量集成改为即时编码后尚未完成五方法全量重跑，因此旧输出中的耗时、编码次数等成本不能代表当前实现；指标展示也应结合保存批次判断，不能把历史结果视为刚运行的结果。

推荐运行时设置 `PYTHONDONTWRITEBYTECODE=1`，避免在方法目录产生 `__pycache__`。
