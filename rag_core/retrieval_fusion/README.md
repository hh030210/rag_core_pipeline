# 融合检索：目录、策略与运行说明

本文说明当前目录规则、策略入口、输出格式及结果版本。主流程评测与融合方法对比分别使用独立入口。

## 当前目录

```text
retrieval_fusion/
├── README.md
├── fusion.py
├── run_run_py_evaluate.py
├── visualization/
└── experiments/
    └── 00_method_comparison/
        ├── code/
        │   ├── online_adaptive.py
        │   ├── dimension_score.py
        │   ├── recommended_fusion.py
        │   └── run_comparison_methods.py
        └── output/
            ├── online_adaptive/
            │   ├── index.json
            │   └── evaluate_*.json
            ├── dimension_score/
            │   ├── index.json
            │   └── evaluate_*.json
            └── recommended_fusion/
                ├── index.json
                └── evaluate_*.json
```

`00_method_comparison` 仅包含 `code` 与 `output`。`code` 中每种策略对应一个 Python 文件，另保留唯一批量入口 `run_comparison_methods.py`，该入口不属于可选策略。

每个策略文件包含自身所需的评分、归一化、候选处理和独立回放逻辑，不依赖其他策略文件或 `tools`；公共函数允许重复。

`output/<方法名>` 与策略文件名去掉 `.py` 后一致。全部输出 JSON 直接放在方法目录，不再嵌套 `dataset` 或 `details`。`index.json` 是批次元数据及查询文件清单，`detail_files` 的值是同目录 JSON 文件名。候选排序、查询、Gold、正文及必要诊断均保存在方法自己的输出中。

每种方法成功运行后覆盖自己的输出，失败保留该方法上一批数据；批量运行中已经成功发布的方法不会因后续方法失败而自动回退。不累积历史批次，不生成额外 summary 文件。

## 主线策略

在 `fusion.py` 开头设置：

```python
FUSION_STRATEGY = "dimension_score.py"
```

| 策略文件 | 用途 |
| --- | --- |
| `online_adaptive.py` | 自适应融合基准，根据候选数量和前五条重叠情况选择分数加权或维度名次分支 |
| `dimension_score.py` | 维度分数基准，在语义分数上加入归一化维度分数和词面分数，当前主线默认策略 |
| `recommended_fusion.py` | 维度角色校准，根据分数间隔、维度角色及局部文本证据调整融合贡献 |

`fusion.py` 的 `apply_fusion` 按文件名调用方法的 `fuse`，保留完整候选 payload。`run_comparison_methods.py` 不能填入配置。修改策略配置后重新启动主流程，已有结果不会自动重新生成。

## 主流程评测入口

从服务器仓库根目录 `/home/humq/rag_core_pipeline` 执行：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/humq/envs/denoise_qa/bin/python rag_core/retrieval_fusion/run_run_py_evaluate.py
```

此入口只执行仓库根目录的 `run.py evaluate`，不自动运行融合对比。数据只覆盖当前 `FUSION_STRATEGY` 对应的 `output/<方法名>`。

默认运行目录为 `/home/humq/rag_core_runs/real_merged7_sdu_v4pro_full_20260921_run1`，数据集为该目录中的 `qa_dataset_435.json`，编码模型为 `/home/humq/rag_db_silm/model/bge-m3`。默认 `top-k=10`、`eval-depth=20`、`ks=1,5,10,15`、查询解析为 `deterministic`。默认启用查询扩展，`--no-query-expansion` 可关闭；Qdrant 地址和集合默认读取运行目录的 `store_manifest.json`。

```bash
PYTHONDONTWRITEBYTECODE=1 /home/humq/envs/denoise_qa/bin/python rag_core/retrieval_fusion/run_run_py_evaluate.py --help
PYTHONDONTWRITEBYTECODE=1 /home/humq/envs/denoise_qa/bin/python rag_core/retrieval_fusion/run_run_py_evaluate.py --dry-run
```

`--dry-run` 只打印主流程命令，不评测、不写输出。

## 融合方法对比入口

```bash
PYTHONDONTWRITEBYTECODE=1 /home/humq/envs/denoise_qa/bin/python rag_core/retrieval_fusion/experiments/00_method_comparison/code/run_comparison_methods.py
```

对比入口读取已保存的语义候选、维度候选、查询分析、正文与 Gold，不调用主流程、编码模型或数据库。算法不使用 Gold 排序，Gold 仅用于评测。默认输入为当前主线策略的输出目录；在任何方法覆盖输出前，先固定全部输入，保证所有方法读取同一批数据。

| 参数 | 默认值 / 含义 |
| --- | --- |
| `--input-dataset` | 当前主线策略的平铺 JSON 输出目录 |
| `--output-dir` | `experiments/00_method_comparison/output`；这里指定各方法输出目录的父目录 |
| `--methods` | 全部三种策略，可用不带 `.py` 的方法名指定部分方法 |
| `--top-k` | `10`，控制最终结果截断，完整融合候选仍保存 |

各策略文件也可单独执行，例如：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/humq/envs/denoise_qa/bin/python rag_core/retrieval_fusion/experiments/00_method_comparison/code/dimension_score.py --help
```

单方法的 `--output-dir` 指定该方法的最终输出目录；这与批量入口指定输出父目录的含义不同。单方法默认输入仍是当前主线策略的输出，默认输出是自身的 `output/<方法名>`。

只修改融合公式时重跑对比即可；更换向量策略、维度检索或数据集后，先运行主流程更新输入，再运行融合对比。

## 新尝试的组织方式

当前没有保留正在尝试的方法。以后与 `00_method_comparison` 平级，按 `01_英文名称`、`02_英文名称` 编号，每种方法单独一个目录：

```text
experiments/01_new_method/
├── code/
│   └── new_method.py
└── output/
    ├── index.json
    └── 各查询.json
```

尝试目录只包含 `code`、`output`，代码目录只放一个实现兼运行入口的 `.py`，输出目录直接放 JSON。正式纳入对比时，将策略文件移入 `00_method_comparison/code`，输出移入同名方法子目录。

## 可视化与结果版本

`visualization` 与 `experiments` 平级，只保存界面代码与说明，直接读取各策略输出。主线标记读取 `fusion.py` 和向量模块 `search.py` 的配置，指标从候选排序和 Gold 现算，不读取预存汇总指标。

页面读取的是已保存的评测批次。改代码或切换策略不会自动执行评测；主流程与融合对比是两个独立入口。不同批次应核对输入和 Gold 分母。组织调整和入口检查没有重跑完整评测。
