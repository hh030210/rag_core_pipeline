# 新景区上下文 Top5 双组对照

数据源仅为师弟目录中的 `scenic_after_top10_chunks.jsonl.gz`，使用 `routes.fusion` 的原顺序前5条，不使用 semantic/dimension 的结果，也不插入 golden。
原始402题中，23题是原训练题，排除出测试集；1题融合结果仅4条，排除。最终两组均为378道独立测试题，匹配原chapter3测试集参考答案；训练仍为原24条golden案例。

两组为 `with_policy`（含噪音与重排序筛选说明）和 `without_policy`（不含两条说明）。两组都从同一个中性五模块 meta-prompt 逐案例生成初始提示，调用参数仅在上下文政策指导上有区别。控制组的初始、逐轮、群智及最终使用提示均检查是否重新出现被排除的说明；不使用会自动补回两条说明的旧默认路径。
本次不是读取原实验提示后删两句话，而是两组各自完成初始生成、自适应迭代和群智归纳。

共同设置：Qwen3-8B、本地同一vLLM端点；workers=80；seed=20261017；三维评分均为0–100分用于训练，总分算术平均；任一维度下降或总分提升<1分时回退并停止，最多30次改写；问题+参考答案BGE向量聚类、6簇；推理temperature=0.1、max_tokens=1800。两组共用只由相同问题和训练参考答案计算的BGE向量缓存，避免重复编码，测试路由只使用问题向量，不使用测试答案。

最终评价沿用原chapter3机械指标与Qwen三维1–5分评分，单裁判，不调用DeepSeek。训练的0–100分不混入最终1–5分表。

运行与状态：

```bash
cd /pub/yanglw/rag_core_pipeline
/pub/yanglw/miniconda3/bin/python -m rag_core.scenic_ablation_job start --workers 80
/pub/yanglw/miniconda3/bin/python -m rag_core.scenic_ablation_job status
```

默认目录 `runs/scenic_new_top5_ablation_20261017/`。`shared/` 是共同训练/测试数据、来源SHA256、排除清单和向量缓存；两个同名组目录分别保存独立案例迭代、群智提示、答案和 `evaluation_qwen/` 最终评分；根目录 `comparison.json/csv/md` 是对比汇总。
历史旧上下文397题结果不覆盖、不混入本次378题对照。

运行中控制组有1个案例的改写JSON在两次尝试后仍有语法错误，原控制组中间产物保存在 `backups/scenic_ablation_control_json_retry_20261017/`。
补充了仅修复JSON语法、仍检查同一组提示约束的修复调用；重新运行该案例，复用其余23个正常完成案例，再重新生成控制组群智提示。不把格式错误当作评分收敛，也不接受含被排除说明的修复结果。
