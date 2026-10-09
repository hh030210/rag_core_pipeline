# 本次 run.py evaluate 重跑

launch.py 调用仓库根目录的原始 run.py evaluate。为了把所有新增文件保存在本目录，复制运行索引/缓存/chunks到run_input，并在本进程将融合快照路径重定向到fusion_snapshots/output。仍查询原Qdrant collection，不重新入库。

配置：query expansion开启并复用既有缓存；query parser为默认deterministic；semantic_pool=20、dimension_pool=100；top_k=10、eval_depth=20；K=1,3,5,10,15,20。完整参数见invocation.json，运行日志见evaluate.log。

435条查询，402条Golden已映射，33条未映射。evaluation/summary.json是原评测程序输出，指标按435条统计；evaluation/mapped_summary.json额外按402条已映射样本统计，方便与向量探索实验比较。

原评测输出保存在evaluation/，435条融合快照保存在fusion_snapshots/output/，原运行目录与已有可视化未覆盖。
