# run33 最后一次运行结果

完整流程及运行方法：[代码说明](../../experiments/collective_context_first/README.md)。

- [完整结果](reports/本轮结果.md) / [全部机器与LLM指标](results.json)
- [实际各阶段提示](reports/各阶段实际提示.md)
- [代表性问题（含实际top5）](reports/代表性问题案例.md)
- [评分执行复核：保留原分，不隐藏矛盾](reports/评分执行复核.md)
- [逐题数据](reports/全量逐题对照.json)
- [独立核验](reports/独立核验.json)
- [逐题网页](旅游群智_完整流程.htm)：下载后用浏览器打开；GitHub文件页不执行HTML。
- `audit_archive.zip`：全部冻结源码、23训练/378测试输入、案例迭代、8簇、反馈记录、1834次原始调用、评价视图缓存、技术修复证明和报告。仅排除Python字节码缓存；未改写原文件内容。
- `archive_inventory.json`：归档逐文件字节数与SHA256。

本批为探索性结果；有效答案377/378，完整评价376/378，2条失败保留；适用性null不填零。群智反馈没有候选被接受，不宣称反馈改善。本轮没有重新运行裸q+top5，不能与旧裁判口径直接比较。模型裁判有已披露的分数/理由不一致。
