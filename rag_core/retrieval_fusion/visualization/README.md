# 融合检索可视化

页面运行时直接读取 `../fusion_engine/output/` 与 `../fusion_engine/output_new/` 中的快照，无需执行 `build_data.py`，也不依赖预生成的展示 JSON。`serve_ui.py` 提供本地只读接口：问题目录来自两个快照目录，打开某题时才读取对应的原融合和新融合 JSON。页面不会触发检索、模型调用或新的融合计算。

`output_new/comparison.jsonl` 若存在，会为页面提供 Gold 数量及语义、维度、原融合、新融合的首个 Gold 名次，用于 badcase 筛选和统计；没有该文件时仍可查看四列候选，但不显示 Gold 指标。这个比较文件本身位于 `output_new`，页面不读取外部评测目录。

在仓库根目录运行：

```bash
python3 rag_core/retrieval_fusion/visualization/serve_ui.py --no-browser
```

服务默认监听 `127.0.0.1:8765`。在远端主机运行时，可通过 SSH 转发该端口，然后打开 `http://127.0.0.1:8765/index.html`。在本地 Windows 工作区也可双击 `start.bat`。如需更换目录，可传入 `--fusion-output` 和 `--fusion-output-new`。

页面默认显示各路 Top-5，可展开完整候选；点击 chunk 编号查看原始快照中的正文。原融合与新融合按相同文件名、query 和候选集合配对，详情接口发现不一致时会报错，避免错误对比。
