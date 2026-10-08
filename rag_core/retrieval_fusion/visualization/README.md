# Fusion Retrieval Visualization

Double-click `start.bat` to rebuild the viewer data and open the page in your browser. The local server runs until you press `Ctrl+C` in the console window.

By default, the builder first reads every `retrieval_fusion_*.json` snapshot in `../fusion_engine/output/`. Each snapshot represents one retrieval call and provides the semantic, dimension, and fused candidate routes, along with chunk content and (for snapshots produced by the current code) the query. Fusion snapshots do not include gold labels, so this mode does not report hit rates.

If the fusion output directory has no snapshots, the builder selects the newest complete run under `runs/`. If there is no complete run there yet, it reads the bundled experiment under:

```text
../result/real_merged7_deepseek_v4pro_rerank_20260920_run1/
```

In evaluation mode it reads that run's evaluation JSONL and chunk corpus, plus `tags_output_v2.json` and `dimension_metadata_v2.json` when available in the run directory or one of its parent directories. Both modes regenerate `fusion_routes.json`, `rerank_chunk_contents.json`, and `chunk_dimension_tags.json` in this folder. The page shows parsed query dimensions, each dimension-retrieved chunk's tags and tag confidence, and retrieval match similarity/IDF values where recorded.

To select the evaluation JSONL mode explicitly, choose a run directory:

```bat
start.bat --run-dir runs\my_latest_run
```

To select a fusion snapshot file or directory explicitly:

```bat
start.bat --fusion-output rag_core\retrieval_fusion\fusion_engine\output
start.bat --fusion-output rag_core\retrieval_fusion\fusion_engine\output\retrieval_fusion_20261008_120000_000000+0800.json
```

Paths passed to the scripts are resolved relative to the repository root. If the run uses a different layout, provide explicit source paths:

```bat
start.bat --run-dir runs\my_latest_run --evaluation runs\my_latest_run\evaluation\results.jsonl --chunks runs\my_latest_run\chunks.json
```

If the tag files are stored elsewhere, pass them explicitly:

```bat
start.bat --run-dir runs\my_latest_run --tags runs\my_latest_run\tags_output_v2.json --dimension-metadata runs\my_latest_run\dimension_metadata_v2.json
```

The repository currently contains raw results for semantic, dimension, and fusion retrieval. It does not contain standalone DeepSeek Listwise or Answerability rerank result files. Those columns appear when a rerank JSON/JSONL file is supplied with per-question `deepseek_results` and/or `answerability_results` arrays. JSONL row example:

```json
{"id":"1","deepseek_results":[{"chunk_id":"doc::chunk_0001","rerank_score":0.82}],"answerability_results":[{"chunk_id":"doc::chunk_0001","answerability":0.91,"quality_score":0.8,"rerank_score":0.86}]}
```

Pass that file with `--run-dir runs\my_latest_run --rerank-results path\to\rerank_results.jsonl` to select evaluation mode explicitly.
