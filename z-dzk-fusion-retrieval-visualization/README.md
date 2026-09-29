# Fusion Retrieval Visualization

Double-click `start.bat` to rebuild the viewer data and open the page in your browser. The local server runs until you press `Ctrl+C` in the console window.

By default, the builder selects the newest complete run under `runs/`. If there is no complete run there yet, it reads the bundled experiment under:

```text
../result/real_merged7_deepseek_v4pro_rerank_20260920_run1/
```

It reads that run's evaluation JSONL and chunk corpus, plus `tags_output_v2.json` and `dimension_metadata_v2.json` when available in the run directory or one of its parent directories. It then regenerates `fusion_routes.json`, `rerank_chunk_contents.json`, and `chunk_dimension_tags.json` in this folder. The page shows parsed query dimensions, each dimension-retrieved chunk's tags and tag confidence, and retrieval match similarity/IDF values where recorded. You can also choose a run directory explicitly:

```bat
start.bat --run-dir runs\my_latest_run
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

Pass that file with `--rerank-results path\to\rerank_results.jsonl`.
