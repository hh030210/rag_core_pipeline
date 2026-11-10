# Retrieval experiments

All experiment entry points in this directory run retrieval against the active
collection. They load a raw QA JSON/JSONL file and remap its Golden evidence to
the current collection payloads at the start of each run. Experiment output
directories are never used as retrieval inputs.

The shared runtime is in `common/live_runtime.py`. It records the raw dataset
path and SHA-256, active collection, embedding model, strategy settings, source
commit, and a unique `run_id`. Query parser and optional query expansion caches
are temporary and isolated to that run. The scenic main evaluation defaults
to fresh LLM query parsing. The Huangcheng and DuRetrieval main evaluations
and strategy comparisons default to deterministic parsing. Query expansion is a separate option enabled with `--query-expansion`.

## Three-route evaluation

Run `00_main_experiment/code/export_retrieval_details.py` to freshly retrieve
the raw dataset through semantic, dimension, and configured fusion routes. The
three JSONL details and `details_manifest.json` are written under
`00_main_experiment/output/runs/latest/`. Each successful export replaces
the previous result, including its reports. The new manifest still records a
unique run ID.

The default parser uses `LLM_API_KEY` or `DASHSCOPE_API_KEY` when available.
Set `LLM_BASE_URL` and `LLM_MODEL` to use the deployed service's endpoint and model.
When the API service gets these values through systemd, launch the evaluation
with the same EnvironmentFile settings. If live parsing is unavailable, this run
uses the archived LLM analyses in
`result/scenic_latest_gitmain_20261104/retrieval_results.jsonl.gz` for the
remaining questions. Each replay matches the record ID, question, and retrieval
query. All three routes search the current collection. The report lists the
parser sources, replay reason, and archive path. With a known API outage, pass
`--replay-query-analysis` to start directly with the archive. Pass `--query-parser-mode deterministic`
to use the rule-based parser for the entire run.

Then calculate the unified report in the same directory. For the default run
location, this also refreshes `00_main_experiment/output/evaluation_report.md`:

```bash
python3 rag_core/retrieval_fusion/experiments/00_main_experiment/code/export_retrieval_details.py
python3 rag_core/retrieval_fusion/experiments/00_main_experiment/code/calculate_metrics.py \
  --output-dir rag_core/retrieval_fusion/experiments/00_main_experiment/output/runs/latest
```

The metric script rejects outputs without a `live_end_to_end` manifest, a run
ID, and matching route rows. Chunk-level macro metrics include every raw QA
row; questions whose Golden evidence cannot be mapped to the active collection
are counted as misses. Micro chunk recall uses mapped Golden chunks as its
denominator. Document precision uses the returned chunks from all evaluated
questions as its denominator.

## Additional main datasets

Use the same two main experiment scripts with the same `--dataset-profile`
in both commands. Each profile reads its original QA and Golden annotations
and searches the matching Qdrant collection anew for semantic, dimension,
and fusion rankings.

| Profile | QA set | Collection | Fixed Markdown report |
| --- | --- | --- | --- |
| `scenic` (default) | Scenic attractions, 435 questions | `rag_core_sdu_v4pro_full_20260921_run1` | `00_main_experiment/output/evaluation_report.md` |
| `huangcheng` | Huangcheng high-precision set, 126 original questions | `rag_core_huangcheng_20261019_v1` | `00_main_experiment/output/huangcheng/evaluation_report.md` |
| `duretrieval_mteb` | DuRetrieval MTEB dev run9, 2,000 questions | `duretrieval_mteb_dev_qrels20k_dim_fresh_20260923` | `00_main_experiment/output/duretrieval_mteb/evaluation_report.md` |

For the two new profiles, detailed outputs are under the report's
sibling `runs/latest/` directory. The default model is the server's local
BGE-M3, with query expansion disabled and deterministic parsing.

```bash
experiment=rag_core/retrieval_fusion/experiments/00_main_experiment
for profile in huangcheng duretrieval_mteb; do
  python3 "$experiment/code/export_retrieval_details.py" --dataset-profile "$profile" --device cuda
  python3 "$experiment/code/calculate_metrics.py" --dataset-profile "$profile"
done
```

## Fusion strategy comparison

`00_method_comparison/code/run_comparison_methods.py` reads the raw
questions and runs all three routes independently for every selected fusion strategy. Each
strategy runs semantic retrieval, dimension retrieval, and fusion, then writes
per-route details and metrics under a unique `output/runs/<run_id>/`
directory. The strategy modules expose scoring functions only; saved route
snapshots cannot be passed to them as an alternate experiment input.

Both entry points accept common runtime options such as `--dataset`,
`--run-dir`, `--backend`, `--qdrant-url`, `--collection`, `--model-path`,
`--device`, candidate pool sizes, `--query-parser-mode`, and
`--query-expansion`. Check `--help` for the full option list.

When query expansion is enabled, a failed API request falls back to matching
question text in the run directory's query_expansion_cache.json. Questions
without a saved expansion use their original text. If LLM parsing also fails,
a saved parse is used only for an exact match of the ID, original question,
and retrieval query. Other queries use the rule-based parser.

The pipeline CLI (rag_core.cli evaluate) publishes separately under its
configured output directory, which defaults to the index run's evaluation
directory. It does not supply experiment inputs.
