# Retrieval experiments

All experiment entry points in this directory run retrieval against the active
collection. They load a raw QA JSON/JSONL file and remap its Golden evidence to
the current collection payloads at the start of each run. Experiment output
directories are never used as retrieval inputs.

The shared runtime is in `common/live_runtime.py`. It records the raw dataset
path and SHA-256, active collection, embedding model, strategy settings, source
commit, and a unique `run_id`. Query parser and optional query expansion caches
are temporary and isolated to that run. The main three-route evaluation defaults
to fresh LLM query parsing. Strategy comparisons default to deterministic
parsing. Query expansion is a separate option enabled with `--query-expansion`.

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

## Fusion strategy comparison

`00_method_comparison/code/run_comparison_methods.py` performs a fresh
semantic/dimension retrieval from the active store for each raw question, then
applies every selected fusion strategy to those same in-run candidates. It
writes each method's details and metrics under a unique `output/runs/<run_id>/`
directory. The strategy modules expose scoring functions only; saved route
snapshots cannot be passed to them as an alternate experiment input.

Both entry points accept common runtime options such as `--dataset`,
`--run-dir`, `--backend`, `--qdrant-url`, `--collection`, `--model-path`,
`--device`, candidate pool sizes, `--query-parser-mode`, and
`--query-expansion`. Check `--help` for the full option list.
