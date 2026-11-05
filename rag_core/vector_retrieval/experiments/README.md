# Vector retrieval experiments

The vector strategy comparison starts from a raw QA JSON/JSONL file on every
run. `00_method_comparison/code/run_comparison_methods.py` freshly parses each
query and searches the active vector collection for each strategy. It does not
load query analyses, Golden mappings, rankings, or candidate snapshots from a
previous method output directory.

Results are written to a unique
`00_method_comparison/output/runs/<run_id>/` directory, with one JSONL detail
file and summary per vector strategy plus a run manifest. The manifest records
the dataset hash, collection, model, active settings, source commit, methods,
and a shared run ID. Old output folders are retained as historical data and
are never used as input to a new run.

Example:

```bash
python3 rag_core/vector_retrieval/experiments/00_method_comparison/code/run_comparison_methods.py \
  --dataset /path/to/qa_dataset_435.json \
  --run-dir /path/to/current/index/run
```

Parser mode is deterministic by default. Pass `--query-parser-mode llm` to
perform fresh LLM query parsing, or `--query-expansion` to generate subqueries
for this run. Parser and expansion caches are isolated to a temporary
run-specific directory. Use `--help` for backend, model, candidate pool,
strategy, and output options.
