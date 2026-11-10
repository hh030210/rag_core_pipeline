# Vector retrieval experiments

The vector strategy comparison starts from a raw QA JSON/JSONL file on every
run. `00_method_comparison/code/run_comparison_methods.py` freshly parses each
query and runs semantic retrieval, dimension retrieval, and fusion against the
active collection for each vector strategy. It remaps Golden evidence and
scores all three fresh routes for each strategy. The legacy metrics field
continues to summarize the semantic route.

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

If the query expansion API is unavailable, the runner may use the matching
expansion in the index run's query_expansion_cache.json. It uses the original
question when no expansion matches. Query parsing can replay an exact matching
archived analysis when LLM mode is enabled and the parser API fails.
