# Dataset catalog

This repository contains the source datasets and retrieval artifacts used by the current experiments.

## Source datasets

- Scenic attraction documents: `data_input/test_data/`.
- DuRetrieval-MTEB dev: `data/dev_external/duretrieval_mteb/`, including the corpus, queries, source data, and prepared full / qrels20k versions.
- db_qa full run input: `data/dbqa_full_20261016/db_qa.txt` (10,451 source lines), with the associated 3,199-query set in `1doc_QA.json` and provenance in `input_manifest.json`.
- Additional evaluation sets: `data/fair_qa_500*.json` and `data/merged_7_rag_test_set.json`.

## Full three-route Top10 chunk results

`data/three_route_chunk_details/` contains the complete compressed JSONL results for the scenic dataset before and after optimization, DuRetrieval-MTEB dev, and db_qa. Each record contains the query and actual semantic, dimension, and fusion hits, including chunk text and gold-hit status.

- 6,003 query/version records across the four archives.
- 177,395 retrieved chunk records.
- The archive manifest records per-dataset query and route-hit counts; all reported chunk text is present.

The accompanying Word report with example queries is at `artifacts/三数据集三路检索Chunk结果明细.docx`.
