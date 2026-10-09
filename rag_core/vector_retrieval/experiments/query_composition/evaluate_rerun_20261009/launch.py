"""Run the real root run.py evaluate with all writable artifacts isolated."""
import json
import os
import runpy
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/humq/rag_core_pipeline")
SOURCE = Path("/home/humq/rag_core_runs/real_merged7_sdu_v4pro_full_20260921_run1")
HERE = Path(__file__).resolve().parent
COPY = HERE / "run_input"
COPY.mkdir(exist_ok=False)
for filename in (
    "run_manifest.json", "store_manifest.json", "V_core_v2.json", "V_cand_v2.json",
    "inverted_index_v2.json", "tags_output_v2.json", "query_expansion_cache.json",
    "query_cache_v4_deterministic.json", "query_cache_v4.json",
    "dimension_tag_evidence_vectors_v2.json", "dimension_tag_evidence_vectors_v2.npy",
    "dimension_metadata_v2.json", "dimension_hierarchy_v2.json",
    "dimension_diagnostics_v2.json", "fact_index_v1.json", "optimized_prompt.json",
):
    src = SOURCE / filename
    if src.exists():
        shutil.copy2(src, COPY / filename)
(COPY / "chunks").mkdir()
shutil.copy2(SOURCE / "chunks/chunks.json", COPY / "chunks/chunks.json")
shutil.copy2(SOURCE / "qa_dataset_435.json", HERE / "dataset.json")
# The saver resolves its output from the module file path. Redirect only this
# process; source files and the production snapshot directory remain untouched.
import rag_core.retrieval_fusion.fusion_engine.fusion as fusion
fusion.__file__ = str(HERE / "fusion_snapshots" / "fusion.py")
os.environ["EMBEDDING_DEVICE"] = "cpu"
os.environ["SEMANTIC_POOL"] = "20"
os.environ["DIMENSION_POOL"] = "100"
args = [
    "evaluate", "--run-dir", str(COPY), "--dataset", str(HERE / "dataset.json"),
    "--output", str(HERE / "evaluation"),
    "--model-path", "/home/humq/rag_db_silm/model/bge-m3",
    "--backend", "qdrant", "--qdrant-url", "http://127.0.0.1:6333",
    "--collection", "rag_core_sdu_v4pro_full_20260921_run1",
    "--top-k", "10", "--eval-depth", "20", "--ks", "1,3,5,10,15,20",
    "--query-parser-mode", "deterministic",
]
(HERE / "invocation.json").write_text(json.dumps({
    "root_entrypoint": str(ROOT / "run.py"), "args": args, "source_run_dir": str(SOURCE),
    "isolation": "copy index/cache/chunks into run_input; process-local fusion saver path redirect",
    "query_expansion": "enabled; reuse copied existing cache",
    "semantic_pool": 20, "dimension_pool": 100,
    "started_at": datetime.now().astimezone().isoformat(),
}, ensure_ascii=False, indent=2), encoding="utf-8")
sys.argv = [str(ROOT / "run.py"), *args]
runpy.run_path(str(ROOT / "run.py"), run_name="__main__")

