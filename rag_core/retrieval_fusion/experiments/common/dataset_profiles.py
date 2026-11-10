"""Dataset and output defaults for the three-route main experiment."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class DatasetProfile:
    title: str
    run_dir: Path | None = None
    dataset: Path | None = None
    collection: str | None = None
    parser_mode: str = "deterministic"


HUANGCHENG = Path("/home/humq/rag_core_runs/huangcheng_20261019")
DURETRIEVAL = REPOSITORY / "result/duretrieval_mteb_dev_deepseek_v4pro_20260923_run9"
PROFILES = {
    "scenic": DatasetProfile("景区数据集", parser_mode="llm"),
    "huangcheng": DatasetProfile(
        "皇城相府高精度扩展集",
        HUANGCHENG / "pipeline_run1",
        HUANGCHENG / "eval_source/gold_slot_audit_v1/retrieval_gold_slot_extended_auto_eval_v1.jsonl",
        "rag_core_huangcheng_20261019_v1",
    ),
    "duretrieval_mteb": DatasetProfile(
        "DuRetrieval MTEB dev",
        DURETRIEVAL,
        DURETRIEVAL / "evaluation_freshschema_noexpansion_20260927/qa_dataset_gold_ids_only.json",
        "duretrieval_mteb_dev_qrels20k_dim_fresh_20260923",
    ),
}
MODEL_PATH = "/home/humq/rag_db_silm/model/bge-m3"


def profile_output_root(root: Path, profile: str) -> Path:
    return root if profile == "scenic" else root.parent / profile / root.name
