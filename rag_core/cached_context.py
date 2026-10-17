"""Strictly separated golden training data and cached top-10 inference evidence."""
import gzip
import hashlib
import json
from pathlib import Path
import unicodedata

from .cluster_prompting import normalize_examples

RETRIEVAL_SOURCE = "result/real_merged7_deepseek_v4pro_rerank_20260920_run1/evaluation_noexpansion/results_top15_full_text.jsonl.gz"


def question_key(value):
    return "".join(c for c in unicodedata.normalize("NFKC", str(value)).casefold() if c.isalnum())


def load_records(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        if path.suffix in {".jsonl", ".gz"}:
            return [json.loads(line) for line in handle if line.strip()]
        data = json.load(handle)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("examples", "questions", "data", "results"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Unsupported data format: {path}")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def prepare_data(train_path, eval_path, retrieval_path, output_dir, *, min_retrieval_depth=10):
    if min_retrieval_depth < 10:
        raise ValueError("Cohort must have at least ten cached contexts")
    raw_train = load_records(train_path)
    raw_eval = load_records(eval_path)
    retrieval_rows = load_records(retrieval_path)
    by_question = {}
    for row in retrieval_rows:
        by_question.setdefault(question_key(row.get("question", row.get("query", ""))), []).append(row)
    evaluation, excluded_eval = [], []
    seen = set()
    for index, raw in enumerate(raw_eval):
        question = str(raw.get("question", raw.get("query", ""))).strip()
        key = question_key(question)
        matches = by_question.get(key, [])
        case_id = str(raw.get("question_id", raw.get("id", f"eval_{index:04d}")))
        reason = ""
        if not key or key in seen:
            reason = "empty_or_duplicate_question"
        elif len(matches) != 1:
            reason = "missing_or_ambiguous_retrieval_match"
        else:
            row = matches[0]
            hits = row.get("retrieval", {}).get("fusion_results", [])
            if len(hits) < min_retrieval_depth:
                reason = "insufficient_retrieval_depth"
            else:
                documents = []
                for rank, hit in enumerate(hits[:10], 1):
                    text = str(hit.get("chunk_text_full") or hit.get("chunk_text") or "").strip()
                    if not text or hit.get("text_truncated") is True:
                        reason = "empty_or_truncated_context"
                        break
                    # Do not propagate oracle fields such as is_gold into inference.
                    documents.append({"position": rank, "kind": "retrieved", "chunk_id": hit.get("chunk_id", ""),
                                      "original_rank": hit.get("rank", rank), "doc_title": hit.get("doc_title", ""),
                                      "text": text})
                if not reason:
                    evaluation.append({"id": case_id, "question": question,
                        "reference_answer": str(raw.get("reference_answer", raw.get("answer", ""))),
                        "retrieval_record_id": str(row.get("id", "")), "retrieved_top10": documents})
                    seen.add(key)
        if reason:
            excluded_eval.append({"id": case_id, "question": question, "reason": reason})
    if not evaluation:
        raise ValueError("No evaluation questions have complete top-10 contexts")
    # Exclude overlap with ALL designated test questions, including unavailable
    # retrieval rows, rather than just this run's selected subset.
    eval_keys = {question_key(row.get("question", row.get("query", ""))) for row in raw_eval}
    training, excluded_train, seen_train = [], [], set()
    for row in normalize_examples(raw_train):
        key = question_key(row["question"])
        reason = ("overlaps_evaluation" if key in eval_keys else
                  "duplicate_training_question" if key in seen_train else
                  "missing_golden_context" if row["context_kind"] != "golden" else
                  "missing_reference_answer" if not row["reference_answer"] else "")
        if reason:
            excluded_train.append({"id": row["id"], "question": row["question"], "reason": reason})
            continue
        # Avoid source-provided IDs escaping the output directory.
        row["source_id"] = row["id"]
        row["id"] = "train_" + hashlib.sha256(key.encode()).hexdigest()[:20]
        training.append(row)
        seen_train.add(key)
    if not training:
        raise ValueError("No independent golden training cases remain after overlap removal")
    manifest = {"version": 1, "training_count": len(training), "evaluation_count": len(evaluation),
        "raw_training_count": len(raw_train), "raw_evaluation_count": len(raw_eval),
        "retrieval_record_count": len(retrieval_rows), "min_retrieval_depth": min_retrieval_depth,
        "cached_context_count": 10, "training_context": "golden", "inference_context": "retrieved_only",
        "question_overlap_count": len(seen_train & eval_keys),
        "excluded_training": excluded_train, "excluded_evaluation": excluded_eval,
        "sources": {label: {"path": str(Path(path).resolve()), "sha256": file_hash(path)}
                    for label, path in (("training", train_path), ("evaluation", eval_path), ("retrieval", retrieval_path))}}
    output = Path(output_dir)
    write_json(output / "train_golden.json", {"examples": training})
    write_json(output / "inference_top10.json", evaluation)
    write_json(output / "data_manifest.json", manifest)
    return manifest


def select_contexts(case, budget=10):
    if not 1 <= int(budget) <= 10:
        raise ValueError("context budget must be an integer from 1 to 10")
    documents = case.get("retrieved_top10", [])
    if len(documents) != 10:
        raise ValueError("Inference record must contain exactly ten cached documents")
    selected = documents[:int(budget)]
    if any(doc.get("kind") != "retrieved" or not doc.get("text") for doc in selected):
        raise ValueError("Only non-empty retrieved documents may enter inference")
    return selected


def render_context(documents):
    return "\n\n".join(f"[来源 {index}] {doc.get('doc_title', '')} ({doc.get('chunk_id', '')})：\n{doc['text']}"
                          for index, doc in enumerate(documents, 1))
