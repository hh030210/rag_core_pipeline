"""Paired with/without evidence-policy runs on the new scenic fusion top5."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from .cached_context import file_hash, load_records, question_key, render_context, write_json
from .case_collective_experiment import bind_config
from .cluster_prompting import ClusterPromptPipeline, ClusterPromptRouter, normalize_examples
from .embedding import EmbeddingModel
from .llm import LLMClient
from .prompt_policy import validate_module
from .prompting import PromptOptimizer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN = ROOT / "runs/scenic_new_top5_ablation_20261017"
VARIANTS = (("with_policy", True), ("without_policy", False))
META = ROOT / "rag_core/prompts/ablation_initial_meta.txt"


def prepare_shared(source, train_source, eval_source, output):
    source, train_source, eval_source, output = map(Path, (source, train_source, eval_source, output))
    source_rows, raw_train, raw_eval = map(load_records, (source, train_source, eval_source))
    training = normalize_examples(raw_train)
    train_keys = {question_key(row["question"]) for row in training}
    heldout = {}
    for row in raw_eval:
        key = question_key(row["question"])
        if key in heldout:
            raise ValueError("Duplicate heldout question")
        heldout[key] = row
    if train_keys & set(heldout):
        raise ValueError("Original training/heldout questions overlap")
    cases, excluded, seen = [], [], set()
    for row in source_rows:
        question = str(row.get("question", "")).strip()
        key = question_key(question)
        reason = ""
        if key in train_keys:
            reason = "training_question_not_allowed_in_test"
        elif key not in heldout:
            reason = "no_heldout_reference_answer"
        elif key in seen:
            reason = "duplicate_retrieval_question"
        hits = row.get("routes", {}).get("fusion", [])
        if not reason and (len(hits) < 5 or any(not str(hit.get("chunk_text", "")).strip() for hit in hits[:5])):
            reason = "fewer_than_five_nonempty_fusion_contexts"
        if reason:
            excluded.append({"query_id": row.get("query_id"), "question": question, "reason": reason})
            continue
        label = heldout[key]
        reference = str(label.get("reference_answer", label.get("answer", ""))).strip()
        if not reference:
            raise ValueError("Matched heldout question lacks reference answer")
        documents = [{"position": index, "kind": "retrieved", "original_rank": hit.get("rank", index),
            "chunk_id": str(hit.get("chunk_id", "")), "doc_title": str(hit.get("doc_title", "")),
            "text": str(hit["chunk_text"]).strip()} for index, hit in enumerate(hits[:10], 1)]
        cases.append({"id": str(label.get("question_id", label.get("id"))), "question": label["question"],
                      "retrieval_query_id": str(row.get("query_id", "")), "reference_answer": reference,
                      "retrieved_contexts": documents})
        seen.add(key)
    if not cases:
        raise ValueError("No matched independent top5 test cases")
    for row in training:
        if row["context_kind"] != "golden" or not row["reference_answer"]:
            raise ValueError("Every independent training case needs golden evidence and reference answer")
        row["source_id"] = row["id"]
        row["id"] = "train_" + hashlib.sha256(question_key(row["question"]).encode()).hexdigest()[:20]
    config = {"retrieval_sha256": file_hash(source), "train_sha256": file_hash(train_source),
              "eval_sha256": file_hash(eval_source), "context_budget": 5, "route": "fusion",
              "version": "scenic_after", "matching": "NFKC_alphanumeric_casefold"}
    bind_config(output / "preparation_config.json", config)
    manifest = {**config, "source_question_count": len(source_rows), "training_count": len(training),
                "evaluation_count": len(cases), "training_test_overlap": 0, "excluded": excluded,
                "source_path": str(source), "source_server": "129.211.170.212",
                "source_directory": "/home/humq/rag_core_experiments/all_datasets_top10_retrieval_results/three_route_chunk_details"}
    write_json(output / "train_golden.json", {"examples": training})
    write_json(output / "test_top5.json", cases)
    write_json(output / "data_manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("source_question_count", "training_count", "evaluation_count", "training_test_overlap")}), flush=True)
    return training, cases, manifest


class CachedEmbeddings:
    def __init__(self, texts, vectors):
        self.lookup = dict(zip(texts, vectors))

    def encode(self, texts):
        return [self.lookup[str(text)] for text in texts]


def shared_embeddings(training, cases, directory, model_path, mock=False):
    texts = [f"{row['question']}\n{row['reference_answer']}".strip() for row in training]
    texts += [row["question"] for row in cases]
    texts = list(dict.fromkeys(texts))
    config = {"texts_sha256": hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest(),
              "model_path": model_path, "mock": mock, "pooling": "normalized_CLS", "max_length": 512}
    bind_config(directory / "embedding_config.json", config)
    path = directory / "vectors.npy"
    if path.exists():
        vectors = np.load(path)
        if len(vectors) != len(texts):
            raise ValueError("Embedding cache does not cover the exact shared data")
    else:
        print(f"Encoding shared QA/question embeddings: {len(texts)} texts", flush=True)
        model = EmbeddingModel(model_path, "cpu", 1024, mock=mock)
        vectors = np.asarray(model.encode(texts), dtype=np.float32)
        np.save(path, vectors)
        write_json(directory / "embedding_texts.json", texts)
        print("Shared embeddings ready for both variants", flush=True)
    return CachedEmbeddings(texts, vectors.tolist())


def inference(cases, embeddings, client, directory, evidence_policy, workers=80, seed=20261017):
    artifact_path = directory / "cluster_prompting.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["config"]["evidence_policy"] == evidence_policy
    router = ClusterPromptRouter(run_dir=directory, embeddings=embeddings, enabled=True, top_k=1,
                                 evidence_policy=evidence_policy)
    routes = {str(row["id"]): router.route(row["question"])[0] for row in cases}
    bind_config(directory / "inference_config.json", {"policy": evidence_policy, "seed": seed,
        "dataset_sha256": hashlib.sha256(json.dumps(cases, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        "artifact_sha256": file_hash(artifact_path), "budget": 5, "model": client.model,
        "base_url": client.base_url, "temperature": 0.1, "max_tokens": 1800})
    path = directory / "answers_top5.jsonl"
    finished = {str(row["id"]): row for row in load_records(path)} if path.exists() else {}

    def answer_one(row):
        documents = row["retrieved_contexts"][:5]
        assert len(documents) == 5 and all(doc["kind"] == "retrieved" for doc in documents)
        context = render_context(documents)
        route = routes[str(row["id"])]
        module = validate_module(route["prompt"], evidence_policy=evidence_policy)
        user = PromptOptimizer._qa_user(row["question"], context, module)
        answer = client.complete(module["system_prompt"], user, temperature=0.1, max_tokens=1800)
        return {"id": row["id"], "query": row["question"], "question": row["question"],
            "reference_answer": row["reference_answer"], "generated_answer": answer, "answer": answer,
            "context_budget": 5, "context_documents": documents, "context": context,
            "full_user_prompt": user, "prompt_module": module, "cluster_id": route["cluster_id"],
            "cluster_similarity": route["score"], "evidence_policy": evidence_policy,
            "mock": False, "seed": seed}

    errors = []
    with ThreadPoolExecutor(max_workers=workers) as executor, path.open("a", encoding="utf-8") as handle:
        futures = {executor.submit(answer_one, row): row for row in cases if str(row["id"]) not in finished}
        for future in as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                errors.append({"id": row["id"], "error": str(exc)})
                print(f"inference error policy={evidence_policy} id={row['id']}: {exc}", flush=True)
                continue
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            finished[str(row["id"])] = result
            print(f"{directory.name} top5 inference: {len(finished)}/{len(cases)}", flush=True)
    write_json(directory / "inference_errors.json", errors)
    if errors or len(finished) != len(cases):
        raise RuntimeError("Inference incomplete; resume missing question IDs before evaluation")


def report(directory):
    rows = []
    for name, policy in VARIANTS:
        summary = json.loads((directory / name / "evaluation_qwen/summary.json").read_text(encoding="utf-8"))
        if not summary["complete"]:
            raise ValueError("Cannot compare incomplete evaluations")
        rows.append({"setting": name, "count": summary["scored_count"], **summary["machine_metrics"],
                     **{"qwen_" + key: value for key, value in summary["qwen_scores_1_to_5"].items()}})
    if rows[0]["count"] != rows[1]["count"]:
        raise ValueError("Variant question coverage differs")
    write_json(directory / "comparison.json", rows)
    fields = ["setting", "count", "bleu-avg", "rouge-L", "precision", "recall", "f1",
              "qwen_answer_quality", "qwen_accuracy", "qwen_completeness"]
    with (directory / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 新景区融合 Top5：双组对照", "", "两组使用完全相同的测试题、参考答案与上下文；Qwen 评分为 1–5 分。", "",
             "| 设置 | 题数 | BLEU-avg | ROUGE-L | Precision | Recall | F1 | Qwen AQ | Accuracy | Completeness |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        label = "含噪音与重排序说明" if row["setting"] == "with_policy" else "不含两条说明"
        lines.append("| " + " | ".join([label, str(row["count"])] + [f"{row[key]:.6f}" for key in fields[2:]]) + " |")
    (directory / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--source", type=Path, default=ROOT / "inputs/scenic_after_top10_chunks.jsonl.gz")
    parser.add_argument("--workers", type=int, default=80)
    parser.add_argument("--model-path", default="/pub/yanglw/models/bge-large-zh-v1.5")
    parser.add_argument("--llm-model", default="/pub/yanglw/models/Qwen/Qwen3-8B")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--max-rounds", type=int, default=30)
    parser.add_argument("--min-gain", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--variant", choices=("all", "with_policy", "without_policy"), default="all")
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    training, cases, _ = prepare_shared(args.source, ROOT / "inputs/iteration_questions.json",
                                        ROOT / "inputs/eval_questions.json", directory / "shared")
    embeddings = shared_embeddings(training, cases, directory / "shared", args.model_path)
    for name, policy in VARIANTS:
        if args.variant != "all" and name != args.variant:
            continue
        target = directory / name
        target.mkdir(parents=True, exist_ok=True)
        print(f"VARIANT_START {name}", flush=True)
        client = LLMClient(api_key=os.getenv("LLM_API_KEY", "EMPTY"), base_url=args.base_url,
                           model=args.llm_model, seed=args.seed)
        bind_config(target / "training_config.json", {"evidence_policy": policy, "max_rounds": args.max_rounds,
            "min_gain": args.min_gain, "seed": args.seed, "model": args.llm_model, "base_url": args.base_url,
            "meta_sha256": file_hash(META), "data_sha256": file_hash(directory / "shared/train_golden.json")})
        artifact_path = target / "cluster_prompting.json"
        if artifact_path.exists():
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            assert artifact["config"]["evidence_policy"] == policy
            for cluster in artifact["clusters"]:
                validate_module(cluster["prompt"], evidence_policy=policy)
        else:
            ClusterPromptPipeline(client=client, embeddings=embeddings, run_dir=target,
                evidence_policy=policy, meta_prompt_path=META).run(training, cluster_count=6,
                iterations=args.max_rounds, min_gain=args.min_gain, workers=args.workers)
        inference(cases, embeddings, client, target, policy, args.workers, args.seed)
        command = ["/pub/yanglw/miniconda3/bin/python", "-m", "rag_core.final_qwen_evaluation",
            "--input", str(target / "answers_top5.jsonl"), "--output-dir", str(target / "evaluation_qwen"),
            "--workers", str(args.workers), "--model", args.llm_model, "--base-url", args.base_url]
        subprocess.run(command, cwd=ROOT, check=True)
        print(f"VARIANT_COMPLETE {name}", flush=True)
    if all((directory / name / "evaluation_qwen/summary.json").exists() for name, _ in VARIANTS):
        report(directory)


if __name__ == "__main__":
    main()
