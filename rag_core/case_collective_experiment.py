"""Golden-only case training and budgeted cached-retrieval inference (no re-retrieval)."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

from .cached_context import (RETRIEVAL_SOURCE, file_hash, load_records, prepare_data,
                             render_context, select_contexts, write_json)
from .cluster_prompting import ClusterPromptPipeline, ClusterPromptRouter
from .embedding import EmbeddingModel
from .llm import LLMClient
from .prompt_policy import validate_module
from .prompting import PromptOptimizer
from .settings import Settings

ROOT = Path(__file__).resolve().parent.parent


def make_client(settings):
    return LLMClient(api_key=settings.llm_api_key, base_url=settings.llm_base_url,
        model=settings.llm_model, openai_compat=True, interval=settings.llm_interval, mock=settings.mock)


def bind_config(path, config):
    path = Path(path)
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != config:
        raise ValueError(f"Configuration changed; use a new run directory instead of mixing outputs: {path}")
    write_json(path, config)


def run_inference(settings, data_path, *, budget=10, workers=80):
    rows = load_records(data_path)
    if workers < 1:
        raise ValueError("workers must be positive")
    if not rows:
        raise ValueError("Empty inference dataset")
    for row in rows:
        select_contexts(row, budget)
    artifact = settings.run_dir / "cluster_prompting.json"
    if not artifact.exists():
        raise FileNotFoundError("Train first: missing cluster_prompting.json")
    mock_dir = settings.run_dir / "MOCK_ONLY" if settings.mock else settings.run_dir
    mock_dir.mkdir(parents=True, exist_ok=True)
    config = {"version": 1, "dataset_sha256": file_hash(data_path), "cluster_sha256": file_hash(artifact),
        "budget": budget, "mock": settings.mock, "model": settings.llm_model,
        "base_url": settings.llm_base_url, "context": "cached_retrieved_only_no_golden",
        "temperature": 0.1, "max_tokens": 1800}
    output = mock_dir / f"answers_top{budget}.jsonl"
    bind_config(mock_dir / f"inference_top{budget}_config.json", config)
    cluster_artifact = json.loads(artifact.read_text(encoding="utf-8"))
    methods = [cluster.get("prompt_method", "") for cluster in cluster_artifact.get("clusters", [])]
    if not settings.mock and any(method != "llm_collective_synthesis" for method in methods):
        raise ValueError("Real inference must not use mock/fallback cluster prompts")
    embeddings = EmbeddingModel(settings.model_path, settings.embedding_device, settings.vector_dim, settings.mock)
    router = ClusterPromptRouter(run_dir=settings.run_dir, embeddings=embeddings, enabled=True, top_k=1)
    if not router.available:
        raise ValueError("Cluster artifact unavailable; refusing silent global-prompt fallback")
    # Route serially to avoid sharing the embedding model across worker threads.
    routes = {}
    for row in rows:
        choices = router.route(row["question"])
        if not choices:
            raise ValueError(f"No cluster route for {row['id']}")
        validate_module(choices[0]["prompt"])
        routes[str(row["id"])] = choices[0]
    finished = {}
    if output.exists():
        for record in load_records(output):
            finished[str(record["id"])] = record
    llm = make_client(settings)

    def answer_one(row):
        documents = select_contexts(row, budget)
        context = render_context(documents)
        route = routes[str(row["id"])]
        module = route["prompt"]
        user_prompt = PromptOptimizer._qa_user(row["question"], context, module)
        if settings.mock:
            answer = "MOCK_ONLY: not a real experiment answer"
        else:
            answer = llm.complete(module["system_prompt"], user_prompt, temperature=0.1, max_tokens=1800)
        return {"id": row["id"], "query": row["question"], "question": row["question"],
            "reference_answer": row.get("reference_answer", ""), "generated_answer": answer,
            "answer": answer, "context_budget": budget, "context_documents": documents,
            "context": context, "context_kind": "retrieved", "prompt_module": module,
            "full_user_prompt": user_prompt, "cluster_id": route["cluster_id"],
            "cluster_similarity": route["score"], "mock": settings.mock}

    errors = []
    with ThreadPoolExecutor(max_workers=workers) as executor, output.open("a", encoding="utf-8") as handle:
        pending = {executor.submit(answer_one, row): row for row in rows if str(row["id"]) not in finished}
        for future in as_completed(pending):
            row = pending[future]
            try:
                record = future.result()
            except Exception as exc:
                errors.append({"id": row["id"], "error": str(exc)})
                continue
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            finished[str(row["id"])] = record
            print(f"inference top{budget}: {len(finished)}/{len(rows)}", flush=True)
    write_json(mock_dir / f"inference_top{budget}_errors.json", errors)
    if errors:
        raise RuntimeError(f"{len(errors)} inference requests failed; rerun to resume missing IDs")
    return {"output": str(output), "count": len(finished), "budget": budget, "mock": settings.mock}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("stage", choices=("prepare", "train", "infer", "all"))
    result.add_argument("--run-dir", type=Path, default=ROOT / "runs/noise_aware_v1")
    result.add_argument("--train-source", type=Path, default=ROOT / "inputs/iteration_questions.json")
    result.add_argument("--eval-source", type=Path, default=ROOT / "inputs/eval_questions.json")
    result.add_argument("--retrieval-source", type=Path, default=ROOT / RETRIEVAL_SOURCE)
    result.add_argument("--min-retrieval-depth", type=int, default=10,
                        help="10: original test questions with full top10; 15: earlier 347-question cohort")
    result.add_argument("--context-budget", type=int, choices=range(1, 11), default=10)
    result.add_argument("--max-rounds", type=int, default=30)
    result.add_argument("--min-gain", type=float, default=1.0)
    result.add_argument("--clusters", type=int, default=6)
    result.add_argument("--workers", type=int, default=80)
    result.add_argument("--model-path", default=os.getenv("BGE_MODEL_PATH", "/pub/yanglw/models/bge-large-zh-v1.5"))
    result.add_argument("--embedding-device", default="cpu")
    result.add_argument("--llm-base-url", default=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    result.add_argument("--llm-model", default=os.getenv("LLM_MODEL", "/pub/yanglw/models/Qwen/Qwen3-8B"))
    result.add_argument("--mock", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.mock and "mock" not in str(args.run_dir).casefold():
        raise ValueError("--mock requires a separate run directory containing 'mock'")
    args.run_dir = args.run_dir.resolve()
    settings = Settings.from_env(args.run_dir, model_path=args.model_path, embedding_device=args.embedding_device,
        llm_base_url=args.llm_base_url, llm_model=args.llm_model,
        llm_api_key=os.getenv("LLM_API_KEY", "EMPTY"), mock=args.mock,
        prompt_max_rounds=args.max_rounds, prompt_min_gain=args.min_gain)
    data_dir = args.run_dir / "prepared"
    if args.stage in {"prepare", "all"}:
        preparation_config = {"training_sha256": file_hash(args.train_source),
            "evaluation_sha256": file_hash(args.eval_source), "retrieval_sha256": file_hash(args.retrieval_source),
            "min_retrieval_depth": args.min_retrieval_depth}
        bind_config(data_dir / "preparation_config.json", preparation_config)
        manifest = prepare_data(args.train_source, args.eval_source, args.retrieval_source,
                                data_dir, min_retrieval_depth=args.min_retrieval_depth)
        print(json.dumps({key: manifest[key] for key in ("training_count", "evaluation_count", "question_overlap_count")}, ensure_ascii=False))
    if args.stage in {"train", "all"}:
        manifest = json.loads((data_dir / "data_manifest.json").read_text(encoding="utf-8"))
        if manifest["question_overlap_count"]:
            raise ValueError("Training/evaluation leakage detected")
        train_path = data_dir / "train_golden.json"
        train_config = {"training_sha256": file_hash(train_path), "max_rounds": args.max_rounds,
            "min_gain": args.min_gain, "clusters": args.clusters, "embedding_model": args.model_path,
            "llm_base_url": settings.llm_base_url, "llm_model": settings.llm_model, "mock": args.mock}
        bind_config(args.run_dir / "training_config.json", train_config)
        embeddings = EmbeddingModel(args.model_path, args.embedding_device, settings.vector_dim, args.mock)
        result = ClusterPromptPipeline(client=make_client(settings), embeddings=embeddings, run_dir=args.run_dir).run(
            load_records(train_path), cluster_count=args.clusters, iterations=args.max_rounds,
            min_gain=args.min_gain, workers=args.workers)
        print(json.dumps(result["metrics"], ensure_ascii=False))
    if args.stage in {"infer", "all"}:
        # Check persisted training parameters/model; never route with a different encoder.
        train_config = json.loads((args.run_dir / "training_config.json").read_text(encoding="utf-8"))
        if train_config["embedding_model"] != args.model_path or train_config["mock"] != args.mock:
            raise ValueError("Inference embedding model/mock mode does not match training")
        print(json.dumps(run_inference(settings, data_dir / "inference_top10.json",
                                      budget=args.context_budget, workers=args.workers), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
