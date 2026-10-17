"""Resumable chapter3 mechanical metrics + Qwen-only 1-5 final judgement."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

from .llm import LLMClient

DIMENSIONS = ("answer_quality", "accuracy", "completeness")
MACHINE_METRICS = ("bleu-avg", "bleu-1", "bleu-2", "bleu-3", "bleu-4", "rouge-L", "precision", "recall", "f1")
PROTOCOL_DEFAULT = "/pub/yanglw/tourist_three_experiments/run_three_experiments.py"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_protocol(path):
    spec = importlib.util.spec_from_file_location("chapter3_original_evaluation_protocol", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ensure_metric_dependencies()
    return module


def evaluate(input_path, output_dir, *, client, protocol, protocol_path, workers=80):
    if workers < 1 or client.mock:
        raise ValueError("Final evaluation requires positive workers and a real Qwen client")
    rows = read_jsonl(input_path)
    if not rows or any(row.get("mock") or not row.get("generated_answer") for row in rows):
        raise ValueError("Empty, incomplete or mock answers cannot be scored as real results")
    ids = [str(row["id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate answer IDs")
    output_dir = Path(output_dir)
    config = {"version": 1, "input_sha256": file_hash(input_path), "protocol_sha256": file_hash(protocol_path),
              "judge_model": client.model, "judge_base_url": client.base_url,
              "judge_scale": "1-5 integers", "machine_aggregation": "per-question macro mean",
              "protocol_path": str(protocol_path), "workers": workers}
    config_path = output_dir / "evaluation_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Evaluation configuration changed; use a fresh output directory")
    save_json(config_path, config)
    result_path = output_dir / "question_scores.jsonl"
    finished = {}
    if result_path.exists():
        for row in read_jsonl(result_path):
            if row.get("status") == "ok":
                finished[str(row["id"])] = row
    # Warm metric packages before starting worker threads.
    protocol.score_answer("预热", "预热")

    def score_one(row):
        answer = protocol.parse_generated_answer(row["generated_answer"])
        payload = {**row, "generated_answer": answer}
        metrics = protocol.score_answer(answer, row.get("reference_answer", ""))
        error = None
        for attempt in range(3):
            try:
                raw = client.complete_json(protocol.JUDGE_SYSTEM_PROMPT if hasattr(protocol, "JUDGE_SYSTEM_PROMPT")
                    else "你是严格的问答质量评测器。只输出指定结构的合法JSON，不输出思考过程。",
                    protocol.build_llm_judge_prompt(payload), temperature=0.0, max_tokens=1600)
                scores = protocol.parse_llm_judge_scores(raw)
                return {"id": row["id"], "status": "ok", "query": row["query"],
                    "reference_answer": row.get("reference_answer", ""), "generated_answer": answer,
                    "context_budget": row.get("context_budget"), "machine_metrics": metrics,
                    "qwen_scores": scores, "judge_model": client.model}
            except Exception as exc:
                error = exc
                if attempt < 2:
                    time.sleep(1 + attempt)
        raise RuntimeError(str(error))

    errors = []
    with ThreadPoolExecutor(max_workers=workers) as executor, result_path.open("a", encoding="utf-8") as handle:
        futures = {executor.submit(score_one, row): row for row in rows if str(row["id"]) not in finished}
        for future in as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                errors.append({"id": row["id"], "error": str(exc)})
                print(f"evaluation failed id={row['id']}: {exc}", flush=True)
                continue
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            finished[str(row["id"])] = result
            print(f"final evaluation: {len(finished)}/{len(rows)}", flush=True)
    save_json(output_dir / "errors.json", errors)
    selected = [finished[case_id] for case_id in ids if case_id in finished]
    summary = {"expected_count": len(rows), "scored_count": len(selected), "complete": len(selected) == len(rows),
        "machine_metrics": {key: sum(row["machine_metrics"][key] for row in selected) / len(selected) if selected else None
                            for key in MACHINE_METRICS},
        "qwen_scores_1_to_5": {key: sum(row["qwen_scores"][key]["score"] for row in selected) / len(selected) if selected else None
                                for key in DIMENSIONS}, "judge_model": client.model}
    save_json(output_dir / "summary.json", summary)
    fields = ["id", *MACHINE_METRICS, *["qwen_" + key for key in DIMENSIONS]]
    with (output_dir / "question_scores.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow({"id": row["id"], **{key: row["machine_metrics"][key] for key in MACHINE_METRICS},
                             **{"qwen_" + key: row["qwen_scores"][key]["score"] for key in DIMENSIONS}})
    labels = ["BLEU-avg", "ROUGE-L", "Precision", "Recall", "F1", "Qwen AQ", "Qwen Accuracy", "Qwen Completeness"]
    values = [summary["machine_metrics"][key] for key in ("bleu-avg", "rouge-L", "precision", "recall", "f1")]
    values += [summary["qwen_scores_1_to_5"][key] for key in DIMENSIONS]
    table = "| " + " | ".join(labels) + " |\n| " + " | ".join(["---"] * len(labels)) + " |\n| "
    table += " | ".join("N/A" if value is None else f"{value:.6f}" for value in values) + " |\n"
    (output_dir / "summary.md").write_text(
        f"# Top5 + 群智提示 + Qwen3-8B\n\n完成 {len(selected)}/{len(rows)}；完整结果：{summary['complete']}\n\n" + table,
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if not summary["complete"]:
        raise RuntimeError("Final evaluation incomplete; rerun to resume missing IDs")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path(PROTOCOL_DEFAULT))
    parser.add_argument("--workers", type=int, default=80)
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "/pub/yanglw/models/Qwen/Qwen3-8B"))
    args = parser.parse_args()
    client = LLMClient(api_key=os.getenv("LLM_API_KEY", "EMPTY"), base_url=args.base_url, model=args.model)
    evaluate(args.input, args.output_dir, client=client, protocol=load_protocol(args.protocol),
             protocol_path=args.protocol, workers=args.workers)


if __name__ == "__main__":
    main()
