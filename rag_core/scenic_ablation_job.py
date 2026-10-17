"""Detached supervisor and bounded status for the new-context paired experiment."""
import argparse
from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN = ROOT / "runs/scenic_new_top5_ablation_20261017"
NAMES = ("with_policy", "without_policy")


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, TypeError, ValueError):
        return False


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def count_lines(path):
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "run", "status"))
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--workers", type=int, default=80)
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    status_path, log_path = directory / "job_status.json", directory / "job.log"
    status = read(status_path) if status_path.exists() else {}
    if args.action == "start":
        if status.get("status") in {"starting", "running"} and alive(status.get("supervisor_pid")):
            print(json.dumps(status, ensure_ascii=False))
            return
        if status.get("status") == "complete":
            print(json.dumps(status, ensure_ascii=False))
            return
        save(status_path, {"status": "starting", "started_at": now(), "workers": args.workers, "context_budget": 5})
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-m", "rag_core.scenic_ablation_job", "run",
                "--run-dir", str(directory), "--workers", str(args.workers)], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"supervisor_pid": process.pid, "run_dir": str(directory), "log": str(log_path)}, ensure_ascii=False))
    elif args.action == "run":
        status.update({"status": "running", "supervisor_pid": os.getpid(), "started_at": now(),
                       "workers": args.workers, "context_budget": 5})
        save(status_path, status)
        print(f"[{now()}] New scenic context paired experiment started", flush=True)
        child = subprocess.Popen(["bash", str(ROOT / "run_scenic_ablation.sh"), "--run-dir", str(directory),
                                  "--workers", str(args.workers)], cwd=ROOT)
        status["pipeline_pid"] = child.pid
        save(status_path, status)
        code = child.wait()
        complete = code == 0 and (directory / "comparison.json").exists()
        complete = complete and all((directory / name / "evaluation_qwen/summary.json").exists() and
            read(directory / name / "evaluation_qwen/summary.json").get("complete") for name in NAMES)
        status.update({"status": "complete" if complete else "failed", "exit_code": code,
                       "finished_at": now(), "comparison_path": str(directory / "comparison.json")})
        save(status_path, status)
        print(f"[{now()}] Paired experiment {status['status']} exit_code={code}", flush=True)
    else:
        result = {**status, "supervisor_alive": alive(status.get("supervisor_pid")), "variants": {}}
        manifest_path = directory / "shared/data_manifest.json"
        if manifest_path.exists():
            manifest = read(manifest_path)
            result["test_question_count"] = manifest["evaluation_count"]
        for name in NAMES:
            folder = directory / name
            cases = list((folder / "cluster_prompting/case_iterations").glob("*.json"))
            completed = 0
            for path in cases:
                try:
                    completed += read(path).get("stop_reason") not in {None, "in_progress"}
                except (OSError, ValueError):
                    pass
            result["variants"][name] = {"case_files": len(cases), "completed_cases": completed,
                "collective_prompt_count": len(list((folder / "cluster_prompting/cluster_prompts").glob("cluster_*.json"))),
                "inference_count": count_lines(folder / "answers_top5.jsonl"),
                "final_scores_count": count_lines(folder / "evaluation_qwen/question_scores.jsonl")}
        if log_path.exists():
            with log_path.open(encoding="utf-8", errors="replace") as handle:
                result["log_tail"] = list(deque(handle, maxlen=14))
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
