"""SSH-independent supervisor for the explicitly authorized top5 Qwen experiment."""
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
DEFAULT_RUN = ROOT / "runs/noise_aware_top5_qwen_20261017"


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "run", "status"))
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--workers", type=int, default=80)
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    status_path, log_path = directory / "job_status.json", directory / "job.log"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    if args.action == "start":
        if status.get("status") in {"starting", "running"} and alive(status.get("supervisor_pid")):
            print(json.dumps(status, ensure_ascii=False))
            return
        if status.get("status") == "complete":
            print(json.dumps(status, ensure_ascii=False))
            return
        save(status_path, {"status": "starting", "started_at": now(), "workers": args.workers, "context_budget": 5})
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-m", "rag_core.top5_job", "run",
                "--run-dir", str(directory), "--workers", str(args.workers)], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"supervisor_pid": process.pid, "run_dir": str(directory), "log": str(log_path)}, ensure_ascii=False))
    elif args.action == "run":
        status.update({"status": "running", "supervisor_pid": os.getpid(), "started_at": now(),
                       "workers": args.workers, "context_budget": 5})
        save(status_path, status)
        environment = {**os.environ, "RUN_DIR": str(directory), "WORKERS": str(args.workers), "PYTHONUNBUFFERED": "1"}
        print(f"[{now()}] Starting complete top5/Qwen-only pipeline", flush=True)
        child = subprocess.Popen(["bash", str(ROOT / "run_top5_qwen_full.sh")], cwd=ROOT, env=environment)
        status["pipeline_pid"] = child.pid
        save(status_path, status)
        code = child.wait()
        summary_path = directory / "evaluation_qwen/summary.json"
        complete = code == 0 and summary_path.exists() and json.loads(summary_path.read_text(encoding="utf-8")).get("complete")
        status.update({"status": "complete" if complete else "failed", "exit_code": code,
                       "finished_at": now(), "summary_path": str(summary_path)})
        save(status_path, status)
        print(f"[{now()}] Pipeline {status['status']} exit_code={code}", flush=True)
    else:
        result = {**status, "supervisor_alive": alive(status.get("supervisor_pid"))}
        cases = list((directory / "cluster_prompting/case_iterations").glob("*.json"))
        completed_cases, attempts = 0, 0
        for path in cases:
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                completed_cases += row.get("stop_reason") not in {None, "in_progress"}
                attempts += row.get("iterations_attempted", 0)
            except (ValueError, OSError):
                pass
        result.update({"case_files": len(cases), "completed_cases": completed_cases, "total_case_rewrites": attempts})
        for label, path in (("inference_count", directory / "answers_top5.jsonl"),
                            ("final_scores_count", directory / "evaluation_qwen/question_scores.jsonl")):
            result[label] = sum(1 for line in path.open(encoding="utf-8") if line.strip()) if path.exists() else 0
        if log_path.exists():
            with log_path.open(encoding="utf-8", errors="replace") as handle:
                result["log_tail"] = list(deque(handle, maxlen=15))
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
