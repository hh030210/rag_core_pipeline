"""Score archived answers only. Isolated, bounded and resumable; never generate QA."""
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

DIMS = ('factual_accuracy', 'context_faithfulness', 'question_completeness',
        'answer_redundancy', 'reference_similarity')


def sha(value):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(value.encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def parse_grade(text):
    clean = text.strip()
    if clean.startswith('```json\n') and clean.endswith('\n```'):
        clean = clean[8:-4].strip()
    elif clean.startswith('```\n') and clean.endswith('\n```'):
        clean = clean[4:-4].strip()
    grade = json.loads(clean)
    assert isinstance(grade, dict) and set(grade) == set(DIMS), 'Wrong dimensions'
    for key, item in grade.items():
        assert isinstance(item, dict) and set(item) == {'score', 'reasoning'}, key
        score = item['score']
        assert score is None or (not isinstance(score, bool) and isinstance(score, (float, int))
                                 and math.isfinite(score) and 0 <= score <= 100), key
        assert isinstance(item['reasoning'], str) and item['reasoning'].strip(), key
    return grade


def request_for(task, rubric):
    assert set(task['input']) == {'question', 'reference_answer', 'candidate_answer', 'contexts'}
    return {'model': 'DeepSeek-V4-Pro', 'messages': [
        {'role': 'system', 'content': rubric},
        {'role': 'user', 'content': json.dumps(task['input'], ensure_ascii=False)}],
        'temperature': 0.1, 'seed': 20261026, 'max_tokens': 5000, 'stream': False}


def main():
    root = Path(sys.argv[1]); manifest = read(root / 'manifest.json')
    rubric = (root / 'rubric.txt').read_text(encoding='utf-8')
    assert sha(rubric) == manifest['rubric_sha256']
    tasks = read(root / 'tasks.json')
    assert sha(tasks) == manifest['tasks_sha256'] and len(tasks) == 40
    for directory in ('records', 'raw'):
        (root / directory).mkdir(exist_ok=True)
    lock = threading.Lock()
    counts = {'complete': 0, 'failed': 0}

    def one(task):
        key = task['key']; path = root / 'records' / (key + '.json')
        payload = request_for(task, rubric); request_hash = sha(payload)
        if path.exists():
            record = read(path)
            assert record['request_sha256'] == request_hash
            return record['status']
        errors = []
        # At most two transport/schema attempts; never resample a valid low score.
        for attempt in range(1, 3):
            raw_path = root / 'raw' / (key + '_' + str(attempt) + '.json')
            if raw_path.exists():
                raw = read(raw_path); assert raw['request_sha256'] == request_hash
            else:
                raw = {'request': payload, 'request_sha256': request_hash,
                       'started_at': time.time(), 'attempt': attempt}
                try:
                    req = urllib.request.Request('http://127.0.0.1:8910/v1/chat/completions',
                        data=json.dumps(payload, ensure_ascii=False).encode(),
                        headers={'Content-Type': 'application/json', 'Authorization': 'Bearer EMPTY'})
                    with urllib.request.urlopen(req, timeout=300) as response:
                        body = response.read().decode('utf-8')
                        raw.update(http_status=response.status, response_body=body)
                    raw['response'] = json.loads(body)
                except Exception as exc:
                    raw['error'] = type(exc).__name__ + ': ' + str(exc)
                raw['seconds'] = time.time() - raw['started_at']
                save(raw_path, raw)
            try:
                assert not raw.get('error'), raw.get('error')
                choice = raw['response']['choices'][0]
                assert choice.get('finish_reason') != 'length', 'Truncated response'
                grade = parse_grade(choice['message']['content'])
                result = {'key': key, 'status': 'complete', 'evaluation': grade,
                          'request_sha256': request_hash, 'raw_record': str(raw_path),
                          'rubric_sha256': manifest['rubric_sha256'], 'effective_repetitions': 1}
                save(path, result)
                return 'complete'
            except Exception as exc:
                errors.append({'attempt': attempt, 'raw_record': str(raw_path),
                               'error': type(exc).__name__ + ': ' + str(exc)})
        save(path, {'key': key, 'status': 'failed', 'evaluation': None,
                    'request_sha256': request_hash, 'errors': errors})
        return 'failed'

    save(root / 'status.json', {'stage': 'running', 'total': len(tasks), **counts})
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(one, task) for task in tasks]
        for future in as_completed(futures):
            state = future.result()
            with lock:
                counts[state] += 1
                save(root / 'status.json', {'stage': 'running', 'total': len(tasks),
                     **counts, 'updated_at': time.time()})
                print(json.dumps(counts), flush=True)
    save(root / 'status.json', {'stage': 'complete', 'total': len(tasks),
         **counts, 'updated_at': time.time(), 'answer_generation_calls': 0})


if __name__ == '__main__':
    main()
