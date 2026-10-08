"""Serve the viewer from live fusion snapshots; no build step or external data."""
from __future__ import annotations

import argparse
import functools
import json
import re
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

VIEWER_DIR = Path(__file__).resolve().parent
ENGINE_DIR = VIEWER_DIR.parent / 'fusion_engine'
SNAPSHOT_NAME = re.compile(r'^retrieval_fusion_[^/\\]+\.json$')


def comparisons(new_dir: Path) -> dict[str, dict]:
    path = new_dir / 'comparison.jsonl'
    if not path.is_file():
        return {}
    result = {}
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            name = row.get('source_snapshot')
            if isinstance(name, str) and SNAPSHOT_NAME.fullmatch(name):
                result[name] = row
    return result


def index_rows(old_dir: Path, new_dir: Path) -> list[dict]:
    by_name = comparisons(new_dir)
    rows = []
    for path in sorted(old_dir.glob('retrieval_fusion_*.json')):
        if not SNAPSHOT_NAME.fullmatch(path.name):
            continue
        comparison = by_name.get(path.name)
        query = comparison.get('query') if comparison else json.loads(path.read_text(encoding='utf-8')).get('query')
        has_new = (new_dir / path.name).is_file()
        rows.append({
            'name': path.name,
            'query': query or path.stem,
            'has_new': has_new,
            'gold_count': comparison.get('gold_count') if comparison else None,
            'semantic_rank': comparison.get('semantic_rank') if comparison else None,
            'dimension_rank': comparison.get('dimension_rank') if comparison else None,
            'old_rank': comparison.get('old_rank') if comparison else None,
            'new_rank': comparison.get('new_rank') if comparison and has_new else None,
        })
    return rows


def compact(items: list[dict]) -> list[dict]:
    fields = ('chunk_id', 'score', 'semantic_score', 'dimension_score',
              'normalized_semantic_score', 'normalized_dimension_score',
              'lexical_score', 'original_fusion_rank', 'original_fusion_score',
              'dimension_paths', 'matched_dimensions', 'matches')
    return [dict(rank=rank, **{key: item[key] for key in fields if key in item})
            for rank, item in enumerate(items, 1)]


def detail_row(name: str, old_dir: Path, new_dir: Path) -> dict:
    if not SNAPSHOT_NAME.fullmatch(name):
        raise ValueError('Invalid snapshot name')
    old_path = old_dir / name
    if not old_path.is_file():
        raise FileNotFoundError(name)
    old = json.loads(old_path.read_text(encoding='utf-8'))
    new_path = new_dir / name
    new = json.loads(new_path.read_text(encoding='utf-8')) if new_path.is_file() else None
    if new and new.get('query') != old.get('query'):
        raise ValueError(f'Old/new query mismatch: {name}')
    semantic = old.get('semantic_candidates', [])
    dimension = old.get('dimension_candidates', [])
    original = old.get('fusion_candidates') or old.get('fusion_results', [])
    updated = new.get('fusion_candidates', []) if new else []
    if new and {x['chunk_id'] for x in original} != {x['chunk_id'] for x in updated}:
        raise ValueError(f'Old/new candidate pools differ: {name}')
    chunks = {}
    for item in semantic + dimension + original + updated:
        cid = str(item.get('chunk_id') or '')
        if not cid or (cid in chunks and chunks[cid].get('text')):
            continue
        chunks[cid] = {
            'title': item.get('chunk_gen_title') or item.get('doc_title') or '',
            'source_file': item.get('source_file') or '',
            'text': item.get('chunk_text_full') or item.get('chunk_text') or '',
        }
    comparison = comparisons(new_dir).get(name, {})
    if comparison and comparison.get('query') != old.get('query'):
        raise ValueError(f'Comparison query mismatch: {name}')
    routes = {'semantic': compact(semantic), 'dimension': compact(dimension),
              'old': compact(original), 'new': compact(updated)}
    known_gold_ids = []
    for route, key in (('semantic', 'semantic_rank'), ('dimension', 'dimension_rank'),
                       ('old', 'old_rank'), ('new', 'new_rank')):
        rank = comparison.get(key)
        if isinstance(rank, int) and 1 <= rank <= len(routes[route]):
            known_gold_ids.append(routes[route][rank - 1]['chunk_id'])
    return {
        'name': name, 'query': old.get('query') or '',
        'created_at': old.get('created_at'),
        'query_analysis': old.get('query_analysis') or {},
        'semantic_top1_gap': new.get('semantic_top1_gap') if new else None,
        'effective_dimension_weight': new.get('effective_dimension_weight') if new else None,
        'gold': comparison, 'known_gold_ids': list(dict.fromkeys(known_gold_ids)),
        'routes': routes, 'chunks': chunks,
    }


class ViewerHandler(SimpleHTTPRequestHandler):
    old_dir: Path = ENGINE_DIR / 'output'
    new_dir: Path = ENGINE_DIR / 'output_new'

    def send_json(self, value: object, status: int = 200) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        url = urlsplit(self.path)
        if not url.path.startswith('/api/'):
            return super().do_GET()
        try:
            if url.path == '/api/questions':
                return self.send_json({'items': index_rows(self.old_dir, self.new_dir)})
            if url.path == '/api/question':
                names = parse_qs(url.query).get('name', [])
                if len(names) != 1:
                    return self.send_json({'error': 'One snapshot name is required'}, 400)
                return self.send_json(detail_row(names[0], self.old_dir, self.new_dir))
            self.send_json({'error': 'Unknown API route'}, 404)
        except FileNotFoundError as exc:
            self.send_json({'error': str(exc)}, 404)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.send_json({'error': str(exc)}, 500)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fusion-output', type=Path, default=ENGINE_DIR / 'output')
    parser.add_argument('--fusion-output-new', type=Path, default=ENGINE_DIR / 'output_new')
    parser.add_argument('--bind', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    ViewerHandler.old_dir = args.fusion_output.expanduser().resolve()
    ViewerHandler.new_dir = args.fusion_output_new.expanduser().resolve()
    if not ViewerHandler.old_dir.is_dir():
        parser.error(f'Missing original fusion output: {ViewerHandler.old_dir}')
    if not ViewerHandler.new_dir.is_dir():
        parser.error(f'Missing new fusion output: {ViewerHandler.new_dir}')
    handler = functools.partial(ViewerHandler, directory=str(VIEWER_DIR))
    server = None
    for port in ([0] if args.port == 0 else range(args.port, args.port + 30)):
        try:
            server = ThreadingHTTPServer((args.bind, port), handler)
            break
        except OSError:
            continue
    if server is None:
        parser.error('No available HTTP port')
    url = f'http://{args.bind}:{server.server_port}/index.html'
    print(f'Original snapshots: {ViewerHandler.old_dir}')
    print(f'New snapshots: {ViewerHandler.new_dir}')
    print(f'Viewer: {url}')
    print('Press Ctrl+C to stop.')
    if not args.no_browser:
        webbrowser.open(url, new=2)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
