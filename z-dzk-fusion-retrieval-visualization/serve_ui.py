"""Refresh the viewer data, then open the local retrieval comparison page."""

from __future__ import annotations

import argparse
import functools
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from build_data import VIEWER_DIR, add_source_arguments, build_viewer_data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_source_arguments(parser)
    args = parser.parse_args()

    result = build_viewer_data(
        args.run_dir,
        args.evaluation,
        args.chunks,
        args.rerank_results,
        args.tags,
        args.dimension_metadata,
    )
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(VIEWER_DIR))

    server = None
    for port in range(8765, 8796):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), handler)
            break
        except OSError:
            continue
    if server is None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)

    url = f"http://127.0.0.1:{server.server_port}/index.html"
    print(f"Updated viewer data from: {result['evaluation']}")
    print(f"Chunk source: {result['chunks']}")
    print(f"Chunk dimension tags: {result['tags'] or 'not found'}")
    print(f"Questions: {result['question_count']}; candidate chunks: {result['candidate_chunk_count']}")
    print("Available routes: " + ", ".join(result["available_routes"]))
    if "deepseek_results" not in result["available_routes"] or "answerability_results" not in result["available_routes"]:
        print("Rerank result files were not supplied; those routes are omitted from this view.")
    print(f"Retrieval viewer: {url}")
    print("Press Ctrl+C in this window to stop the server.")
    webbrowser.open(url, new=2)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
