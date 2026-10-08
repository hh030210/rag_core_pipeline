# Retrieval fusion

Fusion retrieval code, experiment output, and its visualization assets are grouped here.

- `fusion_engine/fusion.py` implements weighted fusion.
- `fusion_engine/output/` stores one timestamped JSON snapshot per fusion call.
- `visualization/` contains the standalone retrieval comparison viewer.

Each retrieval call writes its query and candidate routes to `fusion_engine/output/`. These snapshots have no gold labels; Gold metrics are shown when `fusion_engine/output_new/comparison.jsonl` is available.

Open `visualization/index.html` directly. On Windows, `visualization/start.bat` opens the same static page. No Python server is needed for viewing. After updating snapshots, rebuild the bundled static data from the repository root with `py -3 rag_core/retrieval_fusion/visualization/build_static_data.py`.

