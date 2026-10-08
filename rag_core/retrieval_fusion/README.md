# Retrieval fusion

Fusion retrieval code, experiment output, and its visualization assets are grouped here.

- `fusion_engine/fusion.py` implements weighted fusion.
- `fusion_engine/output/` stores one timestamped JSON snapshot per fusion call.
- `visualization/` contains the standalone retrieval comparison viewer and its bundled data.

The viewer prefers snapshots from `fusion_engine/output/` when any exist. Each retrieval call writes its query and candidate routes there. These snapshots have no gold labels; use `visualization/start.bat --run-dir <run>` to view evaluation results and hit rates instead.

Run the viewer from any working directory with:

```bat
rag_core\retrieval_fusion\visualization\start.bat
```

The viewer resolves default experiment data and relative input paths against the repository root.
