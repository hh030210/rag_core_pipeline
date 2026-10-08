# Retrieval fusion

Fusion retrieval code, experiment output, and its visualization assets are grouped here.

- `fusion_engine/fusion.py` implements weighted fusion.
- `fusion_engine/output/` stores one timestamped JSON snapshot per fusion call.
- `visualization/` contains the standalone retrieval comparison viewer and its bundled data.

Run the viewer from any working directory with:

```bat
rag_core\retrieval_fusion\visualization\start.bat
```

The viewer resolves default experiment data and relative input paths against the repository root.
