#!/usr/bin/env python3
"""Calculate the frozen-data experiment's three-route metrics."""
from importlib import import_module
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPOSITORY))
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    metrics = import_module(
        "rag_core.retrieval_fusion.experiments.00_main_experiment.code.calculate_metrics"
    )
    raise SystemExit(metrics.main(
        output_root=EXPERIMENT_DIR / "output" / "runs",
        expected_experiment=EXPERIMENT_DIR.name,
    ))
