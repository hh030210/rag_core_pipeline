"""Write paired gains/losses and diagnostic groups beside completed results."""
import json
from collections import Counter
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(Path(__file__).resolve().parent.parent / "output"):
        parser.error("Output must be inside this experiment folder")
    dataset = output if output.name == 'dataset' else output / 'dataset'
    manifest = json.loads((dataset / 'index.json').read_text(encoding='utf-8'))
    rows = [json.loads((dataset / relative).read_text(encoding='utf-8'))
            for relative in manifest['detail_files'].values()]
    from importlib import import_module
    summarize = import_module("rag_core.vector_retrieval.experiments.01_query_composition.code.run").summarize
    summary = {'strategy_metrics': summarize(rows)}
    strategies = list(summary['strategy_metrics'])
    paired, grouped = {}, {}
    for name in strategies:
        wins = losses = ties = 0
        groups = {}
        for row in rows:
            control = row["strategies"]["subqueries_concat"]["metrics"]
            metrics = row["strategies"][name]["metrics"]
            delta = int(metrics["hit_at_15"]) - int(control["hit_at_15"])
            wins += delta > 0
            losses += delta < 0
            ties += delta == 0
            labels = ["spot:" + (" / ".join(row["spot_names"]) or "unrestricted"),
                      "subquery_count:" + str(len(row["subqueries"]))]
            for label in labels:
                entry = groups.setdefault(label, {"count": 0, "control_hits15": 0, "hits15": 0})
                entry["count"] += 1
                entry["control_hits15"] += int(control["hit_at_15"])
                entry["hits15"] += int(metrics["hit_at_15"])
        paired[name] = {"rescued_at15": wins, "lost_at15": losses, "unchanged_at15": ties, "net": wins-losses}
        grouped[name] = groups
    diagnostic = {
        "paired_vs_subqueries_concat": paired, "groups": grouped,
        "subquery_source_counts": dict(Counter(r["subquery_source"] for r in rows)),
    }
    print(json.dumps(diagnostic, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
