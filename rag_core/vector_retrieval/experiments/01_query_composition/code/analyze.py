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
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == summary["evaluated_count"]
    strategies = list(summary["strategy_metrics"])
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
    (output / "diagnostics.json").write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 配对对比分析", "",
        "所有差值都相对子查询竖线拼接对照组，使用相同的402条已映射查询（实际条数以 summary.json 为准）。", "",
        "| 方案 | Hit@15差值（百分点） | MRR@10差值 | nDCG@5差值 | 挽回问题数 | 丢失问题数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    control = summary["strategy_metrics"]["subqueries_concat"]
    for name, m in summary["strategy_metrics"].items():
        p = paired[name]
        lines.append(f"| {name} | {(m['hit@15']-control['hit@15'])*100:+.2f} | {m['mrr@10']-control['mrr@10']:+.4f} | {m['ndcg@5']-control['ndcg@5']:+.4f} | {p['rescued_at15']} | {p['lost_at15']} |")
    lines += ["", "按景区范围、子查询数量分组的计数见 diagnostics.json。分组只用于解释结果，没有参与策略选择或调整权重。",
              "", "RRF每个唯一查询分别检索，候选合并成本高于单向量方案；当前数据集上的优胜结果仍需独立测试集验证。"]
    lines[2] = f"所有差值都相对子查询竖线拼接对照组，使用相同的{len(rows)}条已映射查询。"
    (output / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(paired, ensure_ascii=False))

if __name__ == "__main__":
    main()

