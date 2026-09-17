import importlib.util
import pathlib
from typing import Dict, List


def load_final_module(script_dir: pathlib.Path):
    module_path = script_dir / "generate_final_comparison_tables.py"
    spec = importlib.util.spec_from_file_location("final_tables", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("inf")


def format_rank(value: float, bold: bool) -> str:
    cell = f"{value:.2f}"
    return f"\\textbf{{{cell}}}" if bold else cell


def main() -> int:
    script_dir = pathlib.Path(__file__).resolve().parent
    g = load_final_module(script_dir)

    tables = g.empty_tables()
    g.parse_paper_table(script_dir / "results/other_methods/RAwR_paper_results.tex", tables)
    g.parse_other_methods_tex(script_dir / "results/other_methods/node_classification_results_table_no_features.tex", tables)
    g.parse_other_methods_tex(script_dir / "results/other_methods/node_classification_results_table_with_features.tex", tables)
    g.parse_jdr_tex(script_dir / "results/rawr_no_features/jdr_no_features_backbones_table.tex", tables)
    g.parse_jdr_tex(script_dir / "results/rawr_with_features/jdr_with_features_backbones_table.tex", tables)

    per_model_feature_ranks: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model in g.MODELS:
        per_model_feature_ranks[model] = {}
        for feature in g.FEATURE_SETTINGS:
            per_model_feature_ranks[model][feature] = g.compute_avg_ranks(tables[model][feature])

    methods = g.ROW_ORDER
    compact_rows = []
    for method in methods:
        no_feat = mean([per_model_feature_ranks[m]["No"][method] for m in g.MODELS])
        with_feat = mean([per_model_feature_ranks[m]["Yes"][method] for m in g.MODELS])
        compact_rows.append((method, no_feat, with_feat))

    best_no = min(v[1] for v in compact_rows)
    best_yes = min(v[2] for v in compact_rows)

    method_labels = {
        "Baseline": "Backbone",
        "Baseline+MN": "+MN",
        "Baseline+BORF": "+BORF",
        "Baseline+FOSR": "+FOSR",
        "Baseline+SDRF": "+SDRF",
        "Baseline+JDR": "+JDR",
        "Baseline+RepNodes": "+RepNodes",
        "Baseline+RepEdges": "+RepEdges",
    }

    out_dir = script_dir / "results/other_methods/final_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "comparison_compact_avg_rank.tex"

    with out_path.open("w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\resizebox{0.72\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\hline\n")
        f.write("Method & Avg.Rank (No Feat.) & Avg.Rank (With Feat.) \\\\\n")
        f.write("\\hline\n")
        for method, no_feat, with_feat in compact_rows:
            no_cell = format_rank(no_feat, abs(no_feat - best_no) < 1e-12)
            yes_cell = format_rank(with_feat, abs(with_feat - best_yes) < 1e-12)
            f.write(f"{method_labels[method]} & {no_cell} & {yes_cell} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}}\n")
        f.write(
            "\\caption{Compact summary of additional experiments using average rank (lower is better). "
            "Values are averaged over backbones; bold marks the best in each column.}\n"
        )
        f.write("\\label{tab:compact_avg_rank_additional}\n")
        f.write("\\end{table}\n")

    print(f"[DONE] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
