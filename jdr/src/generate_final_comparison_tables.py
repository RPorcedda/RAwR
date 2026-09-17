import argparse
import copy
import pathlib
import re
from typing import Dict, List, Optional, Sequence, Tuple


MODELS = ["GCN", "GAT", "GIN"]
FEATURE_SETTINGS = ["Yes", "No"]
DATASETS = ["Actor", "Chameleon", "Citeseer", "Cora", "Cornell", "PubMed", "Squirrel", "Texas", "Wisconsin"]
ROW_ORDER = [
    "Baseline",
    "Baseline+MN",
    "Baseline+BORF",
    "Baseline+FOSR",
    "Baseline+SDRF",
    "Baseline+JDR",
    "Baseline+RepNodes",
    "Baseline+RepEdges",
]
OTHER_METHOD_SUFFIXES = ["BORF", "FOSR", "SDRF"]

Value = Tuple[float, float]
TableValues = Dict[str, Dict[str, Dict[str, Dict[str, Optional[Value]]]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build merged per-model LaTeX tables (features/no-features) with Avg.Rank from paper, other methods, and JDR."
    )
    parser.add_argument(
        "--paper_tex",
        type=str,
        default="results/other_methods/RAwR_paper_results.tex",
        help="Path to RAwR paper result table (.tex).",
    )
    parser.add_argument(
        "--other_methods_no_features_tex",
        type=str,
        default="results/other_methods/node_classification_results_table_no_features.tex",
        help="LaTeX table for BORF/FOSR/SDRF no-feature runs.",
    )
    parser.add_argument(
        "--other_methods_with_features_tex",
        type=str,
        default="results/other_methods/node_classification_results_table_with_features.tex",
        help="LaTeX table for BORF/FOSR/SDRF feature runs.",
    )
    parser.add_argument(
        "--jdr_no_features_tex",
        type=str,
        default="results/rawr_no_features/jdr_no_features_backbones_table.tex",
        help="JDR LaTeX table for no-feature runs.",
    )
    parser.add_argument(
        "--jdr_with_features_tex",
        type=str,
        default="results/rawr_with_features/jdr_with_features_backbones_table.tex",
        help="JDR LaTeX table for feature runs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/other_methods/final_tables",
        help="Directory for generated per-model LaTeX tables.",
    )
    parser.add_argument(
        "--resize_width",
        type=str,
        default=r"\textwidth",
        help="Width argument for LaTeX resizebox, e.g. '\\textwidth' or '0.95\\textwidth'.",
    )
    return parser.parse_args()


def empty_tables() -> TableValues:
    tables: TableValues = {}
    for model in MODELS:
        tables[model] = {}
        for feature in FEATURE_SETTINGS:
            tables[model][feature] = {}
            for row in ROW_ORDER:
                tables[model][feature][row] = {dataset: None for dataset in DATASETS}
    return tables


def ordered_strip_latex(cell: str) -> str:
    out = cell.strip()
    prev = None
    while prev != out:
        prev = out
        out = re.sub(r"\\cellcolor\{[^{}]*\}", "", out)
        out = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", out)
        out = re.sub(r"\\underline\{([^{}]*)\}", r"\1", out)
        out = re.sub(r"\\multicolumn\{[^{}]*\}\{[^{}]*\}\{([^{}]*)\}", r"\1", out)
    out = out.replace("{", "").replace("}", "").strip()
    # Remove trailing LaTeX row terminator if present in the last cell.
    out = re.sub(r"\\\\\s*$", "", out).strip()
    return out


def parse_cell_value(cell: str) -> Optional[Value]:
    clean = ordered_strip_latex(cell)
    if not clean or clean == "--" or "OOM" in clean:
        return None

    pm_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*\$?\s*\\pm\s*\$?\s*([0-9]+(?:\.[0-9]+)?)", clean)
    if pm_match:
        return float(pm_match.group(1)), float(pm_match.group(2))

    num_match = re.search(r"([0-9]+(?:\.[0-9]+)?)", clean)
    if num_match:
        return float(num_match.group(1)), 0.0

    return None


def parse_paper_table(path: pathlib.Path, tables: TableValues) -> None:
    method_map = {
        "Baseline": "Baseline",
        "MN": "Baseline+MN",
        "RepNodes (SRL*)": "RepNodes_SRL",
        "RepNodes (best)": "Baseline+RepNodes",
        "RepEdges (SRL*)": "RepEdges_SRL",
        "RepEdges (best)": "Baseline+RepEdges",
    }

    current_model: Optional[str] = None
    current_feature: Optional[str] = None
    srl_cache: Dict[str, Dict[str, Dict[str, Dict[str, Optional[Value]]]]] = {
        model: {
            feature: {
                "RepNodes_SRL": {dataset: None for dataset in DATASETS},
                "RepEdges_SRL": {dataset: None for dataset in DATASETS},
            }
            for feature in FEATURE_SETTINGS
        }
        for model in MODELS
    }

    for raw_line in path.read_text().splitlines():
        if "&" not in raw_line or "\\\\" not in raw_line:
            continue
        line = raw_line.strip()
        if line.startswith("\\toprule") or line.startswith("\\midrule") or line.startswith("\\bottomrule"):
            continue
        if "Dataset" in line or "Model & Features & Method" in line:
            continue

        parts = [part.strip() for part in line.split("&")]
        if len(parts) < 3:
            continue

        model_match = re.search(r"\{(GCN|GAT|GIN)\}", parts[0])
        if model_match:
            current_model = model_match.group(1)

        feature_match = re.search(r"\{(Yes|No)\}", parts[1])
        if feature_match:
            current_feature = feature_match.group(1)

        if current_model is None or current_feature is None:
            continue

        method_raw = ordered_strip_latex(parts[2])
        row_label = method_map.get(method_raw)
        if row_label is None:
            continue

        dataset_cells = parts[3 : 3 + len(DATASETS)]
        if len(dataset_cells) < len(DATASETS):
            continue

        for dataset, cell in zip(DATASETS, dataset_cells):
            value = parse_cell_value(cell)

            # In the paper table, "--" on "(best)" rows means "same as SRL*".
            if value is None and row_label in {"Baseline+RepNodes", "Baseline+RepEdges"}:
                clean_cell = ordered_strip_latex(cell)
                if clean_cell == "--":
                    source_row = "RepNodes_SRL" if row_label == "Baseline+RepNodes" else "RepEdges_SRL"
                    value = srl_cache[current_model][current_feature][source_row][dataset]

            if row_label in {"RepNodes_SRL", "RepEdges_SRL"}:
                if value is not None:
                    srl_cache[current_model][current_feature][row_label][dataset] = value
                continue

            if value is not None:
                tables[current_model][current_feature][row_label][dataset] = value


def split_latex_row_cells(line: str) -> List[str]:
    line = line.strip()
    if line.endswith("\\\\"):
        line = line[:-2].strip()
    return [part.strip() for part in line.split("&")]


def map_headers_to_values(dataset_headers: Sequence[str], value_cells: Sequence[str]) -> List[str]:
    if len(value_cells) == len(dataset_headers):
        return list(dataset_headers)

    # Legacy robustness: some tables have one missing dataset value in the row.
    # Keep left-to-right alignment and map the last value to the last dataset header.
    if len(value_cells) == len(dataset_headers) - 1 and len(dataset_headers) >= 2:
        return list(dataset_headers[: len(value_cells) - 1]) + [dataset_headers[-1]]

    return list(dataset_headers[: len(value_cells)])


def parse_other_methods_tex(path: pathlib.Path, tables: TableValues) -> None:
    dataset_headers: Optional[List[str]] = None

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or "&" not in line or "\\\\" not in line:
            continue
        if line.startswith("\\hline") or line.startswith("\\begin") or line.startswith("\\end"):
            continue

        cells = split_latex_row_cells(line)
        if len(cells) >= 4 and cells[0] == "Model" and cells[1] == "Features" and cells[2] == "Method":
            dataset_headers = [ordered_strip_latex(c) for c in cells[3:]]
            continue

        if dataset_headers is None or len(cells) < 4:
            continue

        model = ordered_strip_latex(cells[0])
        feature = ordered_strip_latex(cells[1])
        method = ordered_strip_latex(cells[2])
        if model not in MODELS or feature not in FEATURE_SETTINGS or method not in OTHER_METHOD_SUFFIXES:
            continue

        value_cells = cells[3:]
        mapped_headers = map_headers_to_values(dataset_headers, value_cells)
        row_label = f"Baseline+{method}"

        for dataset, cell in zip(mapped_headers, value_cells):
            if dataset not in DATASETS:
                continue
            value = parse_cell_value(cell)
            if value is not None:
                tables[model][feature][row_label][dataset] = value


def parse_jdr_tex(path: pathlib.Path, tables: TableValues) -> None:
    dataset_headers: Optional[List[str]] = None

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or "&" not in line or "\\\\" not in line:
            continue
        if line.startswith("\\hline") or line.startswith("\\begin") or line.startswith("\\end"):
            continue

        cells = split_latex_row_cells(line)
        if len(cells) >= 4 and cells[0] == "Model" and cells[1] == "Features" and cells[2] == "Method":
            dataset_headers = [ordered_strip_latex(c) for c in cells[3:]]
            continue

        if dataset_headers is None or len(cells) < 4:
            continue

        model = ordered_strip_latex(cells[0])
        feature = ordered_strip_latex(cells[1])
        method = ordered_strip_latex(cells[2])
        if model not in MODELS or feature not in FEATURE_SETTINGS or method != "JDR":
            continue

        value_cells = cells[3:]
        mapped_headers = map_headers_to_values(dataset_headers, value_cells)
        for dataset, cell in zip(mapped_headers, value_cells):
            if dataset not in DATASETS:
                continue
            value = parse_cell_value(cell)
            if value is not None:
                tables[model][feature]["Baseline+JDR"][dataset] = value


def format_value(value: Optional[Value]) -> str:
    if value is None:
        return "--"
    return f"{value[0]:.2f} $\\pm$ {value[1]:.2f}"


def compute_dataset_best(rows: Dict[str, Dict[str, Optional[Value]]]) -> Dict[str, List[str]]:
    best: Dict[str, List[str]] = {dataset: [] for dataset in DATASETS}
    for dataset in DATASETS:
        vals = [(row_name, row_vals[dataset][0]) for row_name, row_vals in rows.items() if row_vals[dataset] is not None]
        if not vals:
            continue
        max_val = max(v for _, v in vals)
        best_rows = [row_name for row_name, v in vals if abs(v - max_val) < 1e-12]
        best[dataset] = best_rows
    return best


def compute_dataset_second_best(rows: Dict[str, Dict[str, Optional[Value]]]) -> Dict[str, List[str]]:
    second: Dict[str, List[str]] = {dataset: [] for dataset in DATASETS}
    for dataset in DATASETS:
        vals = [(row_name, row_vals[dataset][0]) for row_name, row_vals in rows.items() if row_vals[dataset] is not None]
        if len(vals) < 2:
            continue
        unique_vals = sorted({v for _, v in vals}, reverse=True)
        if len(unique_vals) < 2:
            continue
        second_val = unique_vals[1]
        second_rows = [row_name for row_name, v in vals if abs(v - second_val) < 1e-12]
        second[dataset] = second_rows
    return second


def compute_avg_ranks(rows: Dict[str, Dict[str, Optional[Value]]]) -> Dict[str, float]:
    rank_sums = {row_name: 0.0 for row_name in ROW_ORDER}
    dataset_count = 0

    for dataset in DATASETS:
        vals = [(row_name, row_vals[dataset][0]) for row_name, row_vals in rows.items() if row_vals[dataset] is not None]
        if not vals:
            continue
        dataset_count += 1

        vals.sort(key=lambda x: x[1], reverse=True)
        ranks_for_dataset: Dict[str, float] = {}
        i = 0
        while i < len(vals):
            j = i + 1
            while j < len(vals) and abs(vals[j][1] - vals[i][1]) < 1e-12:
                j += 1
            rank = ((i + 1) + j) / 2.0
            for k in range(i, j):
                ranks_for_dataset[vals[k][0]] = rank
            i = j

        missing_rank = len(vals) + 1.0
        for row_name in ROW_ORDER:
            rank_sums[row_name] += ranks_for_dataset.get(row_name, missing_rank)

    if dataset_count == 0:
        return {row_name: float("inf") for row_name in ROW_ORDER}
    return {row_name: rank_sums[row_name] / dataset_count for row_name in ROW_ORDER}


def write_model_table(
    model: str,
    feature: str,
    rows: Dict[str, Dict[str, Optional[Value]]],
    output_path: pathlib.Path,
    resize_width: str,
) -> None:
    best_per_dataset = compute_dataset_best(rows)
    second_per_dataset = compute_dataset_second_best(rows)
    avg_rank = compute_avg_ranks(rows)
    best_avg = min(avg_rank.values()) if avg_rank else float("inf")
    best_avg_rows = {row_name for row_name, value in avg_rank.items() if abs(value - best_avg) < 1e-12}
    unique_avg = sorted(set(avg_rank.values()))
    second_avg_rows = set()
    if len(unique_avg) >= 2:
        second_avg = unique_avg[1]
        second_avg_rows = {row_name for row_name, value in avg_rank.items() if abs(value - second_avg) < 1e-12}

    feature_phrase = "with node features" if feature == "Yes" else "without node features"
    col_spec = "l" + "c" * len(DATASETS) + "c"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write(f"\\resizebox{{{resize_width}}}{{!}}{{%\n")
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
        f.write("\\hline\n")
        f.write("Method & " + " & ".join(DATASETS) + " & Avg.Rank \\\\\n")
        f.write("\\hline\n")
        for row_name in ROW_ORDER:
            if row_name in {"Baseline+RepNodes", "Baseline+RepEdges"}:
                f.write("\\hline\n")

            row_vals = rows[row_name]
            rendered_cells: List[str] = []
            for dataset in DATASETS:
                cell = format_value(row_vals[dataset])
                if row_name in best_per_dataset[dataset] and cell != "--":
                    cell = f"\\textbf{{{cell}}}"
                elif row_name in second_per_dataset[dataset] and cell != "--":
                    cell = f"\\underline{{{cell}}}"
                rendered_cells.append(cell)
            avg_cell = f"{avg_rank[row_name]:.2f}"
            if row_name in best_avg_rows:
                avg_cell = f"\\textbf{{{avg_cell}}}"
            elif row_name in second_avg_rows:
                avg_cell = f"\\underline{{{avg_cell}}}"
            if row_name == "Baseline":
                row_display = model
            elif row_name.startswith("Baseline+"):
                row_display = f"{model}+{row_name.split('+', 1)[1]}"
            else:
                row_display = row_name
            f.write(f"{row_display} & " + " & ".join(rendered_cells) + f" & {avg_cell} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}}\n")
        f.write(
            f"\\caption{{{model} comparison across baselines and augmentations ({feature_phrase}). "
            "Best scores are highlighted in bold and second-best scores are underlined "
            "(for each dataset and Avg.Rank).}\n"
        )
        feature_tag = "with_features" if feature == "Yes" else "no_features"
        f.write(f"\\label{{tab:{model.lower()}_{feature_tag}_comparison}}\n")
        f.write("\\end{table}\n")


def main() -> int:
    args = parse_args()
    script_dir = pathlib.Path(__file__).resolve().parent

    paper_tex = script_dir / args.paper_tex
    other_methods_no_features_tex = script_dir / args.other_methods_no_features_tex
    other_methods_with_features_tex = script_dir / args.other_methods_with_features_tex
    jdr_no_features_tex = script_dir / args.jdr_no_features_tex
    jdr_with_features_tex = script_dir / args.jdr_with_features_tex
    output_dir = script_dir / args.output_dir

    tables = empty_tables()
    parse_paper_table(paper_tex, tables)
    parse_other_methods_tex(other_methods_no_features_tex, tables)
    parse_other_methods_tex(other_methods_with_features_tex, tables)
    parse_jdr_tex(jdr_no_features_tex, tables)
    parse_jdr_tex(jdr_with_features_tex, tables)

    generated_paths: List[pathlib.Path] = []
    for model in MODELS:
        for feature in FEATURE_SETTINGS:
            feature_tag = "with_features" if feature == "Yes" else "no_features"
            out_path = output_dir / f"comparison_{model.lower()}_{feature_tag}.tex"
            write_model_table(model, feature, tables[model][feature], out_path, args.resize_width)
            generated_paths.append(out_path)

    merged_file = output_dir / "comparison_all_tables.tex"
    with merged_file.open("w") as f:
        for i, path in enumerate(generated_paths):
            f.write(path.read_text())
            if i != len(generated_paths) - 1:
                f.write("\n\n")

    # Additional variant requested by user:
    # hide JDR no-feature results on Actor, Citeseer, PubMed, Squirrel.
    masked_tables = copy.deepcopy(tables)
    masked_datasets = ["Actor", "Citeseer", "PubMed", "Squirrel"]
    for model in MODELS:
        for dataset in masked_datasets:
            masked_tables[model]["No"]["Baseline+JDR"][dataset] = None

    masked_paths: List[pathlib.Path] = []
    for model in MODELS:
        for feature in FEATURE_SETTINGS:
            feature_tag = "with_features" if feature == "Yes" else "no_features"
            out_path = output_dir / f"comparison_{model.lower()}_{feature_tag}_masked_jdr_no_features.tex"
            write_model_table(model, feature, masked_tables[model][feature], out_path, args.resize_width)
            masked_paths.append(out_path)

    masked_merged_file = output_dir / "comparison_all_tables_masked_jdr_no_features.tex"
    with masked_merged_file.open("w") as f:
        for i, path in enumerate(masked_paths):
            f.write(path.read_text())
            if i != len(masked_paths) - 1:
                f.write("\n\n")

    for path in generated_paths:
        print(f"[DONE] {path}")
    print(f"[DONE] {merged_file}")
    print(f"[DONE] {masked_merged_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
