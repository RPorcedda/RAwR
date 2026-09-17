import argparse
import csv
import pathlib
import statistics
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_MODELS = ["GCN", "GAT", "GIN"]
DEFAULT_DATASETS = ["Actor", "Chameleon", "Citeseer", "Cora", "Cornell", "PubMed", "Squirrel", "Texas", "Wisconsin"]
DEFAULT_METHODS = ["BORF", "FOSR", "SDRF"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables for other_methods node classification runs (with/without features)."
    )
    parser.add_argument(
        "--no_features_csv",
        type=str,
        default="results/other_methods/node_classification_results.csv",
        help="CSV path for no-feature runs.",
    )
    parser.add_argument(
        "--with_features_csv",
        type=str,
        default="results/other_methods/node_classification_results_with_features",
        help="CSV path for feature-enabled runs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/other_methods",
        help="Directory where LaTeX tables will be written.",
    )
    return parser.parse_args()


def parse_dataset_and_method(dataset_value: str) -> Tuple[str, str]:
    for method in DEFAULT_METHODS:
        if dataset_value.endswith(method):
            dataset = dataset_value[: -len(method)]
            if not dataset:
                break
            return dataset, method
    raise ValueError(f"Could not parse dataset/method from dataset value '{dataset_value}'")


def yes_no(value: str) -> str:
    value_norm = value.strip().lower()
    if value_norm in {"true", "1", "yes"}:
        return "Yes"
    if value_norm in {"false", "0", "no"}:
        return "No"
    raise ValueError(f"Unsupported feat_presence value '{value}'")


def ordered(values: Iterable[str], preferred: Sequence[str]) -> List[str]:
    values_set = set(values)
    ordered_vals = [v for v in preferred if v in values_set]
    ordered_vals.extend(sorted(v for v in values_set if v not in ordered_vals))
    return ordered_vals


def format_cell(mean: float, std: float) -> str:
    return f"{mean:.2f} $\\pm$ {std:.2f}"


def aggregate_trials(csv_path: pathlib.Path) -> Tuple[Dict[Tuple[str, str, str], Tuple[float, float]], List[str], List[str], List[str], str]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in '{csv_path}'")

    trial_values: Dict[Tuple[str, str, str], List[float]] = {}
    models = set()
    methods = set()
    datasets = set()
    feature_values = set()

    for row in rows:
        model = row["model"].strip()
        dataset_raw = row["dataset"].strip()
        feature_values.add(yes_no(row["feat_presence"]))
        dataset, method = parse_dataset_and_method(dataset_raw)
        test_acc = float(row["test_acc"])

        key = (model, method, dataset)
        trial_values.setdefault(key, []).append(test_acc)
        models.add(model)
        methods.add(method)
        datasets.add(dataset)

    if len(feature_values) != 1:
        raise ValueError(f"Expected one feat_presence value in '{csv_path}', found: {sorted(feature_values)}")
    feature_label = next(iter(feature_values))

    aggregated: Dict[Tuple[str, str, str], Tuple[float, float]] = {}
    for key, vals in trial_values.items():
        mean = statistics.fmean(vals) * 100.0
        std = statistics.pstdev(vals) * 100.0 if len(vals) > 1 else 0.0
        aggregated[key] = (mean, std)

    model_order = ordered(models, DEFAULT_MODELS)
    method_order = ordered(methods, DEFAULT_METHODS)
    dataset_order = ordered(datasets, DEFAULT_DATASETS)

    return aggregated, model_order, method_order, dataset_order, feature_label


def write_latex_table(
    aggregated: Dict[Tuple[str, str, str], Tuple[float, float]],
    models: Sequence[str],
    methods: Sequence[str],
    datasets: Sequence[str],
    feature_label: str,
    output_path: pathlib.Path,
    caption: str,
    label: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    col_spec = "lll" + "c" * len(datasets)

    with output_path.open("w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
        f.write("\\hline\n")
        f.write("Model & Features & Method & " + " & ".join(datasets) + " \\\\\n")
        f.write("\\hline\n")
        for model in models:
            for method in methods:
                cells = []
                for dataset in datasets:
                    stats = aggregated.get((model, method, dataset))
                    if stats is None:
                        cells.append("--")
                    else:
                        cells.append(format_cell(stats[0], stats[1]))
                f.write(f"{model} & {feature_label} & {method} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\end{table}\n")


def generate_for_file(csv_path: pathlib.Path, output_path: pathlib.Path, label_suffix: str) -> None:
    aggregated, models, methods, datasets, feature_label = aggregate_trials(csv_path)
    feature_phrase = "with node features" if feature_label == "Yes" else "without node features"
    caption = (
        "Test mean accuracies (\\(\\pm\\) std) for BORF/FOSR/SDRF across backbones and datasets "
        f"({feature_phrase})."
    )
    label = f"tab:other_methods_backbones_{label_suffix}"
    write_latex_table(
        aggregated=aggregated,
        models=models,
        methods=methods,
        datasets=datasets,
        feature_label=feature_label,
        output_path=output_path,
        caption=caption,
        label=label,
    )


def main() -> int:
    args = parse_args()
    script_dir = pathlib.Path(__file__).resolve().parent
    no_features_csv = script_dir / args.no_features_csv
    with_features_csv = script_dir / args.with_features_csv
    output_dir = script_dir / args.output_dir

    no_features_tex = output_dir / "node_classification_results_table_no_features.tex"
    with_features_tex = output_dir / "node_classification_results_table_with_features.tex"

    generate_for_file(no_features_csv, no_features_tex, "no_features")
    generate_for_file(with_features_csv, with_features_tex, "with_features")

    print(f"[DONE] LaTeX (no features): {no_features_tex}")
    print(f"[DONE] LaTeX (with features): {with_features_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
