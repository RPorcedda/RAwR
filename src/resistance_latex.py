from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
REPRO_ROOT = SRC_DIR.parent
DEFAULT_RESISTANCE_CSV = REPRO_ROOT / "results" / "resistance_results.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read resistance_results.csv and print three LaTeX tables for total "
            "resistance (min/mean/max over eps), with percentage variation vs "
            "the original graph in brackets."
        )
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_RESISTANCE_CSV,
        help="Path to resistance results CSV.",
    )
    p.add_argument(
        "--sci-digits",
        type=int,
        default=0,
        help=(
            "Digits after decimal in scientific notation for values. "
            "Use 0 for integer-rounded scientific notation."
        ),
    )
    p.add_argument(
        "--pct-decimals",
        type=int,
        default=2,
        help="Decimals for percentage variation values.",
    )
    p.add_argument(
        "--show-augmentation",
        choices=["both", "repnodes", "repedges"],
        default="both",
        help="Choose whether to show both augmented columns or only one of them.",
    )
    return p.parse_args()


def _escape_latex(text: str) -> str:
    return text.replace("_", r"\_")


def _dataset_order(df: pd.DataFrame) -> list[str]:
    return df["dataset"].dropna().astype(str).drop_duplicates().tolist()


def _fmt_pct(x: float, decimals: int) -> str:
    if pd.isna(x):
        return "--"
    return f"{x:+.{decimals}f}\\%"


def _fmt_value_sci_int(x: float, sci_digits: int) -> str:
    if pd.isna(x):
        return "--"
    rounded = int(round(float(x)))
    return f"{rounded:.{sci_digits}e}"


def _fmt_original_cell(value: float, sci_digits: int) -> str:
    if pd.isna(value):
        return "--"
    return _fmt_value_sci_int(value, sci_digits=sci_digits)


def _fmt_aug_cell(value: float, pct: float, sci_digits: int, pct_decimals: int) -> str:
    if pd.isna(value):
        return "--"
    return f"{_fmt_value_sci_int(value, sci_digits=sci_digits)} ({_fmt_pct(pct, pct_decimals)})"


def _compute_table_data(df: pd.DataFrame, agg_name: str) -> pd.DataFrame:
    required = {"dataset", "aug_level", "total_resistance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    work = df[["dataset", "aug_level", "total_resistance"]].copy()
    work["aug_level"] = pd.to_numeric(work["aug_level"], errors="coerce")
    work["total_resistance"] = pd.to_numeric(work["total_resistance"], errors="coerce")
    work = work.dropna(subset=["dataset", "aug_level", "total_resistance"])

    baseline = (
        work.loc[work["aug_level"] == 0, ["dataset", "total_resistance"]]
        .groupby("dataset", as_index=False)["total_resistance"]
        .mean()
        .rename(columns={"total_resistance": "Original"})
    )
    if baseline.empty:
        raise ValueError("No baseline rows found (aug_level == 0).")

    if agg_name == "min":
        agg_func = "min"
    elif agg_name == "mean":
        agg_func = "mean"
    elif agg_name == "max":
        agg_func = "max"
    else:
        raise ValueError(f"Unsupported aggregation: {agg_name}")

    rep_nodes = (
        work.loc[work["aug_level"] == 1, ["dataset", "total_resistance"]]
        .groupby("dataset", as_index=False)["total_resistance"]
        .agg(agg_func)
        .rename(columns={"total_resistance": "RepNodes"})
    )
    rep_edges = (
        work.loc[work["aug_level"] == 2, ["dataset", "total_resistance"]]
        .groupby("dataset", as_index=False)["total_resistance"]
        .agg(agg_func)
        .rename(columns={"total_resistance": "RepEdges"})
    )

    out = baseline.merge(rep_nodes, on="dataset", how="left")
    out = out.merge(rep_edges, on="dataset", how="left")

    out["RepNodes_pct"] = 100.0 * (out["RepNodes"] - out["Original"]) / out["Original"]
    out["RepEdges_pct"] = 100.0 * (out["RepEdges"] - out["Original"]) / out["Original"]

    order = _dataset_order(df)
    rank = {dataset: i for i, dataset in enumerate(order)}
    out["__rank"] = out["dataset"].map(rank).fillna(10**9).astype(int)
    out = out.sort_values("__rank").drop(columns="__rank").reset_index(drop=True)
    return out


def _latex_table(
    data: pd.DataFrame,
    title: str,
    label: str,
    sci_digits: int,
    pct_decimals: int,
    show_augmentation: str,
) -> str:
    if show_augmentation == "both":
        aug_columns = ["RepNodes", "RepEdges"]
    elif show_augmentation == "repnodes":
        aug_columns = ["RepNodes"]
    elif show_augmentation == "repedges":
        aug_columns = ["RepEdges"]
    else:
        raise ValueError(f"Unsupported show_augmentation: {show_augmentation}")

    tabular_spec = "lr" + ("r" * len(aug_columns))
    header_cols = ["Dataset", "Original", *aug_columns]

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(rf"\begin{{tabular}}{{{tabular_spec}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(header_cols) + r" \\")
    lines.append(r"\midrule")

    for _, row in data.iterrows():
        dataset = _escape_latex(str(row["dataset"]))
        original_cell = _fmt_original_cell(
            row["Original"],
            sci_digits=sci_digits,
        )

        row_cells = [dataset, original_cell]
        if "RepNodes" in aug_columns:
            row_cells.append(
                _fmt_aug_cell(
                    row["RepNodes"],
                    row["RepNodes_pct"],
                    sci_digits=sci_digits,
                    pct_decimals=pct_decimals,
                )
            )
        if "RepEdges" in aug_columns:
            row_cells.append(
                _fmt_aug_cell(
                    row["RepEdges"],
                    row["RepEdges_pct"],
                    sci_digits=sci_digits,
                    pct_decimals=pct_decimals,
                )
            )
        lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{title}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)

    min_data = _compute_table_data(df, agg_name="min")
    mean_data = _compute_table_data(df, agg_name="mean")
    max_data = _compute_table_data(df, agg_name="max")

    min_table = _latex_table(
        min_data,
        title=(
            "Total resistance table using minimum over epsilon for augmented graphs; "
            "brackets report percentage variation vs original."
        ),
        label="tab:resistance_total_min",
        sci_digits=args.sci_digits,
        pct_decimals=args.pct_decimals,
        show_augmentation=args.show_augmentation,
    )
    mean_table = _latex_table(
        mean_data,
        title=(
            "Total resistance table using mean over epsilon for augmented graphs; "
            "brackets report percentage variation vs original."
        ),
        label="tab:resistance_total_mean",
        sci_digits=args.sci_digits,
        pct_decimals=args.pct_decimals,
        show_augmentation=args.show_augmentation,
    )
    max_table = _latex_table(
        max_data,
        title=(
            "Total resistance table using maximum over epsilon for augmented graphs; "
            "brackets report percentage variation vs original."
        ),
        label="tab:resistance_total_max",
        sci_digits=args.sci_digits,
        pct_decimals=args.pct_decimals,
        show_augmentation=args.show_augmentation,
    )

    print(min_table)
    print()
    print(mean_table)
    print()
    print(max_table)


if __name__ == "__main__":
    main()
