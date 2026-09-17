#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from srl_utils import compute_srl
from utils import get_adjacency, read_labels


DATASETS = ["Caterpillar", "Grid", "Ladder", "Line", "Lobster", "Tree"]
DATASET_KEY = {d: f"CanYouHearMeNow/{d}" for d in DATASETS}
FEATURES = ["Yes", "No"]
FEATURE_BOOL = {"Yes": True, "No": False}
AUG_LEVEL_TO_NAME = {1: "RepNodes", 2: "RepEdges"}

METHOD_ORDER = [
    "Baseline",
    "BORF",
    "FOSR",
    "SDRF",
    "JDR",
    "ComFy",
    "TRIGON",
    "MN",
    "RepNodes_SRLstar",
    "RepNodes_best",
    "RepEdges_SRLstar",
    "RepEdges_best",
]

METHOD_LABEL = {
    "Baseline": "Baseline",
    "MN": "MN",
    "RepNodes_SRLstar": "RepNodes (SRL*)",
    "RepNodes_best": "RepNodes (best)",
    "RepEdges_SRLstar": "RepEdges (SRL*)",
    "RepEdges_best": "RepEdges (best)",
    "BORF": "BORF",
    "FOSR": "FOSR",
    "SDRF": "SDRF",
    "JDR": "JDR",
    "ComFy": "ComFy",
    "TRIGON": "TRIGON",
}

THIS_DIR = Path(__file__).resolve().parent
REPRO_ROOT = THIS_DIR.parent
PROJECT_ROOT = REPRO_ROOT
REF_DIR = REPRO_ROOT / "reference_results"
DEFAULT_RESULTS_DIR = REPRO_ROOT / "results"


def _bool_or(A: sp.spmatrix, B: sp.spmatrix) -> sp.spmatrix:
    C = A + B
    if C.nnz:
        C.data[:] = 1
    C.eliminate_zeros()
    return C


def _bool_and_not(A: sp.spmatrix, B: sp.spmatrix) -> sp.spmatrix:
    C = A - A.multiply(B)
    if C.nnz:
        C.data[:] = 1
    C.eliminate_zeros()
    return C


def _bool_matmul(A: sp.spmatrix, B: sp.spmatrix) -> sp.spmatrix:
    C = A @ B
    if C.nnz:
        C.data[:] = 1
    C.eliminate_zeros()
    return C


def k_ncs_exact_virtual_aware(
    adj: sp.spmatrix, labels: np.ndarray, is_labeled: np.ndarray, k: int = 2
) -> Tuple[np.ndarray, float]:
    assert k >= 2
    n = adj.shape[0]
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    m = np.asarray(is_labeled, dtype=np.int8).reshape(-1)
    assert labels.size == n and m.size == n

    if np.any(m == 1):
        num_classes = int(labels[m == 1].max()) + 1
    else:
        num_classes = int(labels.max()) + 1

    Ahat = adj.tocsr()
    Ahat = _bool_or(Ahat, sp.eye(n, format="csr"))

    reach = sp.eye(n, format="csr")
    frontier = reach
    for _ in range(k - 1):
        new = _bool_matmul(frontier, Ahat)
        new = _bool_and_not(new, reach)
        if new.nnz == 0:
            break
        reach = _bool_or(reach, new)
        frontier = new

    rows = np.arange(n, dtype=np.int64)
    Y = sp.csr_matrix((m.astype(np.float32), (rows, labels)), shape=(n, num_classes))
    counts = (reach @ Y).astype(np.float32).toarray()
    labeled_ball_size = np.asarray(reach @ m).astype(np.float32)

    idx_lab = np.flatnonzero(m == 1)
    if idx_lab.size > 0:
        counts[idx_lab, labels[idx_lab]] -= 1.0
        labeled_ball_size[idx_lab] -= 1.0

    denom = np.maximum(labeled_ball_size, 1.0)
    T = counts / denom[:, None]

    S = Ahat @ T
    deg = np.asarray(Ahat.sum(axis=1)).reshape(-1).astype(np.float32)
    deg = np.maximum(deg, 1.0)

    node_k_ncs = np.full(n, np.nan, dtype=np.float32)
    if idx_lab.size > 0:
        node_k_ncs[idx_lab] = S[idx_lab, labels[idx_lab]] / deg[idx_lab]
        graph_k_ncs = float(np.nanmean(node_k_ncs[idx_lab]))
    else:
        graph_k_ncs = float(np.nanmean(node_k_ncs))
    return node_k_ncs, graph_k_ncs


def compute_2ncs(dataset: str, augmentation: str, eps: int) -> float:
    A = get_adjacency(dataset, augmentation=augmentation, eps=eps)
    y_obs = read_labels(f"{dataset}/{dataset}.y").reshape(-1)

    n_obs = y_obs.size
    n_tot = A.shape[0]
    labels_full = np.zeros(n_tot, dtype=np.int64)
    mask_full = np.zeros(n_tot, dtype=np.int8)

    valid = (y_obs != -1) & (y_obs != -100)
    labels_full[:n_obs] = np.where(valid, y_obs, 0).astype(np.int64, copy=False)
    mask_full[:n_obs] = valid.astype(np.int8, copy=False)

    _, g2 = k_ncs_exact_virtual_aware(sp.csr_matrix(A), labels_full, mask_full, k=2)
    return float(g2)


def zscore_group(values: pd.Series) -> pd.Series:
    arr = values.to_numpy(dtype=float)
    mu = float(np.mean(arr))
    sigma = float(np.std(arr))
    if sigma < 1e-12:
        return pd.Series(np.zeros_like(arr), index=values.index, dtype=float)
    return pd.Series((arr - mu) / sigma, index=values.index, dtype=float)


def pick_best_eps_by_mean_acc(group: pd.DataFrame) -> int:
    agg = (
        group.groupby("epsilon", as_index=False)["test_acc"]
        .mean()
        .sort_values(["test_acc", "epsilon"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return int(agg.iloc[0]["epsilon"])


def build_metric_table(rawr_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    gcn = rawr_df[(rawr_df["model"] == "GCN") & (rawr_df["augmentation"].isin([1, 2]))].copy()
    gcn["epsilon"] = pd.to_numeric(gcn["epsilon"], errors="coerce")
    gcn = gcn[gcn["epsilon"].notna()].copy()
    gcn["epsilon"] = gcn["epsilon"].astype(int)

    combos = (
        gcn[["dataset", "augmentation", "epsilon"]]
        .drop_duplicates()
        .sort_values(["dataset", "augmentation", "epsilon"])
        .reset_index(drop=True)
    )

    for r in combos.itertuples(index=False):
        dataset = str(r.dataset)
        aug_level = int(r.augmentation)
        eps = int(r.epsilon)
        aug_name = AUG_LEVEL_TO_NAME[aug_level]

        A, B, R = get_adjacency(dataset, aug_name, eps, return_blocks=True)
        y = read_labels(f"{dataset}/{dataset}.y").reshape(-1)
        out = compute_srl(
            A=sp.csr_matrix(A),
            y=y,
            R=sp.csr_matrix(B),
            Q=sp.csr_matrix(R),
            mode="exact2x2",
        )
        ncs2 = compute_2ncs(dataset=dataset, augmentation=aug_name, eps=eps)

        rows.append(
            {
                "dataset": dataset,
                "dataset_key": DATASET_KEY[dataset],
                "augmentation": aug_name,
                "augmentation_level": aug_level,
                "epsilon": eps,
                "rho": float(out["rho"]),
                "srl": float(out["srl_total"]),
                "ncs2": float(ncs2),
            }
        )

    df = pd.DataFrame(rows)
    df["z_srl"] = df.groupby(["dataset", "augmentation"])["srl"].transform(zscore_group)
    df["z_ncs2"] = df.groupby(["dataset", "augmentation"])["ncs2"].transform(zscore_group)
    rho_clip = np.clip(df["rho"].to_numpy(dtype=float), 0.0, 1.0)
    w = np.power(rho_clip, 0.25)
    df["srl_star"] = w * df["z_srl"].to_numpy(dtype=float) + (1.0 - w) * df["z_ncs2"].to_numpy(dtype=float)
    return df


def build_selection_table(df_metrics: pd.DataFrame) -> pd.DataFrame:
    selected = (
        df_metrics.sort_values(
            ["dataset", "augmentation", "srl_star", "epsilon"],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(["dataset", "augmentation"], keep="first")
        .loc[
            :,
            [
                "dataset",
                "dataset_key",
                "augmentation",
                "augmentation_level",
                "epsilon",
                "srl_star",
            ],
        ]
        .rename(columns={"epsilon": "epsilon_srl_star", "srl_star": "srl_star_max"})
        .reset_index(drop=True)
    )
    return selected


def _cell(kind: str, mean: float | None = None, std: float | None = None, epsilon: int | None = None) -> Dict[str, object]:
    return {
        "kind": kind,  # value | oom | dash | missing
        "mean": mean,
        "std": std,
        "epsilon": epsilon,
    }


def _mean_std_percent(s: pd.Series) -> Tuple[float, float]:
    arr = s.to_numpy(dtype=float)
    mean = float(np.mean(arr) * 100.0)
    std = float(np.std(arr, ddof=1) * 100.0) if arr.size > 1 else 0.0
    return mean, std


def make_empty_table() -> Dict[str, Dict[str, Dict[str, Dict[str, object]]]]:
    table: Dict[str, Dict[str, Dict[str, Dict[str, object]]]] = {}
    for feat in FEATURES:
        table[feat] = {}
        for method in METHOD_ORDER:
            table[feat][method] = {}
            for ds in DATASETS:
                table[feat][method][ds] = _cell("missing")
    return table


def fill_rawr_rows(
    table: Dict[str, Dict[str, Dict[str, Dict[str, object]]]],
    rawr_df: pd.DataFrame,
    selected_df: pd.DataFrame,
) -> None:
    gcn = rawr_df[rawr_df["model"] == "GCN"].copy()
    gcn["epsilon"] = pd.to_numeric(gcn["epsilon"], errors="coerce")

    selected_map = {
        (str(r.dataset), int(r.augmentation_level)): int(r.epsilon_srl_star)
        for r in selected_df.itertuples(index=False)
    }

    for feat in FEATURES:
        feat_sub = gcn[gcn["feat_presence"] == FEATURE_BOOL[feat]]
        for ds in DATASETS:
            ds_sub = feat_sub[feat_sub["dataset"] == ds]

            # Baseline
            b = ds_sub[ds_sub["augmentation"] == 0]["test_acc"]
            if not b.empty:
                mean, std = _mean_std_percent(b)
                table[feat]["Baseline"][ds] = _cell("value", mean, std, None)

            # MN = RAwR at 100th percentile epsilon (max epsilon), RepNodes branch
            mn_sub = ds_sub[(ds_sub["augmentation"] == 1) & (ds_sub["epsilon"].notna())].copy()
            if mn_sub.empty:
                mn_sub = ds_sub[(ds_sub["augmentation"] == 2) & (ds_sub["epsilon"].notna())].copy()
            if not mn_sub.empty:
                mn_sub["epsilon"] = mn_sub["epsilon"].astype(int)
                eps_mn = int(mn_sub["epsilon"].max())
                mn_vals = mn_sub[mn_sub["epsilon"] == eps_mn]["test_acc"]
                mean, std = _mean_std_percent(mn_vals)
                table[feat]["MN"][ds] = _cell("value", mean, std, eps_mn)

            for aug_level, aug_name in [(1, "RepNodes"), (2, "RepEdges")]:
                key_srl = f"{aug_name}_SRLstar"
                key_best = f"{aug_name}_best"

                aug_sub = ds_sub[(ds_sub["augmentation"] == aug_level) & (ds_sub["epsilon"].notna())].copy()
                if aug_sub.empty:
                    continue
                aug_sub["epsilon"] = aug_sub["epsilon"].astype(int)

                eps_best = pick_best_eps_by_mean_acc(aug_sub)
                eps_srl = selected_map[(ds, aug_level)]

                vals_srl = aug_sub[aug_sub["epsilon"] == eps_srl]["test_acc"]
                if not vals_srl.empty:
                    mean_srl, std_srl = _mean_std_percent(vals_srl)
                    table[feat][key_srl][ds] = _cell("value", mean_srl, std_srl, eps_srl)

                if eps_best == eps_srl:
                    # Keep the numeric value for ranking, but render '-' in the table.
                    # This enforces the ex-aequo policy in Avg.Rank.
                    if not vals_srl.empty:
                        table[feat][key_best][ds] = _cell("dash", mean_srl, std_srl, eps_best)
                    else:
                        table[feat][key_best][ds] = _cell("dash", None, None, eps_best)
                else:
                    vals_best = aug_sub[aug_sub["epsilon"] == eps_best]["test_acc"]
                    if not vals_best.empty:
                        mean_best, std_best = _mean_std_percent(vals_best)
                        table[feat][key_best][ds] = _cell("value", mean_best, std_best, eps_best)


def fill_other_rewirings(
    table: Dict[str, Dict[str, Dict[str, Dict[str, object]]]],
    other_df: pd.DataFrame,
) -> None:
    gcn = other_df[other_df["model"] == "GCN"].copy()
    gcn["method"] = gcn["dataset"].str.extract(r"(BORF|FOSR|SDRF)$", expand=False)
    gcn["base_dataset"] = gcn["dataset"].str.replace(r"(BORF|FOSR|SDRF)$", "", regex=True)

    for feat in FEATURES:
        feat_sub = gcn[gcn["feat_presence"] == FEATURE_BOOL[feat]]
        for ds in DATASETS:
            ds_sub = feat_sub[feat_sub["base_dataset"] == ds]
            for method in ["BORF", "FOSR", "SDRF"]:
                vals = ds_sub[ds_sub["method"] == method]["test_acc"]
                if vals.empty:
                    continue
                mean, std = _mean_std_percent(vals)
                table[feat][method][ds] = _cell("value", mean, std, None)


def fill_comfy_trigon(
    table: Dict[str, Dict[str, Dict[str, Dict[str, object]]]],
    comfy_df: pd.DataFrame,
) -> None:
    gcn = comfy_df[comfy_df["model"] == "GCN"].copy()

    for feat in FEATURES:
        feat_sub = gcn[gcn["feat_presence"] == FEATURE_BOOL[feat]]
        for ds in DATASETS:
            ds_key = DATASET_KEY[ds]
            sub = feat_sub[feat_sub["dataset"] == ds_key]
            if sub.empty:
                continue
            for method_tag, method_name in [
                ("comfy", "ComFy"),
                ("trigon", "TRIGON"),
            ]:
                vals = sub[sub["method_tag"].str.lower().str.startswith(method_tag)]["test_acc"]
                if vals.empty:
                    continue
                mean, std = _mean_std_percent(vals)
                table[feat][method_name][ds] = _cell("value", mean, std, None)


def _jdr_map_from_csv(path: Path) -> Dict[str, Dict[str, object]]:
    df = pd.read_csv(path)
    out: Dict[str, Dict[str, object]] = {}
    for r in df.itertuples(index=False):
        ds = str(r.dataset)
        status = str(r.status).strip().lower()
        mean = None if pd.isna(r.test_acc_mean) else float(r.test_acc_mean)
        std = None if pd.isna(r.test_acc_std) else float(r.test_acc_std)
        out[ds] = {"status": status, "mean": mean, "std": std}
    return out


def fill_jdr(
    table: Dict[str, Dict[str, Dict[str, Dict[str, object]]]],
    jdr_no_csv: Path,
    jdr_no_fix_csv: Path | None,
    jdr_yes_csv: Path,
    jdr_yes_fix_csv: Path | None,
) -> None:
    no_map = _jdr_map_from_csv(jdr_no_csv)
    if jdr_no_fix_csv is not None and jdr_no_fix_csv.exists():
        no_fix = _jdr_map_from_csv(jdr_no_fix_csv)
        for ds, rec in no_fix.items():
            if rec.get("status") == "ok":
                no_map[ds] = rec

    yes_map = _jdr_map_from_csv(jdr_yes_csv)
    if jdr_yes_fix_csv is not None and jdr_yes_fix_csv.exists():
        yes_fix = _jdr_map_from_csv(jdr_yes_fix_csv)
        for ds, rec in yes_fix.items():
            if rec.get("status") == "ok":
                yes_map[ds] = rec

    for ds in DATASETS:
        # No features
        rec_no = no_map.get(ds)
        if rec_no is None:
            table["No"]["JDR"][ds] = _cell("missing")
        elif rec_no["status"] == "ok" and rec_no["mean"] is not None:
            table["No"]["JDR"][ds] = _cell("value", float(rec_no["mean"]), float(rec_no["std"] or 0.0), None)
        elif rec_no["status"] == "failed":
            table["No"]["JDR"][ds] = _cell("oom")
        else:
            table["No"]["JDR"][ds] = _cell("missing")

        # Yes features
        rec_yes = yes_map.get(ds)
        if rec_yes is None:
            table["Yes"]["JDR"][ds] = _cell("missing")
        elif rec_yes["status"] == "ok" and rec_yes["mean"] is not None:
            table["Yes"]["JDR"][ds] = _cell("value", float(rec_yes["mean"]), float(rec_yes["std"] or 0.0), None)
        elif rec_yes["status"] == "failed":
            table["Yes"]["JDR"][ds] = _cell("oom")
        else:
            table["Yes"]["JDR"][ds] = _cell("missing")


def compute_best_second(
    table: Dict[str, Dict[str, Dict[str, Dict[str, object]]]]
) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, Dict[str, List[str]]]]:
    best: Dict[str, Dict[str, List[str]]] = {f: {d: [] for d in DATASETS} for f in FEATURES}
    second: Dict[str, Dict[str, List[str]]] = {f: {d: [] for d in DATASETS} for f in FEATURES}

    for feat in FEATURES:
        for ds in DATASETS:
            vals = []
            for m in METHOD_ORDER:
                c = table[feat][m][ds]
                if c["kind"] == "value":
                    vals.append((m, float(c["mean"])))
            if not vals:
                continue
            top = max(v for _, v in vals)
            best[feat][ds] = [m for m, v in vals if abs(v - top) < 1e-12]
            uniq = sorted({v for _, v in vals}, reverse=True)
            if len(uniq) >= 2:
                second_val = uniq[1]
                second[feat][ds] = [m for m, v in vals if abs(v - second_val) < 1e-12]

    return best, second


def compute_avg_ranks(table: Dict[str, Dict[str, Dict[str, Dict[str, object]]]]) -> Dict[str, Dict[str, float]]:
    avg: Dict[str, Dict[str, float]] = {f: {m: 0.0 for m in METHOD_ORDER} for f in FEATURES}
    for feat in FEATURES:
        sums = {m: 0.0 for m in METHOD_ORDER}
        for ds in DATASETS:
            vals = []
            for m in METHOD_ORDER:
                c = table[feat][m][ds]
                if c["kind"] in {"value", "dash"} and c["mean"] is not None:
                    vals.append((m, float(c["mean"])))
            vals.sort(key=lambda x: x[1], reverse=True)
            ranks: Dict[str, float] = {}
            i = 0
            while i < len(vals):
                j = i + 1
                while j < len(vals) and abs(vals[j][1] - vals[i][1]) < 1e-12:
                    j += 1
                # Competition ranking (ex-aequo): [50, 50, 30] -> [1, 1, 3]
                rank = float(i + 1)
                for k in range(i, j):
                    ranks[vals[k][0]] = rank
                i = j
            miss_rank = len(vals) + 1.0
            for m in METHOD_ORDER:
                sums[m] += ranks.get(m, miss_rank)
        for m in METHOD_ORDER:
            avg[feat][m] = sums[m] / float(len(DATASETS))
    return avg


def avg_best_second(avg: Dict[str, Dict[str, float]]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    best: Dict[str, List[str]] = {f: [] for f in FEATURES}
    second: Dict[str, List[str]] = {f: [] for f in FEATURES}
    for feat in FEATURES:
        uniq = sorted(set(avg[feat].values()))
        if uniq:
            best[feat] = [m for m in METHOD_ORDER if abs(avg[feat][m] - uniq[0]) < 1e-12]
        if len(uniq) >= 2:
            second[feat] = [m for m in METHOD_ORDER if abs(avg[feat][m] - uniq[1]) < 1e-12]
    return best, second


def cell_to_string(cell: Dict[str, object]) -> str:
    kind = str(cell["kind"])
    if kind == "value":
        return f"{float(cell['mean']):.2f} $\\pm$ {float(cell['std']):.2f}"
    if kind == "oom":
        return "OOM"
    if kind == "dash":
        return "-"
    return "--"


def write_latex_table(
    table: Dict[str, Dict[str, Dict[str, Dict[str, object]]]],
    out_path: Path,
) -> None:
    best, second = compute_best_second(table)
    avg = compute_avg_ranks(table)
    avg_b, avg_s = avg_best_second(avg)

    lines: List[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{ll" + "c" * len(DATASETS) + "c}")
    lines.append("\\hline")
    lines.append("Features & Method & " + " & ".join(DATASETS) + " & Avg.Rank \\\\")
    lines.append("\\hline")

    for feat in FEATURES:
        n_rows = len(METHOD_ORDER)
        for idx, m in enumerate(METHOD_ORDER):
            vals = []
            for ds in DATASETS:
                cell = table[feat][m][ds]
                txt = cell_to_string(cell)
                if cell["kind"] == "value":
                    if m in best[feat][ds]:
                        txt = f"\\textbf{{{txt}}}"
                    elif m in second[feat][ds]:
                        txt = f"\\underline{{{txt}}}"
                vals.append(txt)

            avg_txt = f"{avg[feat][m]:.2f}"
            if m in avg_b[feat]:
                avg_txt = f"\\textbf{{{avg_txt}}}"
            elif m in avg_s[feat]:
                avg_txt = f"\\underline{{{avg_txt}}}"

            feat_cell = f"\\multirow{{{n_rows}}}{{*}}{{{feat}}}" if idx == 0 else ""
            lines.append(
                f"{feat_cell} & {METHOD_LABEL[m]} & "
                + " & ".join(vals)
                + f" & {avg_txt} \\\\"
            )
        lines.append("\\hline")

    lines.append("\\end{tabular}}")
    lines.append(
        "\\caption{GCN node classification test accuracies at L=2 on CanYouHearMeNow. "
        "For RAwR, we report both the best result over the epsilon grid and the one chosen by the SRL* heuristic; "
        "if they coincide we report '-'. OOM indicates a failed run due to memory. "
        "Best per dataset is bold, second-best underlined; same policy for Avg.Rank (lower is better).}"
    )
    lines.append("\\label{tab:cyhmn_gcn_srlstar}")
    lines.append("\\end{table}")

    out_path.write_text("\n".join(lines) + "\n")


def table_to_summary_df(
    table: Dict[str, Dict[str, Dict[str, Dict[str, object]]]]
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for feat in FEATURES:
        for m in METHOD_ORDER:
            for ds in DATASETS:
                c = table[feat][m][ds]
                rows.append(
                    {
                        "model": "GCN",
                        "features": feat,
                        "method": m,
                        "dataset": DATASET_KEY[ds],
                        "kind": c["kind"],
                        "test_acc_mean": c["mean"],
                        "test_acc_std": c["std"],
                        "epsilon": c["epsilon"],
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description="Build CYHMN GCN table with SRL* policy")
    p.add_argument("--rawr-csv", type=Path, default=REF_DIR / "cyhmn_rawr.csv")
    p.add_argument("--other-csv", type=Path, default=REF_DIR / "cyhmn_other_rewirings.csv")
    p.add_argument(
        "--comfy-trigon-csv",
        type=Path,
        default=REF_DIR / "comfy_trigon_canyouhearmenow_node_classification.csv",
    )
    p.add_argument(
        "--jdr-no-csv",
        type=Path,
        default=REF_DIR / "jdr" / "jdr_no_features_backbones_results.csv",
    )
    p.add_argument(
        "--jdr-no-fix-csv",
        type=Path,
        default=REF_DIR / "jdr" / "cyhmn_jdr_no_features_grid_fix.csv",
    )
    p.add_argument(
        "--jdr-yes-csv",
        type=Path,
        default=REF_DIR / "jdr" / "jdr_with_features_backbones_results_base.csv",
    )
    p.add_argument(
        "--jdr-yes-fix-csv",
        type=Path,
        default=REF_DIR / "jdr" / "jdr_with_features_backbones_results_fix.csv",
    )
    p.add_argument("--out-metrics", type=Path, default=DEFAULT_RESULTS_DIR / "cyhmn_srlstar_metrics.csv")
    p.add_argument("--out-selected", type=Path, default=DEFAULT_RESULTS_DIR / "cyhmn_srlstar_selected_eps.csv")
    p.add_argument("--out-summary", type=Path, default=DEFAULT_RESULTS_DIR / "cyhmn_srlstar_summary.csv")
    p.add_argument("--out-latex", type=Path, default=DEFAULT_RESULTS_DIR / "gcn_cyhmn_rewiring_comparison.tex")
    args = p.parse_args()

    old_cwd = Path.cwd()
    os.chdir(PROJECT_ROOT)
    try:
        rawr_df = pd.read_csv(args.rawr_csv)
        other_df = pd.read_csv(args.other_csv)
        comfy_df = pd.read_csv(args.comfy_trigon_csv)

        metrics = build_metric_table(rawr_df)
        selected = build_selection_table(metrics)

        table = make_empty_table()
        fill_rawr_rows(table, rawr_df, selected)
        fill_other_rewirings(table, other_df)
        fill_comfy_trigon(table, comfy_df)
        fill_jdr(table, args.jdr_no_csv, args.jdr_no_fix_csv, args.jdr_yes_csv, args.jdr_yes_fix_csv)

        summary = table_to_summary_df(table)

        args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
        args.out_selected.parent.mkdir(parents=True, exist_ok=True)
        args.out_summary.parent.mkdir(parents=True, exist_ok=True)
        args.out_latex.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(args.out_metrics, index=False)
        selected.to_csv(args.out_selected, index=False)
        summary.to_csv(args.out_summary, index=False)
        write_latex_table(table, args.out_latex)
    finally:
        os.chdir(old_cwd)

    print(f"[OK] metrics:  {args.out_metrics}")
    print(f"[OK] selected: {args.out_selected}")
    print(f"[OK] summary:  {args.out_summary}")
    print(f"[OK] latex:    {args.out_latex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
