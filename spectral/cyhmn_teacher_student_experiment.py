#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from matplotlib.lines import Line2D

from synth_model.generate_teacher import generate_data_from_S
from srl_utils import compute_srl
from utils import get_adjacency, get_shift_operator, read_labels


DEFAULT_DATASETS = ["Caterpillar", "Grid", "Ladder", "Line", "Lobster", "Tree"]
PCT_LEVELS = [0, 25, 50, 75, 100]
PCT_MARKERS = {0: "v", 25: "+", 50: "x", 75: "3", 100: "4"}
THIS_DIR = Path(__file__).resolve().parent
REPRO_ROOT = THIS_DIR.parent
PROJECT_ROOT = REPRO_ROOT
DEFAULT_CYHMN_ROOT = PROJECT_ROOT / "CanYouHearMeNow"
DEFAULT_OUT_CSV = REPRO_ROOT / "results" / "cyhmn_teacher_student_results.csv"
DEFAULT_OUT_FIG = REPRO_ROOT / "results" / "cyhmn_srl_vs_mse.png"


def ensure_dataset_links(dataset_root: Path, datasets: List[str], cwd: Path) -> None:
    for ds in datasets:
        src = dataset_root / ds
        dst = cwd / ds
        if dst.exists():
            continue
        os.symlink(src, dst)


def degree_percentile_eps(dataset: str) -> List[int]:
    edges = np.loadtxt(f"{dataset}/{dataset}.edgelist")
    if edges.ndim == 1:
        edges = edges.reshape(1, -1)
    edges = edges[:, :2].astype(int)

    y = read_labels(f"{dataset}/{dataset}.y").reshape(-1)
    n = y.shape[0]
    deg = np.zeros(n, dtype=int)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1

    p25, p50, p75 = np.percentile(deg, [25, 50, 75])
    eps_values = [0, int(p25), int(p50), int(p75), int(deg.max())]
    return eps_values


def build_augmented_features(X_obs: np.ndarray, B: np.ndarray) -> np.ndarray:
    counts = np.asarray(B.sum(axis=0)).reshape(-1, 1)
    counts = np.where(counts > 0, counts, 1.0)
    X_virtual = (B.T @ X_obs) / counts
    return np.vstack([X_obs, X_virtual])


class LinearPolynomialGCN(torch.nn.Module):
    def __init__(self, s_powers: List[np.ndarray], x_obs: np.ndarray, d_out: int, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        d_in = x_obs.shape[1]
        self.Z = [torch.tensor(S @ x_obs, dtype=torch.float64) for S in s_powers]
        self.W = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.randn(d_out, d_in, dtype=torch.float64) / np.sqrt(d_in)
                )
                for _ in range(len(self.Z))
            ]
        )

    def forward(self) -> torch.Tensor:
        out = 0.0
        for i, Zi in enumerate(self.Z):
            out = out + Zi @ self.W[i].T
        return out


def train_student_on_original_graph(
    S_obs: np.ndarray,
    X_obs: np.ndarray,
    Y_teacher_obs: np.ndarray,
    epochs: int,
    lr: float,
    seed: int,
) -> float:
    s_powers = [np.linalg.matrix_power(S_obs, i) for i in range(3)]
    model = LinearPolynomialGCN(s_powers, X_obs, d_out=Y_teacher_obs.shape[1], seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    target = torch.tensor(Y_teacher_obs, dtype=torch.float64)

    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model()
        loss = torch.mean((pred - target) ** 2)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = torch.mean((model() - target) ** 2).item()
    return float(final_loss)


def run_experiment(
    datasets: List[str],
    augmentation: str,
    epochs: int,
    lr: float,
    sigma_arr: List[float],
    base_seed: int,
) -> pd.DataFrame:
    rows = []

    Path("partitions").mkdir(exist_ok=True)
    Path("reduced").mkdir(exist_ok=True)

    eps_map: Dict[str, List[int]] = {ds: degree_percentile_eps(ds) for ds in datasets}

    for ds_idx, dataset in enumerate(datasets):
        print(f"\n[Dataset] {dataset}")
        y = read_labels(f"{dataset}/{dataset}.y").reshape(-1)
        num_classes = int(np.unique(y).size)

        X_obs = np.loadtxt(f"{dataset}/{dataset}.x")
        if X_obs.ndim == 1:
            X_obs = X_obs.reshape(-1, 1)

        S_obs = get_shift_operator(dataset, augmentation=None, eps=None)

        for p_idx, (pct, eps) in enumerate(zip(PCT_LEVELS, eps_map[dataset])):
            print(f"  - epsilon percentile {pct}% -> eps={eps}")

            A_aug, B, R = get_adjacency(dataset, augmentation=augmentation, eps=eps, return_blocks=True)
            S_aug = get_shift_operator(dataset, augmentation=augmentation, eps=eps)

            X_aug = build_augmented_features(X_obs, B)

            teacher_seed = base_seed + 1000 * ds_idx
            _, Y_teacher_aug = generate_data_from_S(
                S_aug,
                n_ch=3,
                d_out=num_classes,
                seed=teacher_seed,
                sigma_arr=sigma_arr,
                X=X_aug,
            )

            out_dir = Path(dataset) / augmentation / str(eps)
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savetxt(out_dir / "features.csv", X_aug, delimiter=",")
            np.savetxt(out_dir / "labels.csv", Y_teacher_aug, delimiter=",")

            Y_teacher_obs = Y_teacher_aug[: X_obs.shape[0], :]

            mse = train_student_on_original_graph(
                S_obs=S_obs,
                X_obs=X_obs,
                Y_teacher_obs=Y_teacher_obs,
                epochs=epochs,
                lr=lr,
                seed=base_seed + 1000 * ds_idx + p_idx + 1,
            )

            srl = compute_srl(
                A=sp.csr_matrix(A_aug),
                y=y,
                R=sp.csr_matrix(B),
                Q=sp.csr_matrix(R),
                mode="exact2x2",
            )

            rows.append(
                {
                    "dataset": dataset,
                    "augmentation": augmentation,
                    "percentile": pct,
                    "eps": eps,
                    "srl": float(srl["srl_total"]),
                    "mse": mse,
                }
            )

    return pd.DataFrame(rows)


def make_plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    cmap = plt.get_cmap("tab10")
    datasets = sorted(df["dataset"].unique().tolist())
    ds_color = {ds: cmap(i % 10) for i, ds in enumerate(datasets)}

    for _, r in df.iterrows():
        ax.scatter(
            r["srl"],
            r["mse"],
            color=ds_color[r["dataset"]],
            marker=PCT_MARKERS[int(r["percentile"])],
            s=80,
            linewidths=1.4,
        )

    x = df["srl"].to_numpy()
    y = df["mse"].to_numpy()
    if len(x) >= 2:
        coef = np.polyfit(x, y, deg=1)
        xx = np.linspace(float(x.min()), float(x.max()), 200)
        yy = coef[0] * xx + coef[1]
        ax.plot(xx, yy, "--", color="0.65", linewidth=2)

    ax.set_xlabel("SRL", fontsize=16)
    ax.set_ylabel("MSE on original graph", fontsize=16)
    ax.tick_params(labelsize=14)

    dataset_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=6,
            markerfacecolor=ds_color[ds],
            markeredgecolor="none",
            label=ds,
        )
        for ds in datasets
    ]
    marker_handles = [
        Line2D(
            [0],
            [0],
            marker=PCT_MARKERS[p],
            linestyle="None",
            markersize=8,
            color="k",
            label=f"{p} %",
        )
        for p in PCT_LEVELS
    ]

    leg1 = ax.legend(handles=dataset_handles, loc="upper left", ncol=2, fontsize=12, framealpha=0.95)
    ax.add_artist(leg1)
    ax.legend(handles=marker_handles, loc="lower right", ncol=2, fontsize=12, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Teacher-student SRL/MSE experiment on CanYouHearMeNow datasets")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_CYHMN_ROOT)
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--augmentation", choices=["RepNodes", "RepEdges"], default="RepEdges")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sigma0", type=float, default=1.0)
    parser.add_argument("--sigma1", type=float, default=1.0)
    parser.add_argument("--sigma2", type=float, default=40.0)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-fig", type=Path, default=DEFAULT_OUT_FIG)
    args = parser.parse_args()

    old_cwd = Path.cwd()
    os.chdir(PROJECT_ROOT)
    try:
        ensure_dataset_links(args.dataset_root, args.datasets, Path.cwd())

        df = run_experiment(
            datasets=args.datasets,
            augmentation=args.augmentation,
            epochs=args.epochs,
            lr=args.lr,
            sigma_arr=[args.sigma0, args.sigma1, args.sigma2],
            base_seed=args.seed,
        )
    finally:
        os.chdir(old_cwd)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    make_plot(df, args.out_fig)

    print(f"\n[Done] wrote {args.out_csv} and {args.out_fig}")
    print(df)


if __name__ == "__main__":
    main()
