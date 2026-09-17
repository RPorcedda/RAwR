# srl_main.py
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

from srl_utils import compute_srl, slice_all_eigs_adaptive, SliceParams
from utils import *

dataset_dict = {
    "Actor":      [0, 2, 4, 8, 1303],
    "Caterpillar": [0, 1, 2, 9],
    "Chameleon":  [0, 6, 12, 29, 732],
    "Citeseer":   [0, 1, 2, 3, 99],
    "Cora":       [0, 2, 3, 5, 168],
    "Cornell":    [0, 1, 2, 4, 94],
    "Grid": [0, 3, 4],
    "Ladder": [0, 2, 3, 4],
    "Line": [0, 2, 4],
    "Lobster": [0, 1, 2, 51],
    "PubMed":     [0, 1, 2, 4, 171],
    "Squirrel":   [0, 7, 17, 166, 1905],
    "Texas":      [0, 1, 2, 3, 104],
    "Tree": [0, 3],
    "Wisconsin":  [0, 1, 2, 4, 122]
}

AUGMENTATIONS = ["RepNodes", "RepEdges"]
THIS_DIR = Path(__file__).resolve().parent
REPRO_ROOT = THIS_DIR.parent
PROJECT_ROOT = REPRO_ROOT
DEFAULT_OUT = REPRO_ROOT / "results" / "srl_results.csv"
DEFAULT_EIG_OUT = REPRO_ROOT / "results" / "eig_out"

def run_srl_for_dataset(dataset: str,
                        mode: str,
                        save_eigs: bool,
                        eig_outdir: Path,
                        eig_windows: int,
                        eig_overlap: float,
                        eig_tol: float,
                        materialize: bool,
                        lin_solve_tol: float,
                        lin_solve_maxiter: int) -> pd.DataFrame:

    print(f"\nDATASET: {dataset}")
    y = read_labels(f"{dataset}/{dataset}.y")     # (n,1) or (n,)
    y = y.reshape(-1)
    num_classes = infer_num_classes(y)

    rows = []
    for augmentation in AUGMENTATIONS:
        for eps in dataset_dict[dataset]:
            A, R, Q = get_adjacency(dataset, augmentation, eps, return_blocks=True)
            if not sp.isspmatrix_csr(A): A = sp.csr_matrix(A)
            if not sp.isspmatrix_csr(R): R = sp.csr_matrix(R)
            if not sp.isspmatrix_csr(Q): Q = sp.csr_matrix(Q)
            print("Computing SRL")
            res = compute_srl(
                A, y, R, Q,
                mode=mode,
                eig_windows=eig_windows,
                eig_overlap=eig_overlap,
                eig_tol=eig_tol
            )

            row = {
                "dataset": dataset,
                "augmentation": augmentation,
                "eps": eps,
                "n": A.shape[0],
                "m": int(A.nnz),
                "k_roles": int(R.shape[1]),
                "rho": res["rho"],
                "rho_corr": res["rho_corr"],
                "R_obs": res["R_obs"],
                "delta_plus": res["delta_plus"],
                "delta_neg": res["delta_neg"],
                "srl_plus": res["srl_plus"],
                "srl_plus_corr": res["srl_plus_corr"],
                "srl_neg": res["srl_neg"],
                "srl_neg_corr": res["srl_neg_corr"],
                "srl_total": res["srl_total"],
                "srl_total_corr": res["srl_total_corr"],
                "comm_norm": res["comm_norm"]
            }
            rows.append(row)

    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(description="SRL sweep")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--datasets", type=str, nargs="*", default=None)
    ap.add_argument("--mode", choices=["exact2x2","approx_eigs"], default="exact2x2")
    ap.add_argument("--save-eigs", action="store_true")
    ap.add_argument("--eig-outdir", type=Path, default=DEFAULT_EIG_OUT)
    ap.add_argument("--eig-windows", type=int, default=24)
    ap.add_argument("--eig-overlap", type=float, default=0.01)
    ap.add_argument("--eig-tol", type=float, default=1e-6)
    ap.add_argument("--dtype", choices=["float32","float64"], default="float32")
    ap.add_argument("--materialize", action="store_true",
                    help="Se presente, usa CSR+LU (memoria alta). Di default: operator+MINRES (RAM bassa).")
    ap.add_argument("--lin-solve-tol", type=float, default=1e-4,
                    help="Tolleranza MINRES per (S-sigma I)z=v (solo operator mode).")
    ap.add_argument("--lin-solve-maxiter", type=int, default=500,
                    help="Max iter MINRES (solo operator mode).")
    args = ap.parse_args()


    frames = []
    old_cwd = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT)
        run_datasets = args.datasets if args.datasets else list(dataset_dict.keys())
        for ds in run_datasets:
            if ds not in dataset_dict:
                print(f"[WARN] dataset {ds} not in epsilon map; skipping")
                continue
            print("Vai con run_srl_for_dataset")
            df = run_srl_for_dataset(
                ds, mode=args.mode, save_eigs=args.save_eigs, eig_outdir=args.eig_outdir,
                eig_windows=args.eig_windows, eig_overlap=args.eig_overlap, eig_tol=args.eig_tol,
                materialize=args.materialize,
                lin_solve_tol=args.lin_solve_tol, lin_solve_maxiter=args.lin_solve_maxiter
            )
            frames.append(df)
    finally:
        os.chdir(old_cwd)

    out = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[SRL] wrote {args.out} ({len(out)} rows)")

if __name__ == "__main__":
    main()
