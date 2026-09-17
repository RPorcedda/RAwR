#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPRO_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = REPRO_ROOT
GRAIN_TRAIN = REPRO_ROOT / "grain" / "train.py"
RESULTS_DIR = REPRO_ROOT / "results"

ALL_DATASETS = [
    "Actor",
    "Caterpillar",
    "Chameleon",
    "Citeseer",
    "Cora",
    "Cornell",
    "Grid",
    "Ladder",
    "Line",
    "Lobster",
    "PubMed",
    "Squirrel",
    "Texas",
    "Tree",
    "Wisconsin",
]


def run_cmd(cmd: list[str], dry_run: bool) -> None:
    if dry_run:
        print("DRY-RUN:", " ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run GRAIN and GRAIN+RAwR reproducibility sweeps."
    )
    p.add_argument("--datasets", type=str, default=",".join(ALL_DATASETS))
    p.add_argument("--seeds", type=str, default="111,222,333,444,555")
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--features", choices=["yes", "no"], default="yes")
    p.add_argument("--run-original", action="store_true")
    p.add_argument("--run-repnodes", action="store_true")
    p.add_argument("--run-repedges", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not GRAIN_TRAIN.exists():
        raise FileNotFoundError(f"Missing GRAIN train script: {GRAIN_TRAIN}")

    run_modes: list[str] = []
    if args.run_original:
        run_modes.append("none")
    if args.run_repnodes:
        run_modes.append("rep_nodes")
    if args.run_repedges:
        run_modes.append("rep_edges")
    if not run_modes:
        run_modes = ["none", "rep_nodes", "rep_edges"]

    use_features = "True" if args.features == "yes" else "False"
    datasets = args.datasets
    seeds = args.seeds

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for mode in run_modes:
        out_csv = RESULTS_DIR / f"grain_{mode}_{args.features}_features.csv"
        cmd = [
            sys.executable,
            str(GRAIN_TRAIN),
            "--datasets",
            datasets,
            "--seeds",
            seeds,
            "--layers",
            str(args.layers),
            "--data_source",
            "rawr",
            "--rawr_root",
            str(REPRO_ROOT),
            "--use_features",
            use_features,
            "--rewiring_mode",
            mode,
            "--out_csv",
            str(out_csv),
        ]
        if mode != "none":
            cmd.append("--use_rawr_epsilons")

        run_cmd(cmd, args.dry_run)


if __name__ == "__main__":
    main()
