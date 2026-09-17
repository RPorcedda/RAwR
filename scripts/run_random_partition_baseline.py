#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPRO_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = REPRO_ROOT
MAIN_SCRIPT = REPRO_ROOT / "src" / "main.py"
RESULTS_DIR = REPRO_ROOT / "results"
TMP_DIR = REPRO_ROOT / "tmp_random_partitions"

EPSILONS: Dict[str, List[int]] = {
    "Actor": [0, 2, 4, 8, 1303],
    "Chameleon": [0, 6, 12, 29, 732],
    "Citeseer": [0, 1, 2, 3, 99],
    "Cora": [0, 2, 3, 5, 168],
    "Cornell": [0, 1, 2, 4, 94],
    "PubMed": [0, 1, 2, 4, 171],
    "Squirrel": [0, 7, 17, 166, 1905],
    "Texas": [0, 1, 2, 3, 104],
    "Wisconsin": [0, 1, 2, 4, 122],
    "Caterpillar": [0, 1, 2, 9],
    "Grid": [0, 3, 4],
    "Ladder": [0, 2, 3, 4],
    "Line": [0, 2, 4],
    "Lobster": [0, 1, 2, 51],
    "Tree": [0, 3],
}


def parse_csv_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_csv_ints(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def read_partition_assignment(path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    with path.open() as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            rows.append([int(tok) for tok in s.replace(",", " ").split()])

    if not rows:
        raise RuntimeError(f"Empty partition file: {path}")

    # Assignment format: one integer per line.
    if all(len(r) == 1 for r in rows):
        vals = np.array([r[0] for r in rows], dtype=np.int64)
        if np.min(vals) < 0:
            vals = vals - np.min(vals)
        # Re-map to contiguous ids.
        uniq = np.unique(vals)
        remap = {int(v): i for i, v in enumerate(uniq)}
        return np.array([remap[int(v)] for v in vals], dtype=np.int64)

    # Block-list format.
    max_node = max(max(r) for r in rows)
    assignment = np.full(max_node + 1, -1, dtype=np.int64)
    for pid, block in enumerate(rows):
        for node in block:
            assignment[node] = pid

    if np.any(assignment < 0):
        missing = int(np.sum(assignment < 0))
        raise RuntimeError(f"Partition file {path} leaves {missing} nodes unassigned")
    return assignment


def build_random_assignment(reference_assignment: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = reference_assignment.shape[0]
    perm = rng.permutation(n)
    out = np.empty_like(reference_assignment)
    out[perm] = reference_assignment
    return out


def write_partition_assignment(path: Path, assignment: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for v in assignment.tolist():
            fh.write(f"{int(v)}\n")


def build_partition_edge_file(edge_path: Path, assignment: np.ndarray, out_path: Path) -> None:
    edge_set = set()
    with edge_path.open() as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            u_str, v_str, *_ = s.replace(",", " ").split()
            u = int(u_str)
            v = int(v_str)
            pu = int(assignment[u])
            pv = int(assignment[v])
            if pu == pv:
                continue
            a, b = (pu, pv) if pu < pv else (pv, pu)
            edge_set.add((a, b))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for u, v in sorted(edge_set):
            fh.write(f"{u} {v}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run random-partition baselines by shuffling node-to-partition assignments "
            "while preserving the original partition size profile."
        )
    )
    p.add_argument(
        "--datasets",
        type=str,
        default=",".join(EPSILONS.keys()),
        help="Comma-separated datasets.",
    )
    p.add_argument("--models", type=str, default="GCN,GAT,GIN")
    p.add_argument("--layers", type=str, default="2")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--features", type=str, default="true,false", help="Comma-separated true/false")
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--epochs-clf", type=int, default=500)
    p.add_argument("--skip-lp", action="store_true")
    p.add_argument("--base-seed", type=int, default=111)
    p.add_argument(
        "--augmentation",
        choices=["rep_nodes", "rep_edges", "both"],
        default="both",
        help="Which random-partition variants to run.",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=RESULTS_DIR / "random_partition_node_classification.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    datasets = parse_csv_list(args.datasets)
    models = parse_csv_list(args.models)
    layers = parse_csv_ints(args.layers)
    feature_opts = [s.strip().lower() for s in args.features.split(",") if s.strip()]
    use_features_options = [v in {"1", "true", "yes", "y"} for v in feature_opts]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    seed_counter = defaultdict(int)

    for dataset in datasets:
        if dataset not in EPSILONS:
            print(f"[WARN] dataset not in epsilon map: {dataset}, skipping")
            continue

        edge_path = PROJECT_ROOT / dataset / f"{dataset}.edgelist"
        label_path = PROJECT_ROOT / dataset / f"{dataset}.y"
        feat_path = PROJECT_ROOT / dataset / f"{dataset}.x"
        if not edge_path.exists() or not label_path.exists():
            print(f"[WARN] missing dataset files for {dataset}, skipping")
            continue

        for eps in EPSILONS[dataset]:
            ref_part = PROJECT_ROOT / "partitions" / f"{dataset}P{eps}"
            if not ref_part.exists():
                print(f"[WARN] missing partition file {ref_part}, skipping")
                continue

            ref_assign = read_partition_assignment(ref_part)

            for trial in range(1, args.trials + 1):
                rand_seed = args.base_seed + 10000 * EPSILONS[dataset].index(eps) + trial
                rand_assign = build_random_assignment(ref_assign, rand_seed)

                rand_part_file = TMP_DIR / f"{dataset}_eps{eps}_trial{trial}.P"
                rand_repedges_file = TMP_DIR / f"{dataset}_eps{eps}_trial{trial}.BE.edgelist"
                write_partition_assignment(rand_part_file, rand_assign)
                build_partition_edge_file(edge_path, rand_assign, rand_repedges_file)

                for model in models:
                    for layer in layers:
                        for use_features in use_features_options:
                            seed_counter[(dataset, model, layer, use_features)] += 1
                            train_seed = args.base_seed * seed_counter[(dataset, model, layer, use_features)]

                            base_cmd = [
                                sys.executable,
                                str(MAIN_SCRIPT),
                                "--model",
                                model,
                                "--layers",
                                str(layer),
                                "--hidden",
                                str(args.hidden),
                                "--epochs_clf",
                                str(args.epochs_clf),
                                "--edge_path",
                                str(edge_path),
                                "--label_path",
                                str(label_path),
                                "--dataset_name",
                                dataset,
                                "--random_state",
                                str(train_seed),
                                "--trial",
                                str(trial),
                                "--nc_out_path",
                                str(args.out_csv),
                                "--method_tag",
                                f"random_partition_eps{eps}_trial{trial}",
                            ]

                            if args.skip_lp:
                                base_cmd.append("--skip_lp")
                            if use_features and feat_path.exists():
                                base_cmd += ["--feat_path", str(feat_path)]

                            if args.augmentation in {"rep_nodes", "both"}:
                                cmd_nodes = base_cmd + [
                                    "--augment_partitions",
                                    "--partition_path",
                                    str(rand_part_file),
                                    "--partition_id_offset",
                                    "0",
                                ]
                                subprocess.run(cmd_nodes, check=True)

                            if args.augmentation in {"rep_edges", "both"}:
                                cmd_edges = base_cmd + [
                                    "--augment_partitions",
                                    "--connect_partition_edges",
                                    "--partition_path",
                                    str(rand_part_file),
                                    "--partition_edge_path",
                                    str(rand_repedges_file),
                                    "--partition_id_offset",
                                    "0",
                                ]
                                subprocess.run(cmd_edges, check=True)


if __name__ == "__main__":
    main()
