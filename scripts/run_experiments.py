#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

EPSILONS = {
    "Actor": [0, 2, 4, 8, 1303],
    "Caterpillar": [0, 1, 2, 9],
    "Chameleon": [0, 6, 12, 29, 732],
    "Citeseer": [0, 1, 2, 3, 99],
    "Cora": [0, 2, 3, 5, 168],
    "Cornell": [0, 1, 2, 4, 94],
    "Grid": [0, 3, 4],
    "Ladder": [0, 2, 3, 4],
    "Line": [0, 2, 4],
    "Lobster": [0, 1, 2, 51],
    "PubMed": [0, 1, 2, 4, 171],
    "Squirrel": [0, 7, 17, 166, 1905],
    "Texas": [0, 1, 2, 3, 104],
    "Tree": [0, 3],
    "Wisconsin": [0, 1, 2, 4, 122],
}

SCRIPT_DIR = Path(__file__).resolve().parent
REPRO_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = REPRO_ROOT
MAIN_SCRIPT = REPRO_ROOT / "src" / "main.py"
DEFAULT_RESULTS = REPRO_ROOT / "results" / "node_classification_results.csv"


def parse_csv_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_csv_ints(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run RAwR node-classification sweeps across datasets/models/augmentation levels "
            "and write a single consolidated CSV."
        )
    )
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--models", type=str, default="GCN,GAT,GIN")
    p.add_argument("--layers", type=str, default="2,3,4,5,6,7,8")
    p.add_argument(
        "--datasets",
        type=str,
        default=(
            "Actor,Caterpillar,Chameleon,Citeseer,Cora,Cornell,Grid,"
            "Ladder,Line,Lobster,PubMed,Squirrel,Texas,Tree,Wisconsin"
        ),
    )
    p.add_argument("--augmentation-levels", type=str, default="0,1,2")
    p.add_argument("--feature-modes", type=str, default="false,true")
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--epochs-clf", type=int, default=500)
    p.add_argument("--epochs-lp", type=int, default=500)
    p.add_argument("--skip-lp", action="store_true")
    p.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    datasets = parse_csv_list(args.datasets)
    models = parse_csv_list(args.models)
    layers = parse_csv_ints(args.layers)
    aug_levels = parse_csv_ints(args.augmentation_levels)

    feat_tokens = {x.strip().lower() for x in args.feature_modes.split(",") if x.strip()}
    feat_modes = [False, True]
    if feat_tokens == {"false"}:
        feat_modes = [False]
    elif feat_tokens == {"true"}:
        feat_modes = [True]

    if not MAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Missing main pipeline script: {MAIN_SCRIPT}")

    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    seeds = [111 * i for i in range(1, args.trials + 1)]

    partitions_dir = PROJECT_ROOT / "partitions"
    reduced_dir = PROJECT_ROOT / "reducedNetworks"

    for model in models:
        print(f"MODEL: {model}")
        for n_layers in layers:
            print(f"LAYERS: {n_layers}")
            for dataset in datasets:
                if dataset not in EPSILONS:
                    print(f"[WARN] dataset '{dataset}' not in epsilon map, skipping")
                    continue

                edge_path = PROJECT_ROOT / dataset / f"{dataset}.edgelist"
                label_path = PROJECT_ROOT / dataset / f"{dataset}.y"
                feat_path = PROJECT_ROOT / dataset / f"{dataset}.x"

                if not edge_path.exists() or not label_path.exists():
                    print(f"[WARN] missing files for {dataset}, skipping")
                    continue

                print(f"DATASET: {dataset}")
                for use_features in feat_modes:
                    print(f"FEATURES: {use_features}")
                    for aug_level in aug_levels:
                        print(f"AUGMENTATION LEVEL: {aug_level}")

                        eps_list = [0] if aug_level == 0 else EPSILONS[dataset]
                        for eps in eps_list:
                            if aug_level != 0:
                                print(f"EPSILON: {eps}")

                            for trial in range(1, args.trials + 1):
                                seed = seeds[trial - 1]
                                print(f"TRIAL {trial} (seed={seed})")

                                cmd = [
                                    sys.executable,
                                    "-W",
                                    "ignore",
                                    str(MAIN_SCRIPT),
                                    "--trial",
                                    str(trial),
                                    "--random_state",
                                    str(seed),
                                    "--model",
                                    model,
                                    "--layers",
                                    str(n_layers),
                                    "--hidden",
                                    str(args.hidden),
                                    "--epochs_clf",
                                    str(args.epochs_clf),
                                    "--epochs_lp",
                                    str(args.epochs_lp),
                                    "--edge_path",
                                    str(edge_path),
                                    "--label_path",
                                    str(label_path),
                                    "--dataset_name",
                                    dataset,
                                    "--nc_out_path",
                                    str(args.results_csv),
                                    "--method_tag",
                                    "rawr",
                                ]

                                if use_features and feat_path.exists():
                                    cmd += ["--feat_path", str(feat_path)]

                                if args.skip_lp:
                                    cmd += ["--skip_lp"]

                                if aug_level in {1, 2}:
                                    cmd += [
                                        "--augment_partitions",
                                        "--partition_path",
                                        str(partitions_dir / f"{dataset}P{eps}"),
                                        "--partition_id_offset",
                                        "-1",
                                    ]

                                if aug_level == 2:
                                    cmd += [
                                        "--connect_partition_edges",
                                        "--partition_edge_path",
                                        str(reduced_dir / f"{dataset}BE{eps}.edgelist"),
                                    ]

                                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
