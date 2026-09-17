#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

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

SCRIPT_DIR = Path(__file__).resolve().parent
REPRO_ROOT = SCRIPT_DIR.parent
JDR_SRC = REPRO_ROOT / "jdr" / "src"
RUN_COMPARE_SCRIPT = REPRO_ROOT / "scripts" / "run_compare_rewiring_methods.py"

METHODS_ALL = ["jdr", "borf", "fosr", "sdrf", "comfy", "trigon"]


def parse_csv_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_feature_modes(value: str) -> List[bool]:
    out: List[bool] = []
    for tok in parse_csv_list(value):
        t = tok.lower()
        if t in {"1", "true", "yes", "y"}:
            out.append(True)
        elif t in {"0", "false", "no", "n"}:
            out.append(False)
        else:
            raise ValueError(f"Unsupported feature token: {tok}")
    return out


def extract_test_metrics(stdout: str) -> tuple[float, float]:
    pattern = re.compile(r"test acc mean =\s*([0-9]*\.?[0-9]+)\s*.*test acc std =\s*([0-9]*\.?[0-9]+)")
    for line in stdout.splitlines():
        match = pattern.search(line)
        if match:
            return float(match.group(1)), float(match.group(2))
    raise ValueError("Could not parse test mean/std from train_model.py output")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run competitor rewiring methods used in the paper as peer methods: "
            "JDR, BORF, FOSR, SDRF, ComFy, TRIGON."
        )
    )
    p.add_argument(
        "--methods",
        type=str,
        default=",".join(METHODS_ALL),
        help=f"Comma-separated methods from {{{','.join(METHODS_ALL)}}}.",
    )
    p.add_argument("--datasets", type=str, default=",".join(ALL_DATASETS))
    p.add_argument("--models", type=str, default="GCN,GAT,GIN")
    p.add_argument("--feature-modes", type=str, default="true,false")

    p.add_argument("--jdr-rpmax", type=int, default=5)
    p.add_argument("--jdr-epochs", type=int, default=10000)
    p.add_argument("--jdr-timeout-sec", type=int, default=0)

    p.add_argument("--comfy-trigon-trials", type=int, default=5)
    p.add_argument("--comfy-trigon-layers", type=int, default=2)
    p.add_argument("--comfy-trigon-hidden", type=int, default=16)
    p.add_argument("--comfy-trigon-epochs-clf", type=int, default=500)
    p.add_argument("--comfy-trigon-force-rewire", action="store_true")

    p.add_argument(
        "--out-csv",
        type=Path,
        default=REPRO_ROOT / "results" / "competitors_rewiring_methods.csv",
    )
    p.add_argument(
        "--comfy-trigon-raw-out-csv",
        type=Path,
        default=REPRO_ROOT / "results" / "competitors_comfy_trigon_raw_node_classification.csv",
    )
    p.add_argument(
        "--comfy-trigon-lp-out-csv",
        type=Path,
        default=REPRO_ROOT / "results" / "competitors_comfy_trigon_link_prediction.csv",
    )
    p.add_argument(
        "--rewired-dir",
        type=Path,
        default=REPRO_ROOT / "results" / "rewiredNetworks_baselines",
        help="Directory used to cache ComFy/TRIGON rewired edge lists.",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _mean_std(values: List[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot compute mean/std of empty values")
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def run_methods_via_jdr_stack(
    methods: List[str],
    datasets: List[str],
    models: List[str],
    feature_modes: List[bool],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    selected = [m for m in methods if m in {"jdr", "borf", "fosr", "sdrf"}]
    if not selected:
        return []

    if not JDR_SRC.exists():
        raise FileNotFoundError(f"Missing JDR source directory: {JDR_SRC}")

    rows: list[dict[str, object]] = []

    for method in selected:
        for model in models:
            for dataset in datasets:
                for use_features in feature_modes:
                    feat_token = "Yes" if use_features else "No"
                    base_cmd = [
                        sys.executable,
                        "-u",
                        "train_model.py",
                        "--dataset",
                        dataset,
                        "--net",
                        model,
                        "--dataset_source",
                        "epr_rawr",
                        "--epr_root",
                        str(REPRO_ROOT),
                        "--data_split",
                        "rawr",
                        "--random_sort",
                        "No",
                        "--use_epr_features",
                        feat_token,
                        "--RPMAX",
                        str(args.jdr_rpmax),
                        "--epochs",
                        str(args.jdr_epochs),
                        "--use_yaml",
                        "No",
                        "--no-wandb_log",
                    ]

                    if method == "jdr":
                        extra = [
                            "--denoise",
                            "Yes",
                            "--denoise_default",
                            model,
                            "--rewire",
                            "none",
                        ]
                    elif method == "borf":
                        extra = ["--denoise", "No", "--rewire", "borf"]
                    elif method == "fosr":
                        extra = ["--denoise", "No", "--rewire", "fosr"]
                    else:  # sdrf
                        extra = ["--denoise", "No", "--rewire", "sdrf"]

                    cmd = base_cmd + extra
                    cmd_str = " ".join(cmd)
                    print(f"[REWIRING] method={method} model={model} dataset={dataset} features={feat_token}")

                    if args.dry_run:
                        rows.append(
                            {
                                "method": method,
                                "model": model,
                                "dataset": dataset,
                                "features": feat_token,
                                "status": "dry_run",
                                "test_acc_mean": "",
                                "test_acc_std": "",
                                "num_trials": "",
                                "source_backend": "jdr_stack",
                                "command": cmd_str,
                                "error": "",
                            }
                        )
                        continue

                    try:
                        completed = subprocess.run(
                            cmd,
                            cwd=JDR_SRC,
                            capture_output=True,
                            text=True,
                            timeout=(None if args.jdr_timeout_sec <= 0 else args.jdr_timeout_sec),
                            check=False,
                        )
                    except subprocess.TimeoutExpired:
                        rows.append(
                            {
                                "method": method,
                                "model": model,
                                "dataset": dataset,
                                "features": feat_token,
                                "status": "timeout",
                                "test_acc_mean": "",
                                "test_acc_std": "",
                                "num_trials": "",
                                "source_backend": "jdr_stack",
                                "command": cmd_str,
                                "error": f"timeout={args.jdr_timeout_sec}s",
                            }
                        )
                        continue

                    if completed.returncode != 0:
                        rows.append(
                            {
                                "method": method,
                                "model": model,
                                "dataset": dataset,
                                "features": feat_token,
                                "status": "failed",
                                "test_acc_mean": "",
                                "test_acc_std": "",
                                "num_trials": "",
                                "source_backend": "jdr_stack",
                                "command": cmd_str,
                                "error": f"returncode={completed.returncode}",
                            }
                        )
                        continue

                    try:
                        mean_acc, std_acc = extract_test_metrics(completed.stdout)
                        rows.append(
                            {
                                "method": method,
                                "model": model,
                                "dataset": dataset,
                                "features": feat_token,
                                "status": "ok",
                                "test_acc_mean": mean_acc,
                                "test_acc_std": std_acc,
                                "num_trials": args.jdr_rpmax,
                                "source_backend": "jdr_stack",
                                "command": cmd_str,
                                "error": "",
                            }
                        )
                    except Exception as exc:  # noqa: BLE001
                        rows.append(
                            {
                                "method": method,
                                "model": model,
                                "dataset": dataset,
                                "features": feat_token,
                                "status": "parse_failed",
                                "test_acc_mean": "",
                                "test_acc_std": "",
                                "num_trials": "",
                                "source_backend": "jdr_stack",
                                "command": cmd_str,
                                "error": str(exc),
                            }
                        )

    return rows


def _method_from_tag(method_tag: str) -> str | None:
    t = method_tag.lower()
    if t.startswith("comfy"):
        return "comfy"
    if t.startswith("trigon"):
        return "trigon"
    return None


def _feat_bool_to_yes_no(value: str) -> str:
    v = value.strip().lower()
    if v in {"true", "1", "yes", "y"}:
        return "Yes"
    if v in {"false", "0", "no", "n"}:
        return "No"
    return value


def run_comfy_trigon(
    methods: List[str],
    datasets: List[str],
    models: List[str],
    feature_modes: List[bool],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    selected = [m for m in methods if m in {"comfy", "trigon"}]
    if not selected:
        return []

    if not RUN_COMPARE_SCRIPT.exists():
        raise FileNotFoundError(f"Missing script: {RUN_COMPARE_SCRIPT}")

    method_csv = ",".join(selected)
    dataset_csv = ",".join(datasets)
    model_csv = ",".join(models)
    feature_csv = ",".join("True" if f else "False" for f in feature_modes)

    cmd = [
        sys.executable,
        str(RUN_COMPARE_SCRIPT),
        "--methods",
        method_csv,
        "--datasets",
        dataset_csv,
        "--models",
        model_csv,
        "--use_features_options",
        feature_csv,
        "--trials",
        str(args.comfy_trigon_trials),
        "--layers",
        str(args.comfy_trigon_layers),
        "--hidden",
        str(args.comfy_trigon_hidden),
        "--epochs_clf",
        str(args.comfy_trigon_epochs_clf),
        "--skip_lp",
        "--rewired_dir",
        str(args.rewired_dir),
        "--nc_out",
        str(args.comfy_trigon_raw_out_csv),
        "--lp_out",
        str(args.comfy_trigon_lp_out_csv),
    ]

    if args.comfy_trigon_force_rewire:
        cmd.append("--force_rewire")

    print(f"[REWIRING] methods={method_csv} datasets={dataset_csv} (comfy/trigon stack)")
    if args.dry_run:
        print("DRY-RUN:", " ".join(cmd))
        return []

    subprocess.run(cmd, cwd=REPRO_ROOT, check=True)

    grouped: Dict[tuple[str, str, str, str], List[float]] = defaultdict(list)
    with args.comfy_trigon_raw_out_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = _method_from_tag(str(row.get("method_tag", "")))
            if method is None or method not in selected:
                continue
            model = str(row.get("model", ""))
            dataset = str(row.get("dataset", ""))
            features = _feat_bool_to_yes_no(str(row.get("feat_presence", "")))
            key = (method, model, dataset, features)
            grouped[key].append(float(row["test_acc"]))

    rows: list[dict[str, object]] = []
    for (method, model, dataset, features), values in sorted(grouped.items()):
        mean_acc, std_acc = _mean_std(values)
        rows.append(
            {
                "method": method,
                "model": model,
                "dataset": dataset,
                "features": features,
                "status": "ok",
                "test_acc_mean": mean_acc,
                "test_acc_std": std_acc,
                "num_trials": len(values),
                "source_backend": "comfy_trigon_stack",
                "command": " ".join(cmd),
                "error": "",
            }
        )

    print(f"[REWIRING] wrote raw comfy/trigon rows: {args.comfy_trigon_raw_out_csv}")
    return rows


def main() -> None:
    args = parse_args()

    methods = [m.lower() for m in parse_csv_list(args.methods)]
    datasets = parse_csv_list(args.datasets)
    models = parse_csv_list(args.models)
    feature_modes = parse_feature_modes(args.feature_modes)

    unsupported = [m for m in methods if m not in set(METHODS_ALL)]
    if unsupported:
        raise ValueError(f"Unsupported methods requested: {unsupported}")

    all_rows: list[dict[str, object]] = []
    all_rows.extend(run_methods_via_jdr_stack(methods, datasets, models, feature_modes, args))
    all_rows.extend(run_comfy_trigon(methods, datasets, models, feature_modes, args))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "model",
                "dataset",
                "features",
                "status",
                "test_acc_mean",
                "test_acc_std",
                "num_trials",
                "source_backend",
                "command",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[REWIRING] wrote unified results: {args.out_csv}")


if __name__ == "__main__":
    main()
