#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check required RAwR reproducibility assets.")
    p.add_argument("--require-repedges", action="store_true", help="Also require reducedNetworks/*BE*.edgelist files.")
    p.add_argument(
        "--require-competitors",
        action="store_true",
        help="Also require competitor code entrypoints (JDR and ComFy/TRIGON wrappers).",
    )
    p.add_argument(
        "--require-grain",
        action="store_true",
        help="Also require GRAIN training entrypoint.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    missing: list[Path] = []

    for dataset, eps_list in EPSILONS.items():
        dataset_dir = PROJECT_ROOT / dataset
        required = [
            dataset_dir / f"{dataset}.edgelist",
            dataset_dir / f"{dataset}.y",
        ]

        for path in required:
            if not path.exists():
                missing.append(path)

        for eps in eps_list:
            pfile = PROJECT_ROOT / "partitions" / f"{dataset}P{eps}"
            if not pfile.exists():
                missing.append(pfile)

            if args.require_repedges:
                befile = PROJECT_ROOT / "reducedNetworks" / f"{dataset}BE{eps}.edgelist"
                if not befile.exists():
                    missing.append(befile)

    if missing:
        print(f"[FAIL] Missing {len(missing)} required files:")
        for path in missing:
            print(f"  - {path}")
        raise SystemExit(1)

    if args.require_competitors:
        competitor_files = [
            REPRO_ROOT / "jdr" / "src" / "train_model.py",
            REPRO_ROOT / "scripts" / "run_compare_rewiring_methods.py",
        ]
        comp_missing = [p for p in competitor_files if not p.exists()]
        if comp_missing:
            print(f"[FAIL] Missing {len(comp_missing)} competitor entrypoints:")
            for path in comp_missing:
                print(f"  - {path}")
            raise SystemExit(1)

    if args.require_grain:
        grain_script = REPRO_ROOT / "grain" / "train.py"
        if not grain_script.exists():
            print(f"[FAIL] Missing GRAIN training script: {grain_script}")
            raise SystemExit(1)

    print("[OK] All required reproducibility assets are present.")


if __name__ == "__main__":
    main()
