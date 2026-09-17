#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

SCRIPT_DIR = Path(__file__).resolve().parent
REPRO_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = REPRO_ROOT

SECTIONS_ALL = [
    "rawr",
    "random",
    "competitors",
    "grain",
    "resistance",
    "srl",
    "srlstar",
    "teacher_student",
]


def parse_csv_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def run_cmd(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    if dry_run:
        print("DRY-RUN:", " ".join(cmd))
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run the full LoG reproducibility pipeline (or selected sections) "
            "for the RAwR submission package."
        )
    )
    p.add_argument(
        "--sections",
        type=str,
        default=",".join(SECTIONS_ALL),
        help=f"Comma-separated subset of: {','.join(SECTIONS_ALL)}",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Run a reduced smoke version (fewer trials/seeds/epochs).",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sections = parse_csv_list(args.sections)

    unknown = [s for s in sections if s not in SECTIONS_ALL]
    if unknown:
        raise ValueError(f"Unknown sections: {unknown}")

    py = sys.executable

    if "rawr" in sections:
        cmd = [
            py,
            str(REPRO_ROOT / "scripts" / "run_experiments.py"),
            "--skip-lp",
        ]
        if args.quick:
            cmd += ["--trials", "1", "--layers", "2"]
        run_cmd(cmd, cwd=PROJECT_ROOT, dry_run=args.dry_run)

    if "random" in sections:
        cmd = [
            py,
            str(REPRO_ROOT / "scripts" / "run_random_partition_baseline.py"),
            "--skip-lp",
        ]
        if args.quick:
            cmd += ["--trials", "1", "--layers", "2"]
        run_cmd(cmd, cwd=PROJECT_ROOT, dry_run=args.dry_run)

    if "competitors" in sections:
        cmd = [
            py,
            str(REPRO_ROOT / "scripts" / "run_competitor_suite.py"),
        ]
        if args.quick:
            cmd += ["--jdr-rpmax", "1", "--comfy-trigon-trials", "1"]
        run_cmd(cmd, cwd=PROJECT_ROOT, dry_run=args.dry_run)

    if "grain" in sections:
        cmd = [
            py,
            str(REPRO_ROOT / "scripts" / "run_grain_suite.py"),
        ]
        if args.quick:
            cmd += ["--seeds", "111"]
        run_cmd(cmd, cwd=PROJECT_ROOT, dry_run=args.dry_run)

    if "resistance" in sections:
        cmd = [
            py,
            str(REPRO_ROOT / "src" / "resistance.py"),
            "--output",
            str(REPRO_ROOT / "results" / "resistance_results.csv"),
        ]
        run_cmd(cmd, cwd=PROJECT_ROOT, dry_run=args.dry_run)

    if "srl" in sections:
        cmd = [
            py,
            str(REPRO_ROOT / "spectral" / "srl_main.py"),
            "--out",
            str(REPRO_ROOT / "results" / "srl_results.csv"),
        ]
        if args.quick:
            cmd += ["--datasets", "Cora", "Tree"]
        run_cmd(cmd, cwd=PROJECT_ROOT, dry_run=args.dry_run)

    if "srlstar" in sections:
        cmd = [
            py,
            str(REPRO_ROOT / "spectral" / "build_cyhmn_srlstar_table.py"),
        ]
        run_cmd(cmd, cwd=PROJECT_ROOT, dry_run=args.dry_run)

    if "teacher_student" in sections:
        cmd = [
            py,
            str(REPRO_ROOT / "spectral" / "cyhmn_teacher_student_experiment.py"),
        ]
        if args.quick:
            cmd += ["--epochs", "100"]
        run_cmd(cmd, cwd=PROJECT_ROOT, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
