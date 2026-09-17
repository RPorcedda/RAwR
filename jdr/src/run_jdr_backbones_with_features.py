import argparse
import csv
import pathlib
import re
import subprocess
import sys
import threading
from typing import Dict, List, Tuple


DEFAULT_MODELS = ["GCN", "GAT", "GIN"]
DEFAULT_DATASETS = ["Actor", "Chameleon", "Citeseer", "Cora", "Cornell", "PubMed", "Squirrel", "Texas", "Wisconsin"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run JDR experiments for multiple backbones using node features (EPR/RAwR features)."
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Backbones to run.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, help="Datasets to run.")
    parser.add_argument("--rpmax", type=int, default=5, help="Number of repeated runs (RPMAX).")
    parser.add_argument("--epochs", type=int, default=10000, help="Training epochs.")
    parser.add_argument("--epr_root", type=str, default="../..", help="Path to reproducibility package root.")
    parser.add_argument("--output_dir", type=str, default="results/rawr_with_features", help="Output directory.")
    parser.add_argument("--python_exec", type=str, default=sys.executable, help="Python executable to use for train_model.py.")
    parser.add_argument("--timeout_sec", type=int, default=0, help="Per experiment timeout in seconds (0 disables timeout).")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing.")
    return parser.parse_args()


def extract_test_metrics(stdout: str) -> Tuple[float, float]:
    pattern = re.compile(r"test acc mean =\s*([0-9]*\.?[0-9]+)\s*.*test acc std =\s*([0-9]*\.?[0-9]+)")
    for line in stdout.splitlines():
        match = pattern.search(line)
        if match:
            return float(match.group(1)), float(match.group(2))
    raise ValueError("Could not parse 'test acc mean/std' from train_model.py output.")


def format_cell(mean: float, std: float) -> str:
    return f"{mean:.2f} $\\pm$ {std:.2f}"


def run_and_stream(cmd: List[str], cwd: pathlib.Path, timeout_sec: int) -> Tuple[int, str, bool]:
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    captured_lines: List[str] = []

    def _reader() -> None:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            print(line, end="", flush=True)
            captured_lines.append(line)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    timed_out = False
    try:
        proc.wait(timeout=(None if timeout_sec <= 0 else timeout_sec))
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        proc.wait()

    reader_thread.join()
    return proc.returncode, "".join(captured_lines), timed_out


def write_csv(rows: List[Dict[str, object]], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "features",
        "method",
        "dataset",
        "test_acc_mean",
        "test_acc_std",
        "rpmax",
        "status",
        "command",
        "error",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_latex(rows: List[Dict[str, object]], models: List[str], datasets: List[str], path: pathlib.Path) -> None:
    row_lookup: Dict[Tuple[str, str], Dict[str, object]] = {}
    for row in rows:
        if row["status"] == "ok":
            row_lookup[(str(row["model"]), str(row["dataset"]))] = row

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        n_dataset_cols = len(datasets)
        col_spec = "lll" + "c" * n_dataset_cols
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
        f.write("\\hline\n")
        f.write("Model & Features & Method & " + " & ".join(datasets) + " \\\\\n")
        f.write("\\hline\n")
        for model in models:
            cells = []
            for dataset in datasets:
                item = row_lookup.get((model, dataset))
                if item is None:
                    cells.append("--")
                else:
                    cells.append(format_cell(float(item["test_acc_mean"]), float(item["test_acc_std"])))
            f.write(f"{model} & Yes & JDR & " + " & ".join(cells) + " \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Test mean accuracies (\\(\\pm\\) std) for JDR with node features across backbones and datasets.}\n")
        f.write("\\label{tab:jdr_rawr_with_features_backbones}\n")
        f.write("\\end{table}\n")


def main() -> int:
    args = parse_args()
    script_dir = pathlib.Path(__file__).resolve().parent
    output_dir = script_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []
    for model in args.models:
        for dataset in args.datasets:
            cmd = [
                args.python_exec,
                "-u",
                "train_model.py",
                "--dataset",
                dataset,
                "--net",
                model,
                "--dataset_source",
                "epr_rawr",
                "--epr_root",
                args.epr_root,
                "--data_split",
                "rawr",
                "--use_epr_features",
                "Yes",
                "--denoise",
                "Yes",
                "--rewire",
                "none",
                "--use_yaml",
                "No",
                "--random_sort",
                "No",
                "--RPMAX",
                str(args.rpmax),
                "--epochs",
                str(args.epochs),
                "--no-wandb_log",
            ]
            cmd_str = " ".join(cmd)
            print(f"[RUN] {model:>3} | {dataset:<9} | {cmd_str}", flush=True)
            if args.dry_run:
                results.append(
                    {
                        "model": model,
                        "features": "Yes",
                        "method": "JDR",
                        "dataset": dataset,
                        "test_acc_mean": "",
                        "test_acc_std": "",
                        "rpmax": args.rpmax,
                        "status": "dry_run",
                        "command": cmd_str,
                        "error": "",
                    }
                )
                continue

            returncode, run_output, timed_out = run_and_stream(
                cmd=cmd,
                cwd=script_dir,
                timeout_sec=args.timeout_sec,
            )

            if timed_out:
                (output_dir / f"{model}_{dataset}.log").write_text(run_output + "\n[TIMEOUT]\n")
                results.append(
                    {
                        "model": model,
                        "features": "Yes",
                        "method": "JDR",
                        "dataset": dataset,
                        "test_acc_mean": "",
                        "test_acc_std": "",
                        "rpmax": args.rpmax,
                        "status": "timeout",
                        "command": cmd_str,
                        "error": f"timeout={args.timeout_sec}s",
                    }
                )
                print(f"[TIMEOUT] {model} {dataset}: timeout={args.timeout_sec}s", flush=True)
                continue

            (output_dir / f"{model}_{dataset}.log").write_text(run_output)
            if returncode != 0:
                results.append(
                    {
                        "model": model,
                        "features": "Yes",
                        "method": "JDR",
                        "dataset": dataset,
                        "test_acc_mean": "",
                        "test_acc_std": "",
                        "rpmax": args.rpmax,
                        "status": "failed",
                        "command": cmd_str,
                        "error": f"returncode={returncode}",
                    }
                )
                print(f"[FAIL] {model} {dataset}: returncode={returncode}", flush=True)
                continue

            try:
                mean_acc, std_acc = extract_test_metrics(run_output)
                results.append(
                    {
                        "model": model,
                        "features": "Yes",
                        "method": "JDR",
                        "dataset": dataset,
                        "test_acc_mean": mean_acc,
                        "test_acc_std": std_acc,
                        "rpmax": args.rpmax,
                        "status": "ok",
                        "command": cmd_str,
                        "error": "",
                    }
                )
                print(f"[OK]   {model} {dataset}: test={mean_acc:.2f} ± {std_acc:.2f}", flush=True)
            except Exception as exc:
                results.append(
                    {
                        "model": model,
                        "features": "Yes",
                        "method": "JDR",
                        "dataset": dataset,
                        "test_acc_mean": "",
                        "test_acc_std": "",
                        "rpmax": args.rpmax,
                        "status": "parse_failed",
                        "command": cmd_str,
                        "error": str(exc),
                    }
                )
                print(f"[WARN] {model} {dataset}: could not parse metrics ({exc})", flush=True)

    csv_path = output_dir / "jdr_with_features_backbones_results.csv"
    tex_path = output_dir / "jdr_with_features_backbones_table.tex"
    write_csv(results, csv_path)
    write_latex(results, args.models, args.datasets, tex_path)
    print(f"[DONE] CSV: {csv_path}", flush=True)
    print(f"[DONE] LaTeX: {tex_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
