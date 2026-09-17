from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

DATASETS = [
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

EPSILONS: Dict[str, List[int]] = {
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

AUGMENTATION_NAME = {
    0: "None",
    1: "RepNodes",
    2: "RepEdges",
}
ALL_AUG_LEVELS = [0, 1, 2]

SRC_DIR = Path(__file__).resolve().parent
REPRO_ROOT = SRC_DIR.parent
PROJECT_ROOT = REPRO_ROOT

EPS_BE_SCRIPT = SRC_DIR / "epsBEPython.py"
DEFAULT_PARTITION_DIR = PROJECT_ROOT / "partitions"
DEFAULT_REDUCED_DIR = PROJECT_ROOT / "reducedNetworks"
DEFAULT_OUTPUT_CSV = REPRO_ROOT / "results" / "resistance_results.csv"


def _load_partitions(
    partition_path: Path,
    id_offset: int,
    expected_num_nodes: int | None = None,
) -> Tuple[Dict[str, int], int]:
    rows: List[List[int]] = []
    with partition_path.expanduser().open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append([int(tok) for tok in line.replace(",", " ").split()])

    # Support both partition encodings that appear in this repo:
    # 1) block-list format: each line lists node ids that belong to that block
    # 2) assignment format: one partition id per line (line index is node id)
    is_assignment_format = False
    if expected_num_nodes is not None and len(rows) == expected_num_nodes and rows:
        if all(len(r) == 1 for r in rows):
            vals = [r[0] for r in rows]
            # In assignment format partition ids repeat across nodes.
            # Singleton block-list files have one token per line too, but
            # their node ids are typically all unique.
            if len(set(vals)) < expected_num_nodes:
                is_assignment_format = True

    if is_assignment_format:
        vals = [r[0] for r in rows]
        if min(vals) < 0:
            raise ValueError(
                f"Invalid assignment partition file {partition_path}: negative partition id found."
            )
        uniq = sorted(set(vals))
        pid_remap = {pid: i for i, pid in enumerate(uniq)}
        part_map = {str(node_idx): pid_remap[pid] for node_idx, pid in enumerate(vals)}
        return part_map, len(uniq)

    part_map: Dict[str, int] = {}
    k = 0
    for toks in rows:
        for tok in toks:
            part_map[str(tok + id_offset)] = k
        k += 1
    return part_map, k


def _read_edgelist(edge_path: Path) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    with edge_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            u_str, v_str, *_ = line.replace(",", " ").split()
            edges.append((int(u_str), int(v_str)))
    return edges


def _num_nodes_from_labels(label_path: Path) -> int:
    labels = pd.read_csv(label_path, header=None)
    return int(labels.shape[0])


def _build_original_adjacency(edge_path: Path, num_nodes: int) -> sp.csr_matrix:
    edges = _read_edgelist(edge_path)
    if not edges:
        return sp.csr_matrix((num_nodes, num_nodes), dtype=np.float64)

    edge_arr = np.asarray(edges, dtype=np.int64)
    src = edge_arr[:, 0]
    dst = edge_arr[:, 1]

    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.ones(rows.size, dtype=np.float64)

    adj = sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float64)
    adj.setdiag(0)
    adj.eliminate_zeros()
    if adj.nnz:
        adj.data[:] = 1.0
    return adj


def _build_partition_incidence(
    n_old: int,
    partition_map: Dict[str, int],
    num_partitions: int,
    allow_orphan_singletons: bool,
) -> Tuple[sp.csr_matrix, int]:
    node_part = np.full(n_old, -1, dtype=np.int64)
    for orig_id_str, pid in partition_map.items():
        idx = int(orig_id_str)
        if 0 <= idx < n_old:
            node_part[idx] = pid

    missing = np.flatnonzero(node_part < 0)
    if missing.size > 0:
        if not allow_orphan_singletons:
            raise ValueError(
                f"{missing.size} nodes are missing from the partition map. "
                "Use --allow-orphan-singletons to assign them singleton partitions."
            )
        start_pid = num_partitions
        node_part[missing] = np.arange(start_pid, start_pid + missing.size, dtype=np.int64)
        num_partitions += int(missing.size)

    rows = np.arange(n_old, dtype=np.int64)
    cols = node_part
    data = np.ones(n_old, dtype=np.float64)
    B = sp.csr_matrix((data, (rows, cols)), shape=(n_old, num_partitions), dtype=np.float64)
    return B, num_partitions


def _read_partition_edge_adjacency(
    partition_edge_path: Path,
    num_partitions: int,
    edge_id_offset: int,
) -> sp.csr_matrix:
    pairs: List[Tuple[int, int]] = []
    with partition_edge_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            u_str, v_str, *_ = line.replace(",", " ").split()
            u = int(u_str) + edge_id_offset
            v = int(v_str) + edge_id_offset
            pairs.append((u, v))

    if not pairs:
        return sp.csr_matrix((num_partitions, num_partitions), dtype=np.float64)

    pair_arr = np.asarray(pairs, dtype=np.int64)
    src = pair_arr[:, 0]
    dst = pair_arr[:, 1]

    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.ones(rows.size, dtype=np.float64)

    R = sp.csr_matrix((data, (rows, cols)), shape=(num_partitions, num_partitions), dtype=np.float64)
    R.setdiag(0)
    R.eliminate_zeros()
    if R.nnz:
        R.data[:] = 1.0
    return R


def _infer_partition_edges_from_original(adj_old: sp.csr_matrix, B: sp.csr_matrix) -> sp.csr_matrix:
    R = (B.T @ adj_old @ B).tocsr()
    R.setdiag(0)
    R.eliminate_zeros()
    if R.nnz:
        R.data[:] = 1.0
    return R


def _normalize_adjacency(adj: sp.csr_matrix) -> sp.csr_matrix:
    adj = adj.tocsr(copy=True)
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj = adj.maximum(adj.T)
    if adj.nnz:
        adj.data[:] = 1.0
    return adj


def _ensure_partition_file(
    dataset: str,
    eps: int,
    num_nodes: int,
    partition_dir: Path,
    generate_missing_partitions: bool,
) -> Path:
    partition_path = partition_dir / f"{dataset}P{eps}"
    if partition_path.exists():
        return partition_path

    if not generate_missing_partitions:
        raise FileNotFoundError(
            f"Missing partition file: {partition_path} (needed for eps={eps})"
        )

    if not EPS_BE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Cannot generate missing partition: script not found at {EPS_BE_SCRIPT}"
        )

    print(
        f"    [INFO] Missing partition file {partition_path}. "
        "Generating it with epsBEPython.py..."
    )

    DEFAULT_PARTITION_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(EPS_BE_SCRIPT),
        "--filename",
        dataset,
        "--num_nodes",
        str(num_nodes),
        "--eps",
        str(eps),
    ]

    try:
        subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "epsBEPython.py failed while generating a partition file.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        ) from exc

    generated_path = DEFAULT_PARTITION_DIR / f"{dataset}P{eps}"
    if not generated_path.exists():
        raise FileNotFoundError(
            f"epsBEPython.py completed but did not create expected file: {generated_path}"
        )

    if partition_path.resolve() != generated_path.resolve():
        partition_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generated_path, partition_path)

    return partition_path


def build_adjacency_for_setting(
    dataset: str,
    aug_level: int,
    eps: int,
    partition_dir: Path,
    reduced_dir: Path,
    partition_id_offset: int,
    partition_edge_id_offset: int,
    allow_orphan_singletons: bool,
    infer_partition_edges_if_missing: bool,
    generate_missing_partitions: bool,
) -> Tuple[sp.csr_matrix, int]:
    edge_path = PROJECT_ROOT / dataset / f"{dataset}.edgelist"
    label_path = PROJECT_ROOT / dataset / f"{dataset}.y"

    if not edge_path.exists():
        raise FileNotFoundError(f"Missing edge file: {edge_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label file: {label_path}")

    n_old = _num_nodes_from_labels(label_path)
    adj_old = _build_original_adjacency(edge_path, n_old)

    if aug_level == 0:
        return _normalize_adjacency(adj_old), n_old

    partition_path = _ensure_partition_file(
        dataset=dataset,
        eps=eps,
        num_nodes=n_old,
        partition_dir=partition_dir,
        generate_missing_partitions=generate_missing_partitions,
    )

    part_map, k = _load_partitions(
        partition_path,
        id_offset=partition_id_offset,
        expected_num_nodes=n_old,
    )
    B, k = _build_partition_incidence(
        n_old,
        part_map,
        k,
        allow_orphan_singletons=allow_orphan_singletons,
    )

    if aug_level == 1:
        R = sp.csr_matrix((k, k), dtype=np.float64)
    elif aug_level == 2:
        partition_edge_path = reduced_dir / f"{dataset}BE{eps}.edgelist"
        if partition_edge_path.exists():
            R = _read_partition_edge_adjacency(
                partition_edge_path,
                num_partitions=k,
                edge_id_offset=partition_edge_id_offset,
            )
        else:
            if not infer_partition_edges_if_missing:
                raise FileNotFoundError(
                    f"Missing partition-edge file: {partition_edge_path}"
                )
            R = _infer_partition_edges_from_original(adj_old, B)
    else:
        raise ValueError("aug_level must be one of {0,1,2}")

    adj_aug = sp.bmat(
        [[adj_old, B], [B.T, R]],
        format="csr",
        dtype=np.float64,
    )
    return _normalize_adjacency(adj_aug), n_old


def _laplacian_from_adjacency(adj: sp.csr_matrix) -> sp.csr_matrix:
    deg = np.asarray(adj.sum(axis=1)).reshape(-1)
    return sp.diags(deg, offsets=0, format="csr") - adj


def _total_resistance_from_pinv_submatrix(L_plus_sub: np.ndarray) -> float:
    m = int(L_plus_sub.shape[0])
    if m <= 1:
        return 0.0

    ones = np.ones(m, dtype=np.float64)
    total = float(m * np.trace(L_plus_sub) - ones @ L_plus_sub @ ones)
    return max(total, 0.0)


def _resolve_topk_pair_count(
    all_pairs: int,
    topk_pairs: int | None,
    topk_fraction: float | None,
) -> int:
    if all_pairs < 0:
        raise ValueError("all_pairs must be non-negative")
    if all_pairs == 0:
        return 0

    if topk_pairs is not None and topk_fraction is not None:
        raise ValueError("Use either --topk-pairs or --topk-fraction, not both.")

    if topk_pairs is None and topk_fraction is None:
        topk_fraction = 0.1

    if topk_pairs is not None:
        if topk_pairs <= 0:
            raise ValueError("--topk-pairs must be > 0")
        return min(int(topk_pairs), all_pairs)

    assert topk_fraction is not None
    if not (0.0 < topk_fraction <= 1.0):
        raise ValueError("--topk-fraction must be in (0, 1].")
    return min(max(int(all_pairs * topk_fraction), 1), all_pairs)


def _topk_mean_effective_resistance_from_pinv(
    L_plus_sub: np.ndarray,
    topk_pairs: int,
) -> float:
    m = int(L_plus_sub.shape[0])
    all_pairs = m * (m - 1) // 2
    if all_pairs == 0 or topk_pairs <= 0:
        return float("nan")

    k = int(min(topk_pairs, all_pairs))
    if k == all_pairs:
        total = _total_resistance_from_pinv_submatrix(L_plus_sub)
        return float(total / all_pairs)

    diag = np.diag(L_plus_sub).astype(np.float64, copy=False)
    top_vals = np.empty(0, dtype=np.float64)

    for i in range(m - 1):
        row_vals = diag[i] + diag[i + 1 :] - 2.0 * L_plus_sub[i, i + 1 :]
        row_vals = np.maximum(row_vals, 0.0)

        if row_vals.size > k:
            row_vals = np.partition(row_vals, -k)[-k:]

        if top_vals.size == 0:
            top_vals = row_vals
        else:
            merged = np.concatenate([top_vals, row_vals])
            if merged.size > k:
                top_vals = np.partition(merged, -k)[-k:]
            else:
                top_vals = merged

    if top_vals.size < k:
        k = int(top_vals.size)
        if k == 0:
            return float("nan")

    return float(np.mean(top_vals))


def compute_resistance_metrics(
    adj: sp.csr_matrix,
    num_observed_nodes: int,
    laplacian_mode: str,
    mean_aggregation: str,
    topk_pairs: int | None,
    topk_fraction: float | None,
) -> Dict[str, float | int | str | None]:
    adj = _normalize_adjacency(adj)

    n_nodes = int(adj.shape[0])
    n_edges = int(adj.nnz // 2)
    n_observed = int(min(num_observed_nodes, n_nodes))
    n_virtual = int(max(0, n_nodes - n_observed))
    all_pairs = int(n_observed * (n_observed - 1) // 2)

    if laplacian_mode == "observed":
        adj_for_laplacian = adj[:n_observed, :n_observed].tocsr()
        observed_idx = np.arange(n_observed, dtype=np.int64)
    elif laplacian_mode == "full":
        adj_for_laplacian = adj
        observed_idx = np.arange(n_observed, dtype=np.int64)
    else:
        raise ValueError("laplacian_mode must be one of {'observed', 'full'}")

    L = _laplacian_from_adjacency(adj_for_laplacian)
    L_plus = np.linalg.pinv(L.toarray(), hermitian=True)
    L_plus_obs = L_plus[np.ix_(observed_idx, observed_idx)]
    total_resistance = _total_resistance_from_pinv_submatrix(L_plus_obs)

    mean_all_pairs = float(total_resistance / all_pairs) if all_pairs > 0 else float("nan")

    selected_pairs = all_pairs
    selected_pairs_fraction = 1.0 if all_pairs > 0 else float("nan")
    topk_pairs_used: int | None = None
    effective_topk_fraction: float | None = None

    if mean_aggregation == "all_pairs":
        mean_effective = mean_all_pairs
    elif mean_aggregation == "topk":
        selected_pairs = _resolve_topk_pair_count(
            all_pairs=all_pairs,
            topk_pairs=topk_pairs,
            topk_fraction=topk_fraction,
        )
        topk_pairs_used = int(selected_pairs)
        if all_pairs > 0:
            selected_pairs_fraction = float(selected_pairs / all_pairs)
            effective_topk_fraction = selected_pairs_fraction
        else:
            selected_pairs_fraction = float("nan")
            effective_topk_fraction = None

        if all_pairs == 0:
            mean_effective = float("nan")
        elif selected_pairs == all_pairs:
            mean_effective = mean_all_pairs
        else:
            mean_effective = _topk_mean_effective_resistance_from_pinv(
                L_plus_sub=L_plus_obs,
                topk_pairs=selected_pairs,
            )
    else:
        raise ValueError("mean_aggregation must be one of {'all_pairs', 'topk'}")

    return {
        "num_nodes": n_nodes,
        "num_observed_nodes": n_observed,
        "num_virtual_nodes": n_virtual,
        "num_edges": n_edges,
        "num_components": None,
        "connected_pairs": int(all_pairs),
        "selected_pairs": int(selected_pairs),
        "selected_pairs_fraction": selected_pairs_fraction,
        "mean_aggregation": mean_aggregation,
        "topk_pairs_used": topk_pairs_used,
        "topk_fraction_used": effective_topk_fraction,
        "mean_effective_resistance_all_pairs": mean_all_pairs,
        "mean_effective_resistance": mean_effective,
        "total_resistance": float(total_resistance),
        "exact_components": 1,
        "approx_components": 0,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute mean effective resistance and total resistance for original and "
            "rewired graphs (RepNodes/RepEdges)."
        )
    )

    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument(
        "--augmentation-levels",
        nargs="+",
        type=int,
        default=None,
        help="Augmentation levels to run. If omitted, runs all: 0 1 2.",
    )

    p.add_argument("--partition-dir", type=Path, default=DEFAULT_PARTITION_DIR)
    p.add_argument("--reduced-dir", type=Path, default=DEFAULT_REDUCED_DIR)
    p.add_argument("--partition-id-offset", type=int, default=-1)
    p.add_argument("--partition-edge-id-offset", type=int, default=0)
    p.add_argument(
        "--generate-missing-partitions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If partitions/<dataset>P<eps> is missing, generate it by calling epsBEPython.py."
        ),
    )

    p.add_argument("--allow-orphan-singletons", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--strict-partitions", action="store_true", default=False)
    p.add_argument(
        "--infer-partition-edges-if-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If a reducedNetworks/<dataset>BE<eps>.edgelist file is missing, infer "
            "RepEdges block connectivity from the original graph and partition incidence."
        ),
    )
    p.add_argument(
        "--laplacian-mode",
        choices=["observed", "full"],
        default="observed",
        help=(
            "How to build L before pinv(L). "
            "'observed': only real-node subgraph. "
            "'full': full augmented graph (real+virtual), but resistances are still "
            "aggregated only across real-node pairs."
        ),
    )
    p.add_argument(
        "--mean-aggregation",
        choices=["all_pairs", "topk"],
        default="all_pairs",
        help=(
            "How to compute mean_effective_resistance. "
            "'all_pairs': average over all observed-node pairs. "
            "'topk': average over the top-K largest observed-node pair resistances "
            "(TRIGON-like)."
        ),
    )
    p.add_argument(
        "--topk-pairs",
        type=int,
        default=None,
        help=(
            "When --mean-aggregation=topk, use this many largest pairwise effective "
            "resistances."
        ),
    )
    p.add_argument(
        "--topk-fraction",
        type=float,
        default=None,
        help=(
            "When --mean-aggregation=topk, use this top fraction of pairwise "
            "effective resistances. If neither topk option is set, defaults to 0.1."
        ),
    )
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.strict_partitions:
        allow_orphan_singletons = False
    else:
        allow_orphan_singletons = bool(args.allow_orphan_singletons)

    infer_partition_edges_if_missing = bool(args.infer_partition_edges_if_missing)
    augmentation_levels = ALL_AUG_LEVELS if args.augmentation_levels is None else args.augmentation_levels
    augmentation_levels = list(dict.fromkeys(augmentation_levels))

    if args.mean_aggregation == "all_pairs":
        if args.topk_pairs is not None or args.topk_fraction is not None:
            raise ValueError(
                "--topk-pairs/--topk-fraction are valid only with --mean-aggregation=topk."
            )
    elif args.mean_aggregation == "topk":
        if args.topk_pairs is not None and args.topk_fraction is not None:
            raise ValueError("Use either --topk-pairs or --topk-fraction, not both.")
        if args.topk_pairs is not None and args.topk_pairs <= 0:
            raise ValueError("--topk-pairs must be > 0")
        if args.topk_fraction is not None and not (0.0 < args.topk_fraction <= 1.0):
            raise ValueError("--topk-fraction must be in (0, 1].")

    rows = []

    for dataset in args.datasets:
        if dataset not in EPSILONS:
            print(f"[WARN] Dataset '{dataset}' not in EPSILONS map: skipping")
            continue

        print(f"\nDATASET: {dataset}")
        baseline_total_resistance: float | None = None
        baseline_metrics: Dict[str, float | int | str | None] | None = None
        try:
            baseline_adj, baseline_n_observed = build_adjacency_for_setting(
                dataset=dataset,
                aug_level=0,
                eps=0,
                partition_dir=args.partition_dir,
                reduced_dir=args.reduced_dir,
                partition_id_offset=args.partition_id_offset,
                partition_edge_id_offset=args.partition_edge_id_offset,
                allow_orphan_singletons=allow_orphan_singletons,
                infer_partition_edges_if_missing=infer_partition_edges_if_missing,
                generate_missing_partitions=args.generate_missing_partitions,
            )
            baseline_metrics = compute_resistance_metrics(
                baseline_adj,
                num_observed_nodes=baseline_n_observed,
                laplacian_mode=args.laplacian_mode,
                mean_aggregation=args.mean_aggregation,
                topk_pairs=args.topk_pairs,
                topk_fraction=args.topk_fraction,
            )
            baseline_total_resistance = float(baseline_metrics["total_resistance"])
            print(
                f"  original baseline (aug_level=0, eps=0): "
                f"total_resistance={baseline_total_resistance:.6f}"
            )
        except Exception as exc:
            print(
                "  [WARN] Could not compute original baseline "
                f"(aug_level=0, eps=0): {exc}"
            )

        for aug_level in augmentation_levels:
            if aug_level not in AUGMENTATION_NAME:
                print(f"[WARN] Unsupported aug_level={aug_level}: skipping")
                continue

            eps_list = [0] if aug_level == 0 else EPSILONS[dataset]

            for eps in eps_list:
                label = AUGMENTATION_NAME[aug_level]
                print(f"  aug_level={aug_level} ({label}), eps={eps}")

                try:
                    if aug_level == 0 and eps == 0 and baseline_metrics is not None:
                        metrics = baseline_metrics
                    else:
                        adj, n_observed = build_adjacency_for_setting(
                            dataset=dataset,
                            aug_level=aug_level,
                            eps=eps,
                            partition_dir=args.partition_dir,
                            reduced_dir=args.reduced_dir,
                            partition_id_offset=args.partition_id_offset,
                            partition_edge_id_offset=args.partition_edge_id_offset,
                            allow_orphan_singletons=allow_orphan_singletons,
                            infer_partition_edges_if_missing=infer_partition_edges_if_missing,
                            generate_missing_partitions=args.generate_missing_partitions,
                        )

                        metrics = compute_resistance_metrics(
                            adj,
                            num_observed_nodes=n_observed,
                            laplacian_mode=args.laplacian_mode,
                            mean_aggregation=args.mean_aggregation,
                            topk_pairs=args.topk_pairs,
                            topk_fraction=args.topk_fraction,
                        )

                    row = {
                        "dataset": dataset,
                        "augmentation": label,
                        "aug_level": aug_level,
                        "epsilon": int(eps),
                        "laplacian_mode": args.laplacian_mode,
                        **metrics,
                    }
                    rows.append(row)

                    total_resistance = float(metrics["total_resistance"])
                    if baseline_total_resistance is None:
                        delta_msg = "n/a"
                    elif np.isclose(baseline_total_resistance, 0.0):
                        delta_msg = "+0.00%" if np.isclose(total_resistance, 0.0) else "+inf"
                    else:
                        delta_pct = 100.0 * (total_resistance - baseline_total_resistance) / baseline_total_resistance
                        delta_msg = f"{delta_pct:+.2f}%"

                    print(
                        f"    total_resistance={total_resistance:.6f}, "
                        f"mean_effective={float(metrics['mean_effective_resistance']):.6f}, "
                        f"delta_vs_original={delta_msg}"
                    )

                except Exception as exc:
                    print(f"    [WARN] Skipping {dataset} aug_level={aug_level} eps={eps}: {exc}")

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        print("[WARN] No results collected.")
        return

    out_df = out_df.sort_values(by=["dataset", "aug_level", "epsilon"]).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)

    print(f"\n[RESISTANCE] Wrote {args.output} with {len(out_df)} rows")


if __name__ == "__main__":
    main()
