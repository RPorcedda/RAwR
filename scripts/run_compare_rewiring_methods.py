#!/usr/bin/env python3
"""Run fair RAwR-vs-ComFy-vs-TRIGON comparisons under shared settings.

This script keeps the RAwR evaluation protocol fixed (same model family, layers,
seeded splits, epochs) and swaps only the rewiring stage.
"""

import argparse
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import Delaunay
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

SCRIPT_DIR = Path(__file__).resolve().parent
REPRO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPRO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import load_data_from_edgelist


DATASETS_DEFAULT = [
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

MODELS_DEFAULT = ["GCN", "GAT", "GIN"]
METHODS_DEFAULT = ["rawr", "comfy", "trigon"]

RAWR_EPSILONS = {
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


def parse_csv_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_bool_csv(value: str) -> List[bool]:
    out: List[bool] = []
    for tok in parse_csv_list(value):
        t = tok.lower()
        if t in {"1", "true", "t", "yes", "y"}:
            out.append(True)
        elif t in {"0", "false", "f", "no", "n"}:
            out.append(False)
        else:
            raise ValueError(f"Cannot parse bool token: {tok}")
    return out


def dataset_tag(dataset_spec: str) -> str:
    return dataset_spec.replace("/", "__")


def resolve_dataset_paths(dataset_spec: str) -> Tuple[Path, Optional[Path], Path]:
    base = Path(dataset_spec)
    if not base.is_absolute():
        repro_candidate = REPRO_ROOT / base
        if repro_candidate.exists():
            base = repro_candidate
    stem = base.name
    edge_path = base / f"{stem}.edgelist"
    feat_path = base / f"{stem}.x"
    label_path = base / f"{stem}.y"
    if not edge_path.exists():
        raise FileNotFoundError(f"Missing edge list for dataset '{dataset_spec}': {edge_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Missing labels for dataset '{dataset_spec}': {label_path}")
    return edge_path, (feat_path if feat_path.exists() else None), label_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def edge_index_to_undirected_set(edge_index: torch.Tensor) -> Set[Tuple[int, int]]:
    edges: Set[Tuple[int, int]] = set()
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for u, v in zip(src, dst):
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        edges.add((int(a), int(b)))
    return edges


def write_edgelist(path: Path, edge_set: Iterable[Tuple[int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for u, v in sorted(edge_set):
            f.write(f"{u} {v}\n")


def build_rewiring_features(
    data,
    *,
    use_features_for_model: bool,
    rewiring_feature_mode: str,
) -> torch.Tensor:
    if rewiring_feature_mode == "always_original":
        return data.x.float()

    if rewiring_feature_mode != "match_model":
        raise ValueError(f"Unsupported rewiring_feature_mode: {rewiring_feature_mode}")

    if use_features_for_model:
        return data.x.float()

    # "No features" in model mode can be huge (one-hot); for rewiring we use
    # structural-only input (degree) to keep runs tractable while feature-free.
    deg = torch.bincount(data.edge_index[0], minlength=data.num_nodes).float().unsqueeze(1)
    return deg


def knn_edge_set(features: np.ndarray, k: int) -> Set[Tuple[int, int]]:
    n = features.shape[0]
    if n <= 1:
        return set()
    nn_k = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=nn_k, metric="cosine")
    nn.fit(features)
    _, idx = nn.kneighbors(features, return_distance=True)
    out: Set[Tuple[int, int]] = set()
    for u in range(n):
        for v in idx[u, 1:]:
            v = int(v)
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            out.add((a, b))
    return out


def delaunay_edge_set(features: np.ndarray, seed: int) -> Set[Tuple[int, int]]:
    n = features.shape[0]
    if n < 4:
        return set()
    try:
        x2 = PCA(n_components=2, random_state=seed).fit_transform(features)
        tri = Delaunay(x2, qhull_options="QJ")
    except Exception:
        return set()

    out: Set[Tuple[int, int]] = set()
    for simplex in tri.simplices:
        a, b, c = [int(t) for t in simplex]
        out.add((min(a, b), max(a, b)))
        out.add((min(a, c), max(a, c)))
        out.add((min(b, c), max(b, c)))
    return out


def enumerate_triangles(edge_set: Set[Tuple[int, int]], num_nodes: int) -> List[Tuple[int, int, int]]:
    adj: List[Set[int]] = [set() for _ in range(num_nodes)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)

    triangles: List[Tuple[int, int, int]] = []
    for u in range(num_nodes):
        neigh = sorted([v for v in adj[u] if v > u])
        for i, v in enumerate(neigh):
            common = adj[v].intersection(neigh[i + 1 :])
            for w in common:
                triangles.append((u, v, w))
    return triangles


class TriangleSelectorMLP(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def rewire_trigon(
    data,
    rewiring_features: torch.Tensor,
    *,
    seed: int,
    knn_k: int,
    max_triangles: int,
    epochs: int,
) -> Set[Tuple[int, int]]:
    set_seed(seed)
    num_nodes = data.num_nodes
    orig_edges = edge_index_to_undirected_set(data.edge_index)

    feat_np = rewiring_features.detach().cpu().numpy()
    knn_edges = knn_edge_set(feat_np, k=knn_k)
    del_edges = delaunay_edge_set(feat_np, seed=seed)
    candidate_edges = set(orig_edges)
    candidate_edges.update(knn_edges)
    candidate_edges.update(del_edges)

    triangles = enumerate_triangles(candidate_edges, num_nodes)
    if not triangles:
        return orig_edges

    if len(triangles) > max_triangles:
        rng = random.Random(seed)
        triangles = rng.sample(triangles, max_triangles)

    tri_idx = torch.tensor(triangles, dtype=torch.long)
    f = rewiring_features.float()
    tri_feat = torch.cat([f[tri_idx[:, 0]], f[tri_idx[:, 1]], f[tri_idx[:, 2]]], dim=1)

    y = data.y
    train_mask = data.train_mask
    tri_labels = torch.zeros(len(triangles), dtype=torch.float)
    for i, (a, b, c) in enumerate(triangles):
        if bool(train_mask[a] and train_mask[b] and train_mask[c]):
            lab = {int(y[a]), int(y[b]), int(y[c])}
            tri_labels[i] = 1.0 if len(lab) <= 2 else 0.0

    # If labels are fully degenerate, fallback to feature-cohesion ranking.
    if tri_labels.sum().item() == 0.0:
        f_norm = F.normalize(f, dim=1, eps=1e-12)
        s1 = (f_norm[tri_idx[:, 0]] * f_norm[tri_idx[:, 1]]).sum(dim=1)
        s2 = (f_norm[tri_idx[:, 1]] * f_norm[tri_idx[:, 2]]).sum(dim=1)
        s3 = (f_norm[tri_idx[:, 0]] * f_norm[tri_idx[:, 2]]).sum(dim=1)
        tri_scores = (s1 + s2 + s3) / 3.0
        keep = max(1, int(0.2 * len(triangles)))
        top_idx = torch.topk(tri_scores, k=keep, largest=True).indices.tolist()
        selected_triangles = [triangles[i] for i in top_idx]
    else:
        model = TriangleSelectorMLP(in_dim=tri_feat.size(1))
        pos = tri_labels.sum().item()
        neg = float(len(tri_labels) - pos)
        pos_weight = torch.tensor(max(1.0, neg / max(1.0, pos)), dtype=torch.float)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optim = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

        best_loss = float("inf")
        best_state = None
        patience = 20
        bad = 0
        for _ in range(epochs):
            model.train()
            optim.zero_grad()
            logits = model(tri_feat)
            loss = criterion(logits, tri_labels)
            loss.backward()
            optim.step()

            cur = float(loss.item())
            if cur < best_loss:
                best_loss = cur
                bad = 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(tri_feat))

        chosen = (probs > 0.5).nonzero(as_tuple=False).view(-1).tolist()
        if not chosen:
            keep = max(1, int(0.2 * len(triangles)))
            chosen = torch.topk(probs, k=keep, largest=True).indices.tolist()
        selected_triangles = [triangles[i] for i in chosen]

    rewired: Set[Tuple[int, int]] = set()
    for a, b, c in selected_triangles:
        rewired.add((min(a, b), max(a, b)))
        rewired.add((min(a, c), max(a, c)))
        rewired.add((min(b, c), max(b, c)))

    if len(rewired) < max(1, int(0.3 * len(orig_edges))):
        rewired.update(orig_edges)

    return rewired


def rewire_comfy(
    data,
    rewiring_features: torch.Tensor,
    *,
    seed: int,
    budget_add: int,
    budget_delete: int,
    knn_k: int,
) -> Set[Tuple[int, int]]:
    set_seed(seed)
    num_nodes = data.num_nodes
    edge_set = edge_index_to_undirected_set(data.edge_index)

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_set)

    communities = list(nx.community.louvain_communities(G, seed=seed))
    node2comm: Dict[int, int] = {}
    for cid, comm in enumerate(communities):
        for u in comm:
            node2comm[int(u)] = cid

    x = rewiring_features.float()
    x = F.normalize(x, p=2, dim=1, eps=1e-12).cpu().numpy()

    remove_candidates: List[Tuple[int, float, int, int]] = []
    for u, v in G.edges():
        sim = float(np.dot(x[u], x[v]))
        inter_comm = 1 if node2comm.get(u) != node2comm.get(v) else 0
        remove_candidates.append((inter_comm, sim, int(u), int(v)))

    remove_candidates.sort(key=lambda t: (-t[0], t[1], t[2], t[3]))
    removed = 0
    for _, _, u, v in remove_candidates:
        if removed >= budget_delete:
            break
        if not G.has_edge(u, v):
            continue
        # avoid deleting bridge-like leaves too aggressively
        if G.degree(u) <= 1 or G.degree(v) <= 1:
            continue
        G.remove_edge(u, v)
        removed += 1

    add_scores: Dict[Tuple[int, int], float] = {}
    nn_k = min(knn_k + 1, num_nodes)
    if nn_k >= 2:
        nn = NearestNeighbors(n_neighbors=nn_k, metric="cosine")
        nn.fit(x)
        dist, idx = nn.kneighbors(x, return_distance=True)
        for u in range(num_nodes):
            for d, v in zip(dist[u, 1:], idx[u, 1:]):
                v = int(v)
                if u == v:
                    continue
                a, b = (u, v) if u < v else (v, u)
                if G.has_edge(a, b):
                    continue
                sim = 1.0 - float(d)
                intra = 1 if node2comm.get(a) == node2comm.get(b) else 0
                score = sim + 0.05 * intra
                prev = add_scores.get((a, b))
                if prev is None or score > prev:
                    add_scores[(a, b)] = score

    added = 0
    for (u, v), _ in sorted(add_scores.items(), key=lambda kv: kv[1], reverse=True):
        if added >= budget_add:
            break
        if not G.has_edge(u, v):
            G.add_edge(u, v)
            added += 1

    rewired = {(min(int(u), int(v)), max(int(u), int(v))) for u, v in G.edges() if int(u) != int(v)}
    return rewired


def generate_rewired_graph(
    *,
    dataset: str,
    seed: int,
    method: str,
    use_features_for_model: bool,
    rewiring_feature_mode: str,
    comfy_budget_add: int,
    comfy_budget_delete: int,
    comfy_knn_k: int,
    trigon_knn_k: int,
    trigon_max_triangles: int,
    trigon_epochs: int,
    out_dir: Path,
    force: bool,
) -> Path:
    feat_flag = "feat" if use_features_for_model else "nofeat"
    out_path = out_dir / method / f"{dataset_tag(dataset)}_seed{seed}_{feat_flag}.edgelist"
    if out_path.exists() and not force:
        return out_path

    edge_path, feat_path, label_path = resolve_dataset_paths(dataset)

    full_data, _, _, _ = load_data_from_edgelist(
        edge_path,
        node_feat_path=feat_path,
        label_path=label_path,
        val_ratio=0.05,
        test_ratio=0.10,
        random_state=seed,
    )

    rew_feat = build_rewiring_features(
        full_data,
        use_features_for_model=use_features_for_model,
        rewiring_feature_mode=rewiring_feature_mode,
    )

    if method == "comfy":
        rewired_edges = rewire_comfy(
            full_data,
            rew_feat,
            seed=seed,
            budget_add=comfy_budget_add,
            budget_delete=comfy_budget_delete,
            knn_k=comfy_knn_k,
        )
    elif method == "trigon":
        rewired_edges = rewire_trigon(
            full_data,
            rew_feat,
            seed=seed,
            knn_k=trigon_knn_k,
            max_triangles=trigon_max_triangles,
            epochs=trigon_epochs,
        )
    else:
        raise ValueError(f"Unsupported method: {method}")

    write_edgelist(out_path, rewired_edges)
    return out_path


def run_main(cmd: Sequence[str], *, dry_run: bool) -> None:
    if dry_run:
        print("DRY-RUN:", " ".join(cmd))
        return
    subprocess.run(list(cmd), check=True)


def main() -> None:
    p = argparse.ArgumentParser("Run RAwR/TRIGON/ComFy under shared settings")
    p.add_argument("--methods", default=",".join(METHODS_DEFAULT), help="Comma-list: rawr,comfy,trigon")
    p.add_argument("--datasets", default=",".join(DATASETS_DEFAULT), help="Comma-list of datasets")
    p.add_argument("--models", default=",".join(MODELS_DEFAULT), help="Comma-list of models")
    p.add_argument("--use_features_options", default="True,False", help="Comma-list of bools")

    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--seed_stride", type=int, default=111)
    p.add_argument("--layers", type=int, default=2)

    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--epochs_clf", type=int, default=500)
    p.add_argument("--epochs_lp", type=int, default=500)

    p.add_argument("--rawr_connect_partition_edges", action="store_true")
    p.add_argument("--partition_path", type=str, default="partitions")
    p.add_argument("--reduction_path", type=str, default="reducedNetworks")

    p.add_argument("--rewiring_feature_mode", choices=["match_model", "always_original"], default="match_model")
    p.add_argument("--comfy_budget_add", type=int, default=100)
    p.add_argument("--comfy_budget_delete", type=int, default=100)
    p.add_argument("--comfy_knn_k", type=int, default=50)

    p.add_argument("--trigon_knn_k", type=int, default=10)
    p.add_argument("--trigon_max_triangles", type=int, default=30000)
    p.add_argument("--trigon_epochs", type=int, default=200)

    p.add_argument("--rewired_dir", type=str, default="rewiredNetworks_baselines")
    p.add_argument("--nc_out", type=str, default="node_classification_compare.csv")
    p.add_argument("--lp_out", type=str, default="link_prediction_compare.csv")
    p.add_argument("--skip_lp", action="store_true",
                   help="If set, run node classification only")

    p.add_argument("--force_rewire", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    methods = [m.lower() for m in parse_csv_list(args.methods)]
    datasets = parse_csv_list(args.datasets)
    models = parse_csv_list(args.models)
    use_features_options = parse_bool_csv(args.use_features_options)

    seeds = [args.seed_stride * i for i in range(1, args.trials + 1)]
    root = REPRO_ROOT
    rewired_dir = (root / args.rewired_dir).resolve()

    nc_out_path = (root / args.nc_out).resolve()
    lp_out_path = (root / args.lp_out).resolve()

    for method in methods:
        if method not in {"rawr", "comfy", "trigon"}:
            raise ValueError(f"Unsupported method: {method}")

    for model in models:
        for dataset in datasets:
            for use_features in use_features_options:
                edge_path, feat_path, label_path = resolve_dataset_paths(dataset)
                feat_arg: List[str] = []
                if use_features:
                    if feat_path is None:
                        raise FileNotFoundError(
                            f"Dataset '{dataset}' has no .x file but --use_features_options includes True"
                        )
                    feat_arg = ["--feat_path", str(feat_path)]

                for trial_id, seed in enumerate(seeds, start=1):
                    common = [
                        sys.executable,
                        "-W",
                        "ignore",
                        str(SRC_DIR / "main.py"),
                        "--model",
                        model,
                        "--layers",
                        str(args.layers),
                        "--hidden",
                        str(args.hidden),
                        "--epochs_clf",
                        str(args.epochs_clf),
                        "--epochs_lp",
                        str(args.epochs_lp),
                        "--edge_path",
                        str(edge_path),
                        "--dataset_name",
                        dataset,
                        "--label_path",
                        str(label_path),
                        "--trial",
                        str(trial_id),
                        "--random_state",
                        str(seed),
                        "--nc_out_path",
                        str(nc_out_path),
                        "--lp_out_path",
                        str(lp_out_path),
                    ]
                    if args.skip_lp:
                        common += ["--skip_lp"]

                    for method_name in methods:
                        if method_name == "rawr":
                            for eps in RAWR_EPSILONS[dataset]:
                                cmd = common.copy()
                                cmd += feat_arg
                                cmd += [
                                    "--augment_partitions",
                                    "--partition_path",
                                    f"{args.partition_path}/{dataset}P{eps}",
                                    "--partition_id_offset",
                                    "-1",
                                    "--method_tag",
                                    f"rawr_eps{eps}",
                                ]
                                if args.rawr_connect_partition_edges:
                                    cmd += [
                                        "--connect_partition_edges",
                                        "--partition_edge_path",
                                        f"{args.reduction_path}/{dataset}BE{eps}.edgelist",
                                    ]
                                run_main(cmd, dry_run=args.dry_run)
                        else:
                            feat_flag = "feat" if use_features else "nofeat"
                            rewired_path = rewired_dir / method_name / f"{dataset_tag(dataset)}_seed{seed}_{feat_flag}.edgelist"
                            if not args.dry_run:
                                rewired_path = generate_rewired_graph(
                                    dataset=dataset,
                                    seed=seed,
                                    method=method_name,
                                    use_features_for_model=use_features,
                                    rewiring_feature_mode=args.rewiring_feature_mode,
                                    comfy_budget_add=args.comfy_budget_add,
                                    comfy_budget_delete=args.comfy_budget_delete,
                                    comfy_knn_k=args.comfy_knn_k,
                                    trigon_knn_k=args.trigon_knn_k,
                                    trigon_max_triangles=args.trigon_max_triangles,
                                    trigon_epochs=args.trigon_epochs,
                                    out_dir=rewired_dir,
                                    force=args.force_rewire,
                                )

                            cmd = common.copy()
                            cmd[cmd.index("--edge_path") + 1] = str(rewired_path)
                            cmd += feat_arg
                            if method_name == "comfy":
                                cmd += [
                                    "--method_tag",
                                    f"comfy_add{args.comfy_budget_add}_del{args.comfy_budget_delete}",
                                ]
                            else:
                                cmd += [
                                    "--method_tag",
                                    f"trigon_k{args.trigon_knn_k}_mt{args.trigon_max_triangles}",
                                ]
                            run_main(cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
