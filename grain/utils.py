import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import Planetoid

DATA_PATH = "data"


@dataclass
class SingleGraphDataset:
    data: Data


def evaluate_policy(env, model, render, turns=5):
    scores = 0
    for _ in range(turns):
        s, done, ep_r, steps = env.reset2(), False, 0, 0
        while not done:
            steps += 1
            # Take deterministic actions at test time.
            a = model.select_action(s)
            a = a.reshape(-1, 1)

            s_prime, r, done, _ = env.step2(a)
            if done and steps != 5:
                done = False
            else:
                done = True

            ep_r += r
            s = s_prime
            if render:
                env.render()

        scores += ep_r
    return scores / turns


def str2bool(v):
    """Transfer str to bool for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "True", "true", "TRUE", "t", "y", "1"):
        return True
    if v.lower() in ("no", "False", "false", "FALSE", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_dataset(
    name: str,
    use_lcc: bool = True,
    data_source: str = "planetoid",
    rawr_root: str | os.PathLike | None = None,
    val_ratio: float = 0.05,
    test_ratio: float = 0.10,
    split_seed: int = 0,
    use_features: bool = True,
    rewiring_mode: str = "none",
    rewiring_epsilon: int | None = None,
    partition_path: str | os.PathLike | None = None,
    partition_edge_path: str | os.PathLike | None = None,
    partition_id_offset: int = -1,
) -> InMemoryDataset | SingleGraphDataset:
    if data_source == "rawr":
        if rawr_root is None:
            raise ValueError("`rawr_root` is required when data_source='rawr'.")
        data = load_rawr_graph(
            dataset_name=name,
            rawr_root=rawr_root,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            split_seed=split_seed,
            use_features=use_features,
            rewiring_mode=rewiring_mode,
            rewiring_epsilon=rewiring_epsilon,
            partition_path=partition_path,
            partition_edge_path=partition_edge_path,
            partition_id_offset=partition_id_offset,
        )
        return SingleGraphDataset(data=data)

    if data_source != "planetoid":
        raise ValueError(f"Unsupported data_source: {data_source}")
    if rewiring_mode != "none":
        raise ValueError("Rewiring is currently supported only with data_source='rawr'.")

    path = os.path.join(DATA_PATH, name)
    if name in ["Cora", "Citeseer", "PubMed"]:
        dataset = Planetoid(path, name)
        use_lcc = False
    else:
        raise Exception("Unknown dataset.")

    if use_lcc:
        x_new = dataset.data.x
        y_new = dataset.data.y
        edges = dataset.data.edge_index
        data = Data(
            x=x_new,
            edge_index=torch.LongTensor(edges),
            y=y_new,
            train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
        )
        dataset.data = data

    return dataset


def _read_rawr_edgelist(edge_path: Path) -> torch.Tensor:
    edges_df = pd.read_csv(
        edge_path,
        comment="#",
        header=None,
        sep=r"\s+|,",
        engine="python",
        usecols=[0, 1],
    )
    edges = edges_df.astype(int).to_numpy()
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    # Mirror to undirected to match RAwR's loader.
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return edge_index


def _load_partitions(partition_path: Path, id_offset: int = 0) -> tuple[dict[str, int], int]:
    part_map: dict[str, int] = {}
    partition_count = 0
    with partition_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for tok in line.replace(",", " ").split():
                part_map[str(int(tok) + id_offset)] = partition_count
            partition_count += 1
    return part_map, partition_count


def _read_partition_edges(edge_path: Path) -> torch.Tensor:
    edges = []
    with edge_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            u, v, *_ = line.split()
            edges.append((int(u), int(v)))
    if not edges:
        return torch.empty(2, 0, dtype=torch.long)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return torch.cat([edge_index, edge_index.flip(0)], dim=1)


def _augment_graph_with_partitions(
    data: Data,
    partition_map: dict[str, int],
    num_partitions: int,
    connect_partition_edges: bool = False,
    partition_edge_path: Path | None = None,
) -> Data:
    num_original_nodes = data.num_nodes
    device = data.edge_index.device

    node_to_partition = torch.full((num_original_nodes,), -1, dtype=torch.long)
    for orig_idx_str, partition_idx in partition_map.items():
        node_to_partition[int(orig_idx_str)] = partition_idx
    if (node_to_partition < 0).any():
        missing_nodes = int((node_to_partition < 0).sum().item())
        raise ValueError(
            f"Partition file does not cover all original nodes: {missing_nodes} nodes missing."
        )

    orig_features = data.x
    orig_feature_pad = torch.zeros(num_original_nodes, num_partitions, device=device)
    partition_one_hot = torch.eye(num_partitions, device=device)
    x_aug = torch.cat(
        [
            torch.cat([orig_features, orig_feature_pad], dim=1),
            torch.cat(
                [torch.zeros(num_partitions, orig_features.size(1), device=device), partition_one_hot],
                dim=1,
            ),
        ],
        dim=0,
    )

    edge_aug = data.edge_index.clone()
    src = torch.arange(num_original_nodes, device=device)
    dst = node_to_partition.to(device) + num_original_nodes
    node_partition_edges = torch.stack(
        [torch.cat([src, dst]), torch.cat([dst, src])],
        dim=0,
    )
    edge_aug = torch.cat([edge_aug.to(device), node_partition_edges], dim=1)

    if connect_partition_edges and partition_edge_path is not None:
        partition_edges = _read_partition_edges(partition_edge_path).to(device)
        if partition_edges.numel() > 0:
            edge_aug = torch.cat([edge_aug, partition_edges + num_original_nodes], dim=1)

    if data.y is not None:
        # Virtual nodes must never be part of node classification.
        y_pad = torch.full((num_partitions,), -100, dtype=data.y.dtype, device=device)
        y_aug = torch.cat([data.y, y_pad], dim=0)
    else:
        y_aug = None

    new_data = Data(x=x_aug, edge_index=edge_aug, y=y_aug)
    for mask_name in ("train_mask", "val_mask", "test_mask"):
        if hasattr(data, mask_name):
            mask_pad = torch.zeros(num_partitions, dtype=torch.bool, device=device)
            setattr(new_data, mask_name, torch.cat([getattr(data, mask_name), mask_pad], dim=0))
    return new_data


def set_split_masks_by_ratio(
    seed: int,
    data: Data,
    val_ratio: float = 0.05,
    test_ratio: float = 0.10,
) -> Data:
    num_nodes = int(data.y.shape[0])
    n_test = int(test_ratio * num_nodes)
    n_val = int(val_ratio * num_nodes)
    n_train = num_nodes - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=generator)

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val : n_train + n_val + n_test]

    def _mask(idx):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = 1
        return mask

    data.train_mask = _mask(train_idx)
    data.val_mask = _mask(val_idx)
    data.test_mask = _mask(test_idx)
    return data


def load_rawr_graph(
    dataset_name: str,
    rawr_root: str | os.PathLike,
    val_ratio: float = 0.05,
    test_ratio: float = 0.10,
    split_seed: int = 0,
    use_features: bool = True,
    rewiring_mode: str = "none",
    rewiring_epsilon: int | None = None,
    partition_path: str | os.PathLike | None = None,
    partition_edge_path: str | os.PathLike | None = None,
    partition_id_offset: int = -1,
) -> Data:
    rawr_root = Path(rawr_root).expanduser().resolve()
    dataset_dir = rawr_root / dataset_name
    edge_path = dataset_dir / f"{dataset_name}.edgelist"
    feat_path = dataset_dir / f"{dataset_name}.x"
    label_path = dataset_dir / f"{dataset_name}.y"

    if not edge_path.exists():
        raise FileNotFoundError(f"Missing edge list: {edge_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Missing labels: {label_path}")

    edge_index = _read_rawr_edgelist(edge_path)

    labels = (
        pd.read_csv(
            label_path,
            header=None,
            sep=r"\s+|,",
            engine="python",
        )
        .dropna(axis=1, how="all")
        .to_numpy()
        .reshape(-1)
    )
    y = torch.tensor(labels, dtype=torch.long)

    if use_features and feat_path.exists():
        x = torch.tensor(
            pd.read_csv(
                feat_path,
                header=None,
                sep=r"\s+|,",
                engine="python",
            )
            .dropna(axis=1, how="all")
            .to_numpy(),
            dtype=torch.float,
        )
    else:
        num_nodes = int(max(y.shape[0], edge_index.max().item() + 1))
        x = torch.eye(num_nodes, dtype=torch.float)

    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"Feature/label row mismatch for {dataset_name}: x={x.shape[0]} y={y.shape[0]}"
        )

    data = Data(x=x, edge_index=edge_index, y=y)
    data = set_split_masks_by_ratio(
        seed=split_seed,
        data=data,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    if rewiring_mode not in {"none", "rep_nodes", "rep_edges"}:
        raise ValueError(
            f"Unsupported rewiring mode '{rewiring_mode}'. "
            "Use one of: none, rep_nodes, rep_edges."
        )

    if rewiring_mode != "none":
        if partition_path is None:
            if rewiring_epsilon is None:
                raise ValueError(
                    "`rewiring_epsilon` is required for rewiring unless `partition_path` is provided."
                )
            partition_path = rawr_root / "partitions" / f"{dataset_name}P{rewiring_epsilon}"
        else:
            partition_path = Path(partition_path).expanduser().resolve()

        if not Path(partition_path).exists():
            raise FileNotFoundError(f"Partition file not found: {partition_path}")

        connect_partition_edges = rewiring_mode == "rep_edges"
        if connect_partition_edges:
            if partition_edge_path is None:
                if rewiring_epsilon is None:
                    raise ValueError(
                        "`rewiring_epsilon` is required for rep_edges unless `partition_edge_path` is provided."
                    )
                partition_edge_path = (
                    rawr_root / "reducedNetworks" / f"{dataset_name}BE{rewiring_epsilon}.edgelist"
                )
            else:
                partition_edge_path = Path(partition_edge_path).expanduser().resolve()
            if not Path(partition_edge_path).exists():
                raise FileNotFoundError(f"Partition edge file not found: {partition_edge_path}")

        part_map, num_partitions = _load_partitions(
            Path(partition_path),
            id_offset=partition_id_offset,
        )
        data = _augment_graph_with_partitions(
            data=data,
            partition_map=part_map,
            num_partitions=num_partitions,
            connect_partition_edges=connect_partition_edges,
            partition_edge_path=Path(partition_edge_path) if partition_edge_path else None,
        )
    return data


def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
    visited_nodes = set()
    queued_nodes = set([start])
    row, col = dataset.data.edge_index.numpy()
    while queued_nodes:
        current_node = queued_nodes.pop()
        visited_nodes.update([current_node])
        neighbors = col[np.where(row == current_node)[0]]
        neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
        queued_nodes.update(neighbors)
    return visited_nodes


def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
    remaining_nodes = set(range(dataset.data.x.shape[0]))
    comps = []
    while remaining_nodes:
        start = min(remaining_nodes)
        comp = get_component(dataset, start)
        comps.append(comp)
        remaining_nodes = remaining_nodes.difference(comp)
    return np.array(list(comps[np.argmax(list(map(len, comps)))]))


def remap_edges(edges: list, mapper: dict) -> list:
    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    row = list(map(lambda x: mapper[x], row))
    col = list(map(lambda x: mapper[x], col))
    return [row, col]


def get_node_mapper(lcc: np.ndarray) -> dict:
    mapper = {}
    counter = 0
    for node in lcc:
        mapper[node] = counter
        counter += 1
    return mapper


def get_adj_matrix(dataset: InMemoryDataset) -> np.ndarray:
    num_nodes = dataset.data.x.shape[0]
    adj_matrix = np.zeros(shape=(num_nodes, num_nodes))
    for i, j in zip(dataset.data.edge_index[0], dataset.data.edge_index[1]):
        adj_matrix[i, j] = 1.0
    return adj_matrix


def load_geom_gcn_dataset(name):
    fulldata = scipy.io.loadmat(f"{DATA_PATH}/{name}.mat")
    edge_index = fulldata["edge_index"]
    node_feat = fulldata["node_feat"]
    label = np.array(fulldata["label"], dtype=int).flatten()
    num_nodes = node_feat.shape[0]

    dataset = NCDataset(name)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    node_feat = torch.tensor(node_feat, dtype=torch.float)
    dataset.graph = {
        "edge_index": edge_index,
        "node_feat": node_feat,
        "edge_feat": None,
        "num_nodes": num_nodes,
    }
    label = torch.tensor(label, dtype=torch.long)
    dataset.label = label
    return dataset


class NCDataset(object):
    def __init__(self, name, root=f"{DATA_PATH}"):
        self.name = name
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type="random", train_prop=0.5, valid_prop=0.25):
        if split_type == "random":
            ignore_negative = False if self.name == "ogbn-proteins" else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label,
                train_prop=train_prop,
                valid_prop=valid_prop,
                ignore_negative=ignore_negative,
            )
            split_idx = {
                "train": train_idx,
                "valid": valid_idx,
                "test": test_idx,
            }
            return split_idx
        raise ValueError(f"Unsupported split_type: {split_type}")

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, len(self))


def rand_train_test_idx(label, train_prop=0.5, valid_prop=0.25, ignore_negative=True):
    """Randomly split labels into train/valid/test indices."""
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))
    train_indices = perm[:train_num].type(torch.long)
    val_indices = perm[train_num : train_num + valid_num].type(torch.long)
    test_indices = perm[train_num + valid_num :].type(torch.long)

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]
    return train_idx, valid_idx, test_idx


def set_train_val_test_split(
    seed: int,
    data: Data,
    num_development: int = 1500,
    num_per_class: int = 20,
) -> Data:
    num_nodes = data.y.shape[0]
    all_idx = np.arange(num_nodes)

    train_idx = []
    rnd_state = np.random.RandomState(seed)
    for c in range(data.y.max() + 1):
        class_idx = all_idx[np.where(data.y.cpu() == c)[0]]
        if len(class_idx) < num_per_class:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, num_per_class, replace=False))

    ctrain_idx = np.array([i for i in np.arange(num_nodes) if i not in train_idx])
    ctrain_idx_val = rnd_state.choice(ctrain_idx, num_development, replace=False)
    val_idx = rnd_state.choice(ctrain_idx, num_development, replace=False)
    test_idx = ctrain_idx[
        [i for i in np.arange(num_nodes - len(train_idx)) if i not in ctrain_idx_val]
    ]

    def get_mask(idx):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = 1
        return mask

    data.train_mask = get_mask(train_idx)
    data.val_mask = get_mask(val_idx)
    data.test_mask = get_mask(test_idx)
    return data
