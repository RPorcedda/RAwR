from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
from torch_geometric.data import Data


# --------------------------------------------------------------------- #
# Partition helpers                                                     #
# --------------------------------------------------------------------- #
def load_partitions(
    partition_path: str | Path,
    *,
    id_offset: int = 0,               # −1 when IDs start from 1
) -> Tuple[Dict[str, int], int]:
    """
    Returns
    -------
    map_origID→partition : dict[str, int]
    num_partitions       : int
    """
    part_map: Dict[str, int] = {}
    k = 0
    with Path(partition_path).expanduser().open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            for tok in line.replace(",", " ").split():
                part_map[str(int(tok) + id_offset)] = k
            k += 1
    return part_map, k


def _read_partition_edges(edge_path: str | Path) -> torch.Tensor:
    """Return a 2 × E tensor of undirected partition edges (may be empty)."""
    pairs = []
    with Path(edge_path).expanduser().open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            u, v, *_ = line.split()
            pairs.append((int(u), int(v)))
    if not pairs:
        return torch.empty(2, 0, dtype=torch.long)
    mat = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    return torch.cat([mat, mat.flip(0)], dim=1)   # make undirected


# --------------------------------------------------------------------- #
# Augmentation                                                          #
# --------------------------------------------------------------------- #
def augment_graph(
    data: Data,
    *,
    partition_map: Dict[str, int],
    num_partitions: int,
    connect_partition_edges: bool = False,
    partition_edge_path: Optional[str | Path] = None,
    allow_orphan_singletons: bool = True,
) -> Data:
    """Return a **new** Data object whose node set is originals + partitions."""
    n_old = data.num_nodes
    dev = data.edge_index.device

    # --- map every original node IDX → partition IDX ------------------
    node_part = torch.zeros(n_old, dtype=torch.long)
    # cur_p = num_partitions
    # for orig_id_str, idx in id2idx.items():
    #     pid = partition_map.get(orig_id_str)
    #     if pid is None:
    #         if not allow_orphan_singletons:
    #             raise KeyError(f"Node {orig_id_str} missing from partition file")
    #         pid = cur_p
    #         partition_map[orig_id_str] = cur_p
    #         cur_p += 1
    #     node_part[idx] = pid
    # num_partitions = cur_p                                # may have grown
    for orig_str, p in partition_map.items():
        idx = int(orig_str)
        node_part[idx] = p

    # --- features -----------------------------------------------------
    # if data.x is not None:
    #     zeros = torch.zeros(num_partitions, data.x.size(1), dtype=data.x.dtype, device=dev)
    #     x_aug = torch.cat([data.x, zeros], dim=0)
    # else:
    #     x_aug = None

    # part_feats = -torch.eye(num_partitions, data.x.size(1), dtype=data.x.dtype, device=dev)
    # x_aug = torch.cat([data.x, part_feats], dim=0)

    orig_feats = data.x
    # zero‐pad originals in the new “partition” axes
    orig_pad  = torch.zeros(n_old, num_partitions, device=dev)
    # one‐hot for partition nodes in those axes
    part_onehot = torch.eye(num_partitions, device=dev)
    x_aug = torch.cat([
        torch.cat([orig_feats, orig_pad], dim=1),
        torch.cat([torch.zeros(num_partitions, orig_feats.size(1), device=dev), part_onehot], dim=1),
    ], dim=0)

    # --- edges: original graph ---------------------------------------
    edge_aug = data.edge_index.clone()

    #     • node ↔ its partition
    src = torch.arange(n_old, device=dev)
    dst = node_part.to(dev) + n_old
    node_part_edges = torch.stack([torch.cat([src, dst]),
                                   torch.cat([dst, src])], dim=0)
    edge_aug = torch.cat([edge_aug.to(dev), node_part_edges], dim=1)

    #     • partition ↔ partition (optional)
    if connect_partition_edges and partition_edge_path:
        pp = _read_partition_edges(partition_edge_path).to(dev)
        if pp.numel():
            edge_aug = torch.cat([edge_aug, pp + n_old], dim=1)

    # --- build new Data ----------------------------------------------
    #new = Data(x=x_aug, edge_index=edge_aug, y=data.y)
    
    # --- pad y con label dummy (ignore_index=-100) -------------
    if data.y is not None:
        pad_y = torch.full((num_partitions,), -100, dtype=data.y.dtype, device=dev)
        y_aug = torch.cat([data.y, pad_y], dim=0)
    else:
        y_aug = None
    # --- build new Data ----------------------------------------------
    new = Data(x=x_aug, edge_index=edge_aug, y=y_aug)
    
    for mask in ("train_mask", "val_mask", "test_mask"):
        if hasattr(data, mask):
            pad = torch.zeros(num_partitions, dtype=torch.bool, device=dev)
            setattr(new, mask, torch.cat([getattr(data, mask), pad], dim=0))
    return new
