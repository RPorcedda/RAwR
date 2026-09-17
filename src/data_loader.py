from pathlib import Path
from typing import Tuple, Dict, Optional

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit

import sklearn
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler

from augment_graphs import load_partitions, augment_graph


# --------------------------------------------------------------------------- #
# Raw edge-list I/O                                                           #
# --------------------------------------------------------------------------- #
def _read_edge_file(edge_path: str | Path) -> pd.DataFrame:
    """
    Supports:
    • *.edgelist*    → whitespace-separated src dst (ignores comments/#)
    • *.csv / .tsv*  → first two columns are src, dst
    """
    edge_path = Path(edge_path).expanduser()
    if edge_path.suffix == ".edgelist":
        edges = []
        with edge_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                u, v, *_ = line.split()
                edges.append((u, v))
        df = pd.DataFrame(edges, columns=["src", "dst"])
    else:
        df = pd.read_csv(edge_path, comment="#", header=None, usecols=[0, 1])
        df.columns = ["src", "dst"]
    return df





def encode_string_columns(
    df: pd.DataFrame,
    nodeLabelEncoder: Optional[bool]=False,
    index: Optional[str]=None
    ):# -> Union[pd.DataFrame, Tuple[Union[[pd.DataFrame, sklearn.preprocessing.LabelEncoder]]]]:
    """
    OneHotEncoding for categorical features and remap node labels.
    Params:
        df:  DataFrame containing the node features
        nodeLabelEncoder: if True, labels are remapped and
        index: label column name
    Return:
        df:  remapped DataFrame of node features
        node_lb: fitted LabelEncoder
    """
    if nodeLabelEncoder:
        df[index] = df[index].astype(str)
        node_lb = LabelEncoder()
        node_lb.fit(df[index])
        df[index] = node_lb.transform(df[index])
    for i, col in enumerate(df.columns):
        if index is not None and col==index: # Leave node labels
            continue
        if df[col].dtype=="object":
            df[col] = df[col].astype(str)
        if pd.api.types.is_string_dtype(df[col]):
            lb = LabelBinarizer()
            lb_results = lb.fit_transform(df[col])
            if len(lb.classes_)==2:
                classes =[lb.classes_[0]]
            else:
                classes = lb.classes_
            lb_results_df = pd.DataFrame(lb_results,
                                        columns=[f"{col}_{cls}" for cls in classes], index=df.index)

            df = pd.concat([df, lb_results_df], axis=1)
            df.drop(columns=[col], inplace=True)

    if not nodeLabelEncoder:
        df["COPIED_INDEX"]=df.index

    if nodeLabelEncoder:
        return df, node_lb
    else:
        return df



def standardize_non_binary_columns(node_data):
    """
    Apply StandardScaler on non-binary features
    
    Params:
        node_data: DataFrame containing node features
        
    Returns:
        standardized_node_data: DataFrame containing standardized columns
    """
    scaler = scaler = StandardScaler()
    standardized_node_data = node_data.copy()
    
    # Iterate through the columns
    for column in node_data.columns:
        # Check if the column is binary (contains only 0s and 1s)
        if node_data[column].nunique() > 2:  # Column has more than 2 unique values, so it's not binary
            # Normalize the column using Min-Max scaling
            standardized_node_data[column] = scaler.fit_transform(node_data[[column]])
            # Transform nans into zeros
            # standardized_node_data[column][np.isnan(standardized_node_data[column])]=0
            standardized_node_data.loc[np.isnan(standardized_node_data[column]), column] = 0  
    
    return standardized_node_data


# --------------------------------------------------------------------------- #
# Build PyG `Data` object                                                     #
# --------------------------------------------------------------------------- #
def _build_pyg_data(
    edges_df: pd.DataFrame,
    features: Optional[pd.DataFrame] = None,
    labels: Optional[pd.DataFrame] = None,
) -> Tuple[Data, Dict[int, int]]:
    import torch
    from torch_geometric.data import Data

    edges_int = edges_df.astype(int)
    edge_index = torch.tensor([
        edges_int["src"].values,
        edges_int["dst"].values
    ], dtype=torch.long)

    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    if features is not None:
        features_enc = encode_string_columns(features.copy())
        features_std = standardize_non_binary_columns(features_enc)
        x = torch.tensor(features_std.values, dtype=torch.float)
    else:
        x = torch.eye(labels.shape[0])

    y = torch.tensor(labels.values, dtype=torch.long).squeeze()

    data = Data(x=x, edge_index=edge_index, y=y)

    return data



# --------------------------------------------------------------------------- #
# Public loader                                                               #
# --------------------------------------------------------------------------- #
def load_data_from_edgelist(
    edge_path: str | Path,
    *,
    node_feat_path: Optional[str | Path] = None,
    label_path: Optional[str | Path] = None,
    augment_partitions: bool = False,
    partition_path: Optional[str | Path] = None,
    connect_partition_edges: bool = False,
    partition_edge_path: Optional[str | Path] = None,
    partition_id_offset: int = 0,
    allow_orphan_singletons: bool = True,
    test_ratio: float = 0.1,
    val_ratio: float = 0.05,
    random_state: int = 42,
    undirected: bool = True,
    ) -> Tuple[Data, Data, Data, Data]:
    """
    Returns `(train_data, val_data, test_data)` after a `RandomLinkSplit`
    **or** a single `Data` object if `val_ratio = test_ratio = 0`.
    """
    # ------------------------------------------------------------------ #
    # 0) raw I/O                                                         #
    # ------------------------------------------------------------------ #
    edges_df = _read_edge_file(edge_path)

    # ---- load node features (optional) ---- #
    if node_feat_path:
        features = pd.read_csv(node_feat_path, header=None)
    else:
        features = None
        
    # ---- load labels (optional) ---- #
    if label_path:
        labels = pd.read_csv(label_path, header=None)
    else:
        labels = None

    
    
    data = _build_pyg_data(edges_df, features, labels)

    # ─── create node‐classification masks if we have labels ───────────
    if labels is not None:
        num_nodes = data.num_nodes
        # reproducible shuffle
        # keep node splits reproducible across runs/methods
        g = torch.Generator().manual_seed(random_state)
        perm = torch.randperm(num_nodes, generator=g)

        # split sizes
        n_test  = int(test_ratio * num_nodes)
        n_val   = int(val_ratio  * num_nodes)
        n_train = num_nodes - n_val - n_test

        # pick indices
        train_idx = perm[:n_train]
        val_idx   = perm[n_train : n_train + n_val]
        test_idx  = perm[n_train + n_val : n_train + n_val + n_test]

        # build boolean masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask  = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[train_idx] = True
        val_mask[val_idx]     = True
        test_mask[test_idx]   = True

        data.train_mask = train_mask
        data.val_mask   = val_mask
        data.test_mask  = test_mask

    # ------------------------------------------------------------------ #
    # 2) train / val / test split for LINK PREDICTION                    #
    # ------------------------------------------------------------------ #
    if val_ratio == 0 and test_ratio == 0:
        return data  # raw graph only

    splitter = RandomLinkSplit(
        is_undirected=True,
        num_val=val_ratio,
        num_test=test_ratio,
        add_negative_train_samples=True,
        split_labels=False,
    )

    train_data, val_data, test_data = splitter(data)

    # ---------- optional graph augmentation --------------------------
    if augment_partitions:
        if partition_path is None:
            raise ValueError("augment_partitions=True but no partition_path given")
        part_map, k = load_partitions(partition_path, id_offset=partition_id_offset)
        data = augment_graph(
            data,
            partition_map=part_map,
            num_partitions=k,
            connect_partition_edges=connect_partition_edges,
            partition_edge_path=partition_edge_path,
            allow_orphan_singletons=allow_orphan_singletons,
        )

    return data, train_data, val_data, test_data
