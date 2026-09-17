import argparse
import csv
from pathlib import Path
from time import time
import random

import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, GINConv
import numpy as np
import copy

from model import GenericModel

from trainers import (
    train_node_clf,
    eval_node_clf,
    train_link_pred,
    eval_link_pred,
)
from data_loader import load_data_from_edgelist

from utils import mad_global

SRC_DIR = Path(__file__).resolve().parent
REPRO_ROOT = SRC_DIR.parent
DEFAULT_NC_OUT = REPRO_ROOT / "results" / "node_classification_results.csv"
DEFAULT_LP_OUT = REPRO_ROOT / "results" / "link_prediction_results.csv"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def feature_presence(args):
    if args.feat_path:
        return True
    else:
        return False

def determine_augmentation_level(args):
    if not args.augment_partitions:
        return 0
    elif not args.connect_partition_edges:
        return 1
    else:
        return 2

def extract_dataset_name(edge_path):
    # assumes path like: "Cora/Cora.edgelist"
    return Path(edge_path).parent.name

def extract_epsilon(partition_path):
    if not partition_path:
        return None
    for part in partition_path.split("P"):
        if part.isdigit():
            return int(part)
    return None



def parse_args():
    p = argparse.ArgumentParser("BEGONE pipeline")
    p.add_argument("--trial", help="Number of trial in experiment")
    p.add_argument("--random_state", type=int, help="Random seed", default=42)

    p.add_argument("--edge_path", required=True, help="Path to edgelist CSV/TSV")
    p.add_argument("--feat_path", default=None, help="Optional node‑features CSV")
    p.add_argument("--label_path", default=None, help="Optional node‑label CSV")
    p.add_argument("--dataset_name", default=None,
                   help="Optional dataset name override for result metadata")

    p.add_argument("--model", type=str, default="GCN")
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--layers", type=int, default=2)
    #p.add_argument("--embed", type=int, default=64)

    p.add_argument("--epochs_clf", type=int, default=100)
    p.add_argument("--epochs_lp", type=int, default=100)

    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=5e-4)

    p.add_argument("--augment_partitions", action="store_true",
                   help="Add partition nodes and star edges")
    p.add_argument("--partition_path", default=None,
                   help="Text file where each line lists node IDs of a partition")
    p.add_argument("--connect_partition_edges", action="store_true",
                   help="Also connect partition nodes via their own edgelist")
    p.add_argument("--partition_edge_path", default=None,
                   help="Edge list connecting partitions (indices match row order)")
    p.add_argument("--partition_id_offset", type=int, default=-1,
                   help="Add this to every ID read from partition file "
                        "(use -1 when IDs start at 1)")
    p.add_argument("--strict_partitions", action="store_true",
                   help="Error out if any node is missing from the partition file "
                        "(default: create singleton partitions)")

    p.add_argument("--hits_K", nargs="+", default=[1,5,10,20,50,100])
    p.add_argument("--mrr_K", nargs="+", default=["all"])
    p.add_argument("--nc_out_path", type=Path, default=DEFAULT_NC_OUT,
                   help="Output CSV path for node classification metrics")
    p.add_argument("--lp_out_path", type=Path, default=DEFAULT_LP_OUT,
                   help="Output CSV path for link prediction metrics")
    p.add_argument("--skip_lp", action="store_true",
                   help="If set, skip link prediction training/evaluation entirely")
    p.add_argument("--method_tag", type=str, default=None,
                   help="Optional rewiring/method tag stored in output metadata")

    return p.parse_args()

patience = 50

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SEED = args.random_state  # or fixed constant
    set_seed(SEED)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    full_data, train_data_lp, val_data_lp, test_data_lp = load_data_from_edgelist(
        args.edge_path,
        node_feat_path=args.feat_path,
        label_path=args.label_path,
        # new ↓↓↓
        augment_partitions=args.augment_partitions,
        partition_path=args.partition_path,
        connect_partition_edges=args.connect_partition_edges,
        partition_edge_path=args.partition_edge_path,
        partition_id_offset=args.partition_id_offset,
        allow_orphan_singletons=not args.strict_partitions,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    num_classes = int(full_data.y.max().item() + 1) if full_data.y is not None else 0

    if args.model=="GCN":
        convolution_type=GCNConv
    elif args.model=="GAT":
        convolution_type=GATConv
    elif args.model=="GIN":
        convolution_type=GINConv
    else:
        raise Exception()

    model = GenericModel(
        convolution_type=convolution_type,
        in_channels=full_data.num_features,
        hidden_channels=args.hidden,
        #embed_dim=args.embed,
        num_layers=args.layers,
        num_classes=num_classes,
    ).to(device)

    # ------------------------------------------------------------------
    # 1. Node classification (only if labels are present)
    # ------------------------------------------------------------------
    clf_metrics = {"train_acc": None, "val_acc": None, "test_acc": None}
    if full_data.y is not None:
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion_clf = nn.CrossEntropyLoss()
        start = time()
        best_val_loss_clf = float('inf')
        #best_val_acc_clf = 0.
        for epoch in range(1, args.epochs_clf + 1):
            loss = train_node_clf(model, full_data, optim, criterion_clf, device)

            tr, va, te, val_loss = eval_node_clf(
                model, full_data, criterion_clf, device
            )
            # if epoch % 25 == 0 or epoch == args.epochs_clf:
            #     print(f"[NC] epoch={epoch:<3d} loss={loss:.4f} train={tr:.3f} val={va:.3f}")
            # Early stopping inline
            if val_loss < best_val_loss_clf:
                best_val_loss_clf = val_loss
            # if va > best_val_acc_clf:
            #     best_val_acc_clf = va
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    model.load_state_dict(best_state)
                    tr, va, te, val_loss = eval_node_clf(
                        model, full_data, criterion_clf, device
                    )
                    print(f"[NC] Early stopping at epoch {epoch}, best_val_loss={best_val_loss_clf:.4f}")
                    # print(f"[NC] Early stopping at epoch {epoch}, best_val_acc={best_val_acc_clf:.4f}")
                    break
                #print(f"[NC] epoch={epoch:<3d} loss={loss:.4f} train={tr:.3f} val={va:.3f}")
        print(f"[NC] test={te:.3f}")
        # mad_global
        emb = model(full_data.x.to(device), full_data.edge_index.to(device), return_emb=True)
        clf_mad = mad_global(emb)
        clf_metrics = {
                        "train_acc": tr,
                        "val_acc": va,
                        "test_acc": te,
                        "mad_val": clf_mad,
                        "nc_time": time() - start
                       }

    lp_metrics = None
    if not args.skip_lp:
        # ------------------------------------------------------------------
        # 2. Link prediction
        # ------------------------------------------------------------------
        # optim_lp = torch.optim.Adam(model.parameters(), lr=args.lr)
        # criterion_lp = nn.BCEWithLogitsLoss()
        # start_lp = time()
        # for epoch in range(1, args.epochs_lp + 1):
        #     loss, auc_train, ap_train = train_link_pred(model, full_data, train_data_lp, optim_lp, criterion_lp, device)
        #     if epoch % 25 == 0 or epoch == args.epochs_lp:
        #         auc_val, ap_val = eval_link_pred(model, full_data, val_data_lp, device)
        optim_lp     = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion_lp  = nn.BCEWithLogitsLoss()
        start_lp      = time()
        # Early stopping LP
        best_val_loss_lp = float('inf')
        # best_val_metric_lp = 0.
        no_improve_lp    = 0

        for epoch in range(1, args.epochs_lp + 1):
            loss_train_lp, auc_train, ap_train = train_link_pred(
                model, full_data, train_data_lp,
                optim_lp, criterion_lp, device
            )

            # validazione & early stopping
            lp_val_metrics = eval_link_pred(
                model, full_data, val_data_lp,
                criterion_lp, device,
                hits_K_list=args.hits_K,
                mrr_K_list=args.mrr_K
            )
            if lp_val_metrics["val_loss"] < best_val_loss_lp:
                best_val_loss_lp = lp_val_metrics["val_loss"]
            # if np.mean([auc_val, ap_val]) > best_val_metric_lp:
            #     best_val_metric_lp = np.mean([auc_val, ap_val])
                best_state = copy.deepcopy(model.state_dict())
                no_improve_lp    = 0
            else:
                no_improve_lp += 1
                if no_improve_lp >= patience:
                    model.load_state_dict(best_state)
                    print(f"[LP] Early stopping at epoch {epoch}, best_val_loss={best_val_loss_lp:.4f}")
                    # print(f"[LP] Early stopping at epoch {epoch}, best_val_metric={best_val_metric_lp:.4f}")
                    break
            # if epoch % 25 == 0 or epoch == args.epochs_lp:
            #     print(f"[LP] train_auc={auc_train:.3f} train_ap={ap_train:.3f} val_auc={auc_val:.3f} val_ap={ap_val:.3f}")
        #auc_test, ap_test, _ = eval_link_pred(model, full_data, test_data_lp, criterion_lp, device)
        #print(f"[LP] test_auc={auc_test:.3f} test_ap={ap_test:.3f}")

        # mad_global
        emb = model(full_data.x.to(device), full_data.edge_index.to(device), return_emb=True)
        link_mad = mad_global(emb)

        # --- Test set evaluation ---
        lp_test_metrics = eval_link_pred(
            model, full_data, test_data_lp,
            criterion_lp, device,
            hits_K_list=args.hits_K,
            mrr_K_list=args.mrr_K
        )

        #print(f"[LP] test_auc={lp_test_metrics['AUC']:.3f} test_ap={lp_test_metrics['AP']:.3f}")
        print(f"[LP]:\n{lp_test_metrics}")

        lp_metrics = {
            "train_auc": auc_train,
            "train_ap": ap_train,
            "mad": link_mad,
            "lp_time": time() - start_lp,
        }
        # aggiungi tutte le metriche di test con prefisso
        lp_metrics.update({f"test_{k.lower()}": v for k, v in lp_test_metrics.items()})

    

    # ------------------------------------------------------------------
    # Persist results (DYNAMIC version) ---------------------------------
    # ------------------------------------------------------------------
    augmentation_level = determine_augmentation_level(args)
    dataset_name = args.dataset_name if args.dataset_name else extract_dataset_name(args.edge_path)
    epsilon_value = extract_epsilon(args.partition_path)
    feat_presence = feature_presence(args)

    # Common metadata for both files
    metadata = {
        "model": args.model,
        "layers": args.layers,
        "dataset": dataset_name,
        "feat_presence": feat_presence,
        "augmentation": augmentation_level,
        "epsilon": epsilon_value,
        "trial": args.trial,
        "method_tag": args.method_tag,
    }

    # -----------------------------
    # File 1: Node Classification
    # -----------------------------
    nc_out_path = Path(args.nc_out_path)
    nc_out_path.parent.mkdir(parents=True, exist_ok=True)
    nc_row = {**metadata, **{"test_acc": clf_metrics["test_acc"], "mad": clf_metrics["mad_val"]}}

    # scrittura dinamica
    file_exists = nc_out_path.exists()
    with nc_out_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(nc_row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(nc_row)

    # -----------------------------
    # File 2: Link Prediction
    # -----------------------------
    if lp_metrics is not None:
        lp_out_path = Path(args.lp_out_path)
        lp_out_path.parent.mkdir(parents=True, exist_ok=True)
        # lp_metrics già contiene train/test metrics + dinamicamente Hits@K, MRR@K, ecc.
        lp_row = {**metadata, **lp_metrics}

        file_exists = lp_out_path.exists()
        with lp_out_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(lp_row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(lp_row)

    print()


if __name__ == "__main__":
    main()
