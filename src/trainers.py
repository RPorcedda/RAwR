from typing import Tuple

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from torch import nn, Tensor
from torch_geometric.data import Data

def _accuracy(pred: Tensor, target: Tensor) -> float:
    return accuracy_score(target.cpu(), pred.cpu())


# ----------------------------------------------------------------------
# Node classification
# ----------------------------------------------------------------------

def train_node_clf(model, data, optimizer, criterion: nn.Module, device) -> float:
    """One step of supervised training for node classification."""
    model.train()
    optimizer.zero_grad()
    #logits = model.classify(data.to(device))
    #loss = criterion(logits[data.train_mask], data.y[data.train_mask].to(device))
    out = model.classify(data.to(device))
    # mantieni solo i nodi addestrabili (masked & y>=0)
    mask = data.train_mask & (data.y.to(device) >= 0)
    loss = criterion(out[mask], data.y.to(device)[mask])
    loss.backward()
    optimizer.step()
    return float(loss.item())


#def eval_node_clf(model, data, device) -> Tuple[float, float, float]:
def eval_node_clf(model, data, criterion: nn.Module, device) -> Tuple[float, float, float, float]:
    """Return (train_acc, val_acc, test_acc)."""
    model.eval()
    with torch.no_grad():
        logits = model.classify(data.to(device))
    pred = logits.argmax(dim=1)

    # filtra y>=0
    train_mask = data.train_mask & (data.y >= 0)
    val_mask   = data.val_mask   & (data.y >= 0)
    test_mask  = data.test_mask  & (data.y >= 0)
    tr = _accuracy(pred[train_mask], data.y[train_mask])
    va = _accuracy(pred[val_mask],   data.y[val_mask])
    te = _accuracy(pred[test_mask],  data.y[test_mask])
    #return tr, va, te
    # Calcolo della validation loss
    val_loss = float(criterion(logits[val_mask],
        data.y[val_mask].to(device)).item())
    return tr, va, te, val_loss


# ----------------------------------------------------------------------
# Link prediction
# ----------------------------------------------------------------------

def train_link_pred(model, data, split_data, optimizer, criterion: nn.Module, device) -> float:
    """One optimisation step for link prediction (binary classification)."""
    model.train()
    optimizer.zero_grad()

    logits = model.link_logits(
        data.to(device),
        split_data.edge_label_index.to(device),
    )

    labels     = split_data.edge_label.to(device).float()
    loss       = criterion(torch.sigmoid(logits), labels)
    probs      = torch.sigmoid(logits).cpu().detach().numpy()

    auc = roc_auc_score(labels.cpu().detach().numpy(), probs)
    ap  = average_precision_score(labels.cpu().detach().numpy(), probs)
    loss.backward()
    optimizer.step()
    return float(loss.item()), auc, ap

def eval_link_pred(
    model,
    data,
    split_data,
    criterion: nn.Module,
    device,
    hits_K_list,
    mrr_K_list
) -> dict:
    """
    Valutazione link prediction:
    ritorna un dizionario con AUC, AP, validation loss, Hits@K e MRR@K.
    """

    model.eval()
    with torch.no_grad():
        logits = model.link_logits(
            data.to(device),
            split_data.edge_label_index.to(device),
        )

    labels = split_data.edge_label.to(device).float()
    val_loss = criterion(torch.sigmoid(logits), labels).item()
    probs = torch.sigmoid(logits).cpu()

    auc = roc_auc_score(labels.cpu(), probs)
    ap  = average_precision_score(labels.cpu(), probs)

    pos_scores = logits[labels == 1]
    neg_scores = logits[labels == 0]

    all_neg = neg_scores.view(-1)

    metrics = {"AUC": auc, "AP": ap, "val_loss": val_loss}

    ranking_list = []
    for pos in pos_scores:
        optimistic_rank = (all_neg >= pos).sum().item()
        pessimistic_rank = (all_neg > pos).sum().item()
        rank = 0.5 * (optimistic_rank + pessimistic_rank) + 1
        ranking_list.append(rank)

    ranking_list = torch.tensor(ranking_list, dtype=torch.float, device=device)

    # Hits@K
    for K in hits_K_list:
        if K == "all":
            Kval = all_neg.numel() + 1
        else:
            Kval = int(K)
        metrics[f"Hits@{K}"] = (ranking_list <= Kval).float().mean().item()

    # MRR@K
    for K in mrr_K_list:
        if K == "all":
            rr = 1.0 / ranking_list.float()  # standard globale
        else:
            rr = (1.0 / ranking_list.float()) * (ranking_list <= int(K))
        metrics[f"MRR@{K}"] = rr.mean().item()

    return metrics
