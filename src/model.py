from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import torch_geometric
from torch_geometric.nn import GINConv


class GenericModel(nn.Module):

    def __init__(
        self,
        convolution_type: torch_geometric.nn.conv.MessagePassing,
        in_channels: int,
        hidden_channels: int,
        #embed_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be ≥ 2 — set at least one hidden layer and an output layer")

        layers: List[nn.Module] = []

        def _gin_mlp(in_ch, out_ch):
            # A tiny MLP for GIN (standard: Linear-ReLU-Linear)
            return nn.Sequential(
                nn.Linear(in_ch, out_ch),
                nn.ReLU(),
                nn.Linear(out_ch, out_ch),
            )

        # input → hidden
        if convolution_type is GINConv:
            layers.append(GINConv(_gin_mlp(in_channels, hidden_channels)))
            # hidden → hidden (repeated)
            for _ in range(num_layers - 2):
                layers.append(GINConv(_gin_mlp(hidden_channels, hidden_channels)))
        else:
            layers.append(convolution_type(in_channels, hidden_channels))
            # hidden → hidden (repeated)
            layers += [convolution_type(hidden_channels, hidden_channels) for _ in range(num_layers - 2)]
        
        # hidden → embedding
        #layers.append(convolution_type(hidden_channels, embed_dim))
        self.convs = nn.ModuleList(layers)

        #self.classifier = nn.Linear(embed_dim, num_classes) if num_classes > 0 else None
        self.classifier = nn.Linear(hidden_channels, num_classes) if num_classes > 0 else None
        self.dropout = dropout

    # ------------------------------------------------------------------
    # Forward / helpers
    # ------------------------------------------------------------------
    def forward(self, x: Tensor, edge_index: Tensor, *, return_emb: bool = False) -> Tensor:
        """Compute embeddings **or** logits.

        If *return_emb* is ``True``: returns a tensor of shape
        *(N, embed_dim)* with node embeddings.

        Otherwise: returns raw *logits* of shape *(N, num_classes)* to be
        consumed with ``F.log_softmax`` + NLLLoss (node‑classification).
        """
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)

        if return_emb:
            return x  # (N, embed_dim)
        return self.classifier(x)  # (N, num_classes) — raw logits

    # --------------------------- task‑specific wrappers -----------------
    def classify(self, data) -> Tensor:
        """Return logits for node classification."""
        return self.forward(data.x, data.edge_index)


    def link_logits(self, data, edge_label_index):
        """
        Calcola uno score scalar per ciascun arco in edge_label_index,
        utilizzando il dot-product delle embedding dei due endpoint.
        Ritorna un vettore 1-D di lunghezza M.
        """
        # 1) ottieni embedding dei nodi
        z = self.forward(data.x, data.edge_index)
        # 2) split index
        src, dst = edge_label_index
        # 3) dot-product → vettore [M]
        return (z[src] * z[dst]).sum(dim=-1)
