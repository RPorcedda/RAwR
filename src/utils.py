import torch
import torch.nn.functional as F

def mad_global(embeddings: torch.Tensor) -> float:
    normed = F.normalize(embeddings, p=2, dim=1)
    sim = torch.mm(normed, normed.t())
    dist = 1.0 - sim
    mask = ~torch.eye(dist.size(0), dtype=torch.bool, device=dist.device)
    vals = dist[mask]
    return vals.mean().item()