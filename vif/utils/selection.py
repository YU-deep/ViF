import torch
def select_relay_tokens(sig: torch.Tensor, relay_ratio: float, omega: float=0.3):
    B, N = sig.shape; k = max(1, int(N*relay_ratio))
    x = (sig - sig.mean(dim=1, keepdim=True)) / (sig.std(dim=1, keepdim=True)+1e-6)
    x = torch.sigmoid(x / max(1e-6, omega))
    vals, idx = torch.topk(x, k, dim=1)
    mask = torch.zeros_like(x, dtype=torch.bool); mask.scatter_(1, idx, True)
    return idx, mask
def aggregate_keynorm(K: torch.Tensor) -> torch.Tensor:
    B,N,D = K.shape
    return torch.norm(K.view(B,N,D), dim=-1)
