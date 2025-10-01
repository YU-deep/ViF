import torch, torch.nn.functional as F
def softmax_with_temperature(scores, tau: float):
    return F.softmax(scores / max(1e-6, tau), dim=-1)
def reallocate_middle(attn, inactive_mask, instr_mask, alpha: float):
    col = alpha * (attn * (inactive_mask | instr_mask)).sum(dim=-1, keepdim=True)
    active = ~(inactive_mask | instr_mask)
    give = torch.where(active, torch.ones_like(attn), torch.zeros_like(attn))
    give = give / (give.sum(dim=-1, keepdim=True)+1e-6)
    new = attn - alpha*attn*(inactive_mask|instr_mask) + col*give
    return new / (new.sum(dim=-1, keepdim=True)+1e-6)
def reallocate_deep(attn, vision_mask, instr_mask, alpha: float):
    col = alpha * (attn * vision_mask).sum(dim=-1, keepdim=True)
    give = torch.where(instr_mask, torch.ones_like(attn), torch.zeros_like(attn))
    give = give / (give.sum(dim=-1, keepdim=True)+1e-6)
    new = attn - alpha*attn*vision_mask + col*give
    return new / (new.sum(dim=-1, keepdim=True)+1e-6)
