import torch, torch.nn as nn
from .vif_block import ViFBlock
from ..utils.selection import select_relay_tokens, aggregate_keynorm
from ..utils.reallocation import softmax_with_temperature, reallocate_middle, reallocate_deep
class ViFWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, dim: int, relay_ratio=0.02, omega=0.3, tau=0.8, alpha_mid=0.2, alpha_deep=0.1, use_keynorm=True):
        super().__init__(); self.base=base_model; self.dim=dim; self.vif=ViFBlock(dim)
        self.relay_ratio=relay_ratio; self.omega=omega; self.tau=tau; self.alpha_mid=alpha_mid; self.alpha_deep=alpha_deep; self.use_keynorm=use_keynorm
    def forward(self, batch):
        out = self.base(batch)
        vision_mask = out['vision_mask']; instr_mask = out['instruction_mask']; hidden = out['hidden_states']
        B,T,D = hidden.shape
        vision_tokens = hidden[vision_mask].reshape(B, -1, D); instr_tokens = hidden[instr_mask].reshape(B, -1, D)
        mid_signal = out.get('mid_attn_scores', None); mid_keys = out.get('mid_key_states', None)
        prom = aggregate_keynorm(mid_keys) if (mid_signal is None or self.use_keynorm) else mid_signal
        idx, mask = select_relay_tokens(prom, self.relay_ratio, self.omega)
        bidx = torch.arange(B, device=hidden.device).unsqueeze(-1)
        R = vision_tokens[bidx, idx, :]; R_hat = self.vif(R, instr_tokens); vision_tokens[bidx, idx, :] = R_hat
        new_hidden = hidden.clone(); new_hidden[vision_mask] = vision_tokens.reshape(-1, D)
        mid_scores = out.get('mid_scores', None); deep_scores = out.get('deep_scores', None)
        if mid_scores is not None:
            mid_probs = softmax_with_temperature(mid_scores, self.tau)
            mid_probs = reallocate_middle(mid_probs, out['inactive_vision_mask'], instr_mask, self.alpha_mid); out['mid_probs']=mid_probs
        if deep_scores is not None:
            deep_probs = softmax_with_temperature(deep_scores, self.tau)
            deep_probs = reallocate_deep(deep_probs, vision_mask, instr_mask, self.alpha_deep); out['deep_probs']=deep_probs
        out['hidden_states'] = new_hidden; return out
