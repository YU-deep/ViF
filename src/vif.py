from typing import Optional, List, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass


@dataclass
class ViFConfig:
    hidden_dim: int = 1024
    num_relay_tokens: int = 16
    unimodal_salience: float = 0.3
    tolerance: float = 0.05
    tau: float = 0.8
    alpha_middle: float = 0.1
    alpha_deep: float = 0.3
    keynorm_buffer: int = 3
    nhead: int = 16
    # device: str = "cuda" if torch.cuda.is_available() else "cpu"


"""
    modified from DeLighT
    more information in https://github.com/sacmehta/delight/tree/master
"""
class LightweightTransformerBlock(nn.Module):
    """
    Lightweight Transformer block f(·), adapted from DeLighT
    """

    def __init__(self, hidden_dim: int, nhead: int = 4,
                 ffn_red: int = 4, dropout: float = 0.1,
                 norm_type: str = "ln", act_type: str = "swish"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.dropout = dropout
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim,
                                               num_heads=nhead,
                                               dropout=dropout,
                                               batch_first=True)
        ffn_hidden = hidden_dim // ffn_red
        self.linear1 = nn.Linear(hidden_dim, ffn_hidden)
        self.linear2 = nn.Linear(ffn_hidden, hidden_dim)

        if act_type == "relu":
            self.activation = nn.ReLU()
        elif act_type == "gelu":
            self.activation = nn.GELU()
        else:  # default swish
            self.activation = nn.SiLU()

        if norm_type == "ln":
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
        elif norm_type == "bn":
            self.norm1 = nn.BatchNorm1d(hidden_dim)
            self.norm2 = nn.BatchNorm1d(hidden_dim)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [batch, seq, hidden_dim]
        residual = x
        x_norm = self.norm1(x)
        attn_output, _ = self.self_attn(x_norm, x_norm, x_norm,
                                        key_padding_mask=key_padding_mask)
        x = residual + self.dropout_layer(attn_output)

        # FFN
        residual = x
        x_norm = self.norm2(x)
        x_ffn = self.linear2(self.dropout_layer(self.activation(self.linear1(x_norm))))
        x = residual + self.dropout_layer(x_ffn)
        return x


class ViFModule(nn.Module):
    def __init__(self, cfg: ViFConfig):
        super().__init__()
        self.cfg = cfg
        # could use different transformer blocks
        self.ctx_block = LightweightTransformerBlock(cfg.hidden_dim,
                                                     nhead=cfg.nhead,
                                                     ffn_red=cfg.ffn_red if hasattr(cfg, "ffn_red") else 4,
                                                     dropout=cfg.dropout if hasattr(cfg, "dropout") else 0.1,
                                                     norm_type="ln",
                                                     act_type="swish").to(cfg.device)
        self.pe = None
        self.projector = None


    @staticmethod
    def _smooth_series(series: np.ndarray, window: int = 3) -> np.ndarray:
        if window <= 1:
            return series
        pad = window // 2
        padded = np.pad(series, (pad, pad), mode='edge')
        kernel = np.ones(window) / window
        sm = np.convolve(padded, kernel, mode='valid')
        return sm

    def _is_unimodal(self, series: np.ndarray, salience: float, tol: float) -> bool:
        """
        detect single peak
        """
        if np.all(series <= 0):
            return False
        s = self._smooth_series(series, window=3)
        peak_idx = int(np.argmax(s))
        peak_val = float(s[peak_idx])
        # relative salience
        if peak_val < salience * float(np.max(s)):
            return False
        before = s[:peak_idx + 1]
        after = s[peak_idx:]

        # allow small violations
        def frac_violations(arr, increasing=True):
            diffs = np.diff(arr)
            if increasing:
                violations = np.sum(diffs < -tol)
            else:
                violations = np.sum(diffs > tol)
            return violations / max(1, len(diffs))

        if frac_violations(before, increasing=True) > 0.25:
            return False
        if frac_violations(after, increasing=False) > 0.25:
            return False

        # the peak to be not at the boundaries
        if peak_idx == 0 or peak_idx == (len(s) - 1):
            return False
        return True

    def select_by_attention(self,
                            vision_attentions_mid: torch.Tensor,
                            max_relay_tokens: Optional[int] = None,
                            salience: Optional[float] = None,
                            tol: Optional[float] = None) -> List[int]:
        """
        select unimodal vision tokens based on attention allocations in middle layers
        """
        salience = salience if salience is not None else self.cfg.unimodal_salience
        tol = tol if tol is not None else self.cfg.tolerance
        L_mid, m = vision_attentions_mid.shape
        vis_np = vision_attentions_mid.detach().cpu().numpy()
        unimodal_idxs = []
        for tok in range(m):
            series = vis_np[:, tok]
            if self._is_unimodal(series, salience=salience, tol=tol):
                unimodal_idxs.append(tok)
        # keep top-k by peak magnitude
        if len(unimodal_idxs) == 0:
            return []
        peaks = [(tok, float(np.max(vis_np[:, tok]))) for tok in unimodal_idxs]
        peaks.sort(key=lambda x: -x[1])
        k = max_relay_tokens if max_relay_tokens is not None else self.cfg.num_relay_tokens
        selected = [p[0] for p in peaks[:k]]
        return selected

    def select_by_keynorm(self,
                          key_vectors: torch.Tensor,
                          grid_shape: Optional[Tuple[int, int]] = None,
                          max_relay_tokens: Optional[int] = None,
                          buffer_radius: Optional[int] = None) -> List[int]:
        """
        Key-Norm alternative
        """
        buffer_radius = buffer_radius if buffer_radius is not None else self.cfg.keynorm_buffer
        device = key_vectors.device
        n_vis = key_vectors.shape[0]
        norms = torch.norm(key_vectors, p=2, dim=1).detach().cpu().numpy()

        k = max_relay_tokens if max_relay_tokens is not None else self.cfg.num_relay_tokens
        topk = int(min(k, n_vis))
        top_idxs = list(np.argsort(-norms)[:topk])

        if grid_shape is None:
            s = int(math.sqrt(n_vis))
            if s * s == n_vis:
                H, W = s, s
            else:
                # fallback: 1 x n_vis
                H, W = 1, n_vis
        else:
            H, W = grid_shape

        def idx_to_rc(idx):
            return divmod(idx, W)

        selected = set(top_idxs)
        # add buffer
        for idx in top_idxs:
            r, c = idx_to_rc(idx)
            for dr in range(-buffer_radius, buffer_radius + 1):
                for dc in range(-buffer_radius, buffer_radius + 1):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < H and 0 <= cc < W:
                        selected.add(rr * W + cc)
                    if len(selected) >= (k * 4):
                        break
                if len(selected) >= (k * 4):
                    break
            if len(selected) >= (k * 4):
                break
        selected = sorted(list(selected))
        selected = selected[: max_relay_tokens if max_relay_tokens is not None else self.cfg.num_relay_tokens]
        return selected

    def contextualize_relay_tokens(self,
                                   vision_tokens: torch.Tensor,
                                   instruction_tokens: torch.Tensor,
                                   relay_idxs: List[int]) -> torch.Tensor:

        if len(relay_idxs) == 0:
            return torch.zeros((0, vision_tokens.shape[-1]), device=vision_tokens.device)
        device = vision_tokens.device
        R = vision_tokens[relay_idxs]  # [n, hidden_dim]
        R = R.unsqueeze(0) if R.dim() == 2 else R
        I = instruction_tokens.unsqueeze(0) if instruction_tokens.dim() == 2 else instruction_tokens
        x = torch.cat([R, I], dim=1).to(device)
        x = x.to(self.cfg.device)
        out = self.ctx_block(x)  # [1, seq, hidden_dim]
        n = R.shape[1] if R.dim() == 3 else R.shape[0]
        out = out.squeeze(0)[:n, :]
        return out

    @staticmethod
    def softmax_with_temperature(scores: torch.Tensor, tau: float) -> torch.Tensor:
        # scores: [..., seq_len]
        return F.softmax(scores / tau, dim=-1)

    def apply_attention_reallocation(self,
                                     attention_scores: torch.Tensor,
                                     layer_idx: int,
                                     layer_type: str,
                                     vision_token_indices: List[int],
                                     inactive_visual_indices: List[int],
                                     instruction_indices: List[int],
                                     tau: Optional[float] = None,
                                     alpha: Optional[float] = None) -> torch.Tensor:
        """
        modify attention scores for one layer (in-place safe copy returned).
        """
        tau = tau if tau is not None else self.cfg.tau
        if alpha is None:
            alpha = self.cfg.alpha_middle if layer_type == 'middle' else self.cfg.alpha_deep
        # unify shapes
        had_heads = False
        if attention_scores.dim() == 3:
            # [num_heads, S, S]
            s = attention_scores.mean(dim=0)
            had_heads = True
        else:
            s = attention_scores.clone()
        seq_len = s.shape[-1]
        #  temperature scaling with softmax
        A = self.softmax_with_temperature(s, tau)
        mask_collect = torch.zeros((seq_len, seq_len), device=A.device)
        idxs_collect = set(inactive_visual_indices) | set(instruction_indices)
        if len(idxs_collect) > 0:
            mask_collect[:, list(idxs_collect)] = 1.0
        collected_mass = (A * mask_collect).sum(dim=-1) * alpha
        denom = (A * mask_collect).sum(dim=-1, keepdim=True)
        denom = denom.clamp(min=1e-12)
        # remove alpha
        removed = (A * mask_collect) * (alpha * (1.0 / denom))
        A_new = A.clone()
        A_new = A_new - removed
        # reallocate
        V_set = set(vision_token_indices)
        V_inactive_set = set(inactive_visual_indices)
        V_active = sorted(list(V_set - V_inactive_set))
        if len(V_active) == 0:
            s_out = torch.log(A_new.clamp(min=1e-12))
        else:
            # distribute
            denom2 = A[:, V_active].sum(dim=-1, keepdim=True).clamp(min=1e-12)
            add_portion = (A[:, V_active] / denom2) * collected_mass.unsqueeze(-1)
            A_new[:, V_active] = A_new[:, V_active] + add_portion
            # re-normalize
            A_new = A_new / A_new.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            s_out = torch.log(A_new.clamp(min=1e-12))
        if had_heads:
            out = s_out.unsqueeze(0).repeat(attention_scores.shape[0], 1, 1)
        else:
            out = s_out
        return out


def _synthetic_test():
    cfg = ViFConfig(hidden_dim=256, num_relay_tokens=8, unimodal_salience=0.3, tau=0.8,
                    alpha_middle=0.1, alpha_deep=0.3, keynorm_buffer=3, device="cpu")
    model = ViFModule(cfg)
    device = cfg.device
    n_vis = 256
    hidden_dim = cfg.hidden_dim
    vis_tokens = torch.randn((n_vis, hidden_dim), device=device)
    inst_tokens = torch.randn((12, hidden_dim), device=device)
    L_mid = 4
    vis_att_mid = torch.rand((L_mid, n_vis), device=device) * 0.05
    for tok in range(12):
        peak_layer = np.random.randint(1, L_mid - 1)
        base = 0.02 * np.random.rand()
        series = np.array([base + (0.08 * math.exp(-((l - peak_layer) ** 2))) for l in range(L_mid)])
        vis_att_mid[:, tok] = torch.tensor(series, dtype=torch.float32)
    selected = model.select_by_attention(vis_att_mid, max_relay_tokens=cfg.num_relay_tokens)
    print("Selected (attention) indices:", selected)
    relay_vecs = model.contextualize_relay_tokens(vis_tokens, inst_tokens, selected)
    print("Relay vecs shape:", relay_vecs.shape)
    num_heads = 8
    S = 1 + n_vis + inst_tokens.shape[
        0] + 10
    att = torch.rand((num_heads, S, S), device=device)
    vision_indices = list(range(1, 1 + n_vis))
    inactive = vision_indices[::20]  # every 20th vision token inactive (toy)
    inst_indices = list(range(S - inst_tokens.shape[0], S))
    updated_scores = model.apply_attention_reallocation(att, layer_idx=0, layer_type='middle',
                                                        vision_token_indices=vision_indices,
                                                        inactive_visual_indices=inactive,
                                                        instruction_indices=inst_indices)
    print("Updated scores shape:", updated_scores.shape)
    print("Synthetic test completed.")


if __name__ == "__main__":
    _synthetic_test()
