import torch, torch.nn as nn
class ViFBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int = 8, n_layers: int = 2, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        enc = nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads, dim_feedforward=int(dim*mlp_ratio),
                                         dropout=dropout, batch_first=True, activation='gelu', norm_first=True)
        self.tr = nn.TransformerEncoder(enc, num_layers=n_layers)
    def forward(self, R, I):
        x = torch.cat([R, I], dim=1)
        x = self.tr(x)
        return x[:, : R.size(1), :]
