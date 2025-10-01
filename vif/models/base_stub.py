import torch, torch.nn as nn, torch.nn.functional as F
class BaseVLMStub(nn.Module):
    def __init__(self, dim=1024):
        super().__init__(); self.dim=dim
        self.projector = nn.Linear(768, dim)
        self.llm = nn.Transformer(d_model=dim, nhead=8, num_encoder_layers=2, num_decoder_layers=2, batch_first=True)
        self.loss_head = nn.Linear(dim, 10)
    def freeze_llm(self, freeze=True):
        for p in self.llm.parameters(): p.requires_grad_(not freeze)
    def forward(self, batch):
        B= batch['image'].shape[0]; T=64; D=self.dim; dev = batch['image'].device
        hidden = torch.randn(B,T,D, device=dev)
        vision_mask = torch.zeros(B,T, dtype=torch.bool, device=dev); vision_mask[:, :T//4] = True
        instr_mask  = torch.zeros(B,T, dtype=torch.bool, device=dev); instr_mask[:, T//4:T//2] = True
        inactive_vision_mask = torch.zeros(B,T, dtype=torch.bool, device=dev); inactive_vision_mask[:, :T//8] = True
        mid_scores  = torch.randn(B,T, device=dev); deep_scores = torch.randn(B,T, device=dev)
        mid_attn_scores = torch.randn(B, T//4, device=dev); mid_key_states = torch.randn(B, T//4, D, device=dev)
        logits = self.loss_head(hidden.mean(dim=1)); loss = F.cross_entropy(logits, torch.zeros(B, dtype=torch.long, device=dev))
        return {'hidden_states': hidden, 'vision_mask': vision_mask, 'instruction_mask': instr_mask,
                'inactive_vision_mask': inactive_vision_mask, 'mid_scores': mid_scores, 'deep_scores': deep_scores,
                'mid_attn_scores': mid_attn_scores, 'mid_key_states': mid_key_states, 'logits': logits, 'loss': loss}
