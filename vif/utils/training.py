import math, torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
def build_optimizers(model, lr, mm_lr, weight_decay):
    llm_params, mm_params = [], []
    for n,p in model.named_parameters():
        if not p.requires_grad: continue
        if any(k in n for k in ['vif.', 'projector', 'vision']): mm_params.append(p)
        else: llm_params.append(p)
    return AdamW([{'params': llm_params, 'lr': lr, 'weight_decay': weight_decay},
                  {'params': mm_params, 'lr': mm_lr, 'weight_decay': weight_decay}])
def train_epochs(model, dataloader, val_loader, optimizer, epochs, device='cuda', clip=1.0, precision='bf16', warmup_ratio=0.05):
    steps = len(dataloader)*epochs; warmup_steps = int(steps*warmup_ratio); global_step=0; model.train()
    for ep in range(epochs):
        pbar = tqdm(dataloader, desc=f'epoch {ep+1}/{epochs}')
        for batch in pbar:
            global_step += 1
            for k in list(batch.keys()):
                if hasattr(batch[k], 'to'): batch[k]=batch[k].to(device)
            out = model(batch); loss = out['loss']
            for g in optimizer.param_groups:
                base_lr = g.get('initial_lr', g['lr'])
                if global_step < warmup_steps: g['lr'] = base_lr * (global_step/max(1,warmup_steps))
                else:
                    pct = (global_step-warmup_steps)/max(1, steps-warmup_steps); g['lr'] = 0.5*base_lr*(1+math.cos(math.pi*pct))
            optimizer.zero_grad(set_to_none=True); loss.backward(); clip_grad_norm_(model.parameters(), clip); optimizer.step()
            pbar.set_postfix({'loss': float(loss.detach().cpu())})
def make_dataloader(dataset, batch_size=16, shuffle=True, num_workers=2):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
