import argparse, yaml, torch
from vif.data.dataset import MMInstructionDataset
from vif.utils.training import make_dataloader, build_optimizers, train_epochs
from vif.models.wrapper import ViFWrapper
from vif.models.base_stub import BaseVLMStub
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config)); device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_ds = MMInstructionDataset(cfg['train_jsonl'], cfg['image_root']); val_ds = MMInstructionDataset(cfg['val_jsonl'], cfg['image_root'])
    train_dl = make_dataloader(train_ds, batch_size=cfg['batch_size']); val_dl = make_dataloader(val_ds, batch_size=cfg['batch_size'], shuffle=False)
    base = BaseVLMStub(dim=1024); base.freeze_llm(False)
    model = ViFWrapper(base, dim=1024, relay_ratio=cfg['relay_ratio'], omega=cfg['omega'], tau=cfg['tau'],
                       alpha_mid=cfg['alpha_mid'], alpha_deep=cfg['alpha_deep'], use_keynorm=cfg['use_keynorm']).to(device)
    optim = build_optimizers(model, lr=cfg['lr'], mm_lr=cfg['mm_lr'], weight_decay=cfg['weight_decay'])
    for g in optim.param_groups: g['initial_lr'] = g['lr']
    train_epochs(model, train_dl, val_dl, optim, epochs=cfg['epochs'], device=device, clip=cfg['clip_grad_norm'], precision=cfg['precision'], warmup_ratio=cfg['warmup_ratio'])
if __name__ == '__main__': main()
