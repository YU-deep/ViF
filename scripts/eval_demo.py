import argparse, yaml, json, torch
from PIL import Image
from torchvision import transforms
from vif.models.base_stub import BaseVLMStub
from vif.models.wrapper import ViFWrapper
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--images_dir', required=True); ap.add_argument('--questions_file', required=True); args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config)); device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base = BaseVLMStub(dim=1024)
    model = ViFWrapper(base, dim=1024, relay_ratio=cfg.get('relay_ratio',0.02), omega=cfg.get('omega',0.3),
                       tau=cfg.get('tau',0.8), alpha_mid=cfg.get('alpha_mid',0.2), alpha_deep=cfg.get('alpha_deep',0.1),
                       use_keynorm=cfg.get('use_keynorm',True)).to(device)
    model.eval()
    tfm = transforms.Compose([transforms.Resize((336,336)), transforms.ToTensor()])
    qa = json.load(open(args.questions_file, 'r'))
    for ex in qa:
        path = f"{ex['image']}"
        img = Image.open(path).convert('RGB'); batch = {'image': tfm(img).unsqueeze(0).to(device), 'instruction':[ex['instruction']], 'answer':['N/A']}
        with torch.no_grad(): out = model(batch)
        print(path, ex['instruction'], '-> (demo logits head)')
if __name__=='__main__': main()
