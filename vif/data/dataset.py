import json, os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
class MMInstructionDataset(Dataset):
    def __init__(self, jsonl_path, image_root, image_size=336):
        self.samples = json.load(open(jsonl_path, 'r')); self.image_root=image_root
        self.tfm = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        e = self.samples[idx]; path = os.path.join(self.image_root, e['image']); img = Image.open(path).convert('RGB')
        img = self.tfm(img); return {'image': img, 'instruction': e['instruction'], 'answer': e['answer']}
