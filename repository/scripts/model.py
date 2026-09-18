from pathlib import Path
import os, copy
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import timm
ROOT=Path(__file__).resolve().parents[1]
CKPT=Path(os.environ.get('TOBRFV_BASE_CHECKPOINT',str(ROOT/'weights/base_swin.pt')))
MODS=['RGB','VNIR','VNIR_800nm','VNIR_1000nm']
NAMES=['Tomato_Healthy','Tomato_Virus','Pepper_Healthy','Pepper_Virus']
class InputImages(Dataset):
    def __init__(self,paths):
        self.paths=paths
        self.transform=transforms.Compose([transforms.Resize(int(224*1.14)),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    def __len__(self): return len(self.paths)
    def __getitem__(self,i):
        with Image.open(self.paths[i]) as im: return self.transform(im.convert('RGB'))

class GatedFusion(nn.Module):
    def __init__(self):
        super().__init__()
        def adapter(): return nn.Sequential(nn.LayerNorm(768),nn.Linear(768,256),nn.GELU(),nn.Dropout(.2))
        self.rgb=adapter(); self.vnir=adapter()
        self.modality=nn.Parameter(torch.zeros(4,256))
        self.gate=nn.Sequential(nn.Linear(256,64),nn.Tanh(),nn.Linear(64,1))
        self.classifier=nn.Sequential(nn.LayerNorm(256),nn.Dropout(.2),nn.Linear(256,4))
        self.infection=nn.Linear(256,2)
    def forward(self,x):
        h=torch.cat([self.rgb(x[:,:1]),self.vnir(x[:,1:])],dim=1)+self.modality
        weights=torch.softmax(self.gate(h),dim=1)
        fused=(weights*h).sum(1)
        return self.classifier(fused),self.infection(fused),weights.squeeze(-1)

def load_base(checkpoint=None):
    m=timm.create_model('swin_tiny_patch4_window7_224',pretrained=False,num_classes=4)
    cp=torch.load(CKPT if checkpoint is None else checkpoint,map_location='cpu',weights_only=False)
    assert cp['class_names']==NAMES; m.load_state_dict(cp['model']); return m

class Tail(nn.Module):
    def __init__(self,base):
        super().__init__(); self.blocks=copy.deepcopy(base.layers[-1].blocks); self.norm=copy.deepcopy(base.norm)
    def forward(self,x): return self.norm(self.blocks(x)).mean((1,2))
