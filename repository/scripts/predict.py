"""Inference and optional true joint-score Grad-CAM for a matched quartet."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from torch import nn
from model import ROOT,MODS,NAMES,InputImages,load_base,Tail,GatedFusion

class Joint(nn.Module):
    def __init__(self,weights):
        super().__init__();self.base=load_base(weights/'base_swin.pt');self.tail=Tail(self.base)
        self.tail.load_state_dict(torch.load(weights/'ce_tail.pt',map_location='cpu',weights_only=True))
        self.fusion=GatedFusion();self.fusion.load_state_dict(torch.load(weights/'ce_gated.pt',map_location='cpu',weights_only=True))
        self.base.requires_grad_(False)
    def forward(self,x):
        z=self.base.patch_embed(x)
        for layer in self.base.layers[:-1]:z=layer(z)
        z=self.base.layers[-1].downsample(z)
        logits,_,gates=self.fusion(self.tail(z)[None])
        return logits,gates
def main():
    ap=argparse.ArgumentParser()
    for key in ['rgb','vnir','nm800','nm1000']:ap.add_argument('--'+key,required=True)
    ap.add_argument('--weights',type=Path,default=ROOT/'weights')
    ap.add_argument('--gradcam',type=Path,help='Optional output directory for heatmaps and overlays')
    a=ap.parse_args();torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=Joint(a.weights).to(device).eval();paths=[a.rgb,a.vnir,a.nm800,a.nm1000]
    ds=InputImages(paths);x=torch.stack([ds[i] for i in range(4)]).to(device).requires_grad_(bool(a.gradcam))
    holder={}
    if a.gradcam:
        def hook(module,args,out):holder['activation']=out;out.retain_grad()
        handle=model.tail.blocks[-1].norm1.register_forward_hook(hook)
    with torch.set_grad_enabled(bool(a.gradcam)):logits,gates=model(x)
    p=logits.softmax(-1)[0];target=int(p.argmax())
    result={'class':NAMES[target],'probabilities':p.detach().cpu().tolist(),'modality_order':MODS,'gate_weights':gates.detach().cpu().tolist()}
    if a.gradcam:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        a.gradcam.mkdir(parents=True,exist_ok=True)
        logits[0,target].backward();activation=holder['activation'];grad=activation.grad
        assert grad is not None and torch.isfinite(grad).all()
        raw=torch.relu((grad.mean((1,2),keepdim=True)*activation).sum(-1)).detach().cpu().numpy()
        maps=[];fig,axes=plt.subplots(1,4,figsize=(10,3))
        for j in range(4):
            cam=raw[j];cam=(cam-cam.min())/max(float(cam.max()-cam.min()),1e-12)
            up=torch.nn.functional.interpolate(torch.tensor(cam)[None,None],size=(224,224),mode='bilinear',align_corners=False)[0,0].numpy();maps.append(up)
            rgb=x[j].detach().cpu().permute(1,2,0).numpy()*np.array([.229,.224,.225])+np.array([.485,.456,.406])
            alpha=.5*up[...,None];overlay=(1-alpha)*rgb+alpha*plt.colormaps['inferno'](up)[...,:3]
            axes[j].imshow(np.clip(overlay,0,1));axes[j].set_title(MODS[j]);axes[j].axis('off')
        fig.tight_layout();fig.savefig(a.gradcam/'gradcam.png',dpi=300);plt.close(fig)
        np.savez_compressed(a.gradcam/'gradcam.npz',raw=raw,normalized=maps)
        result['raw_cam_max']=raw.reshape(4,-1).max(1).tolist();handle.remove()
        (a.gradcam/'prediction.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
