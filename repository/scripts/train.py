from pathlib import Path
import argparse, json, copy, time, hashlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from model import ROOT, CKPT, MODS, NAMES, InputImages, GatedFusion, load_base, Tail
OUT=ROOT/'runs/retrain'
SEED=42
def save(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def cache_maps(frame):
    identity={'checkpoint_sha256':sha(CKPT),'manifest_sha256':sha(OUT/'paired_manifest.csv'),'dtype':'float32','shape':[len(frame),4,7,7,768],'location':'last stage after downsample; before final two blocks'}
    if (OUT/'cache_complete.json').exists():
        assert json.loads((OUT/'cache_complete.json').read_text())==identity
        return np.load(OUT/'maps.npy',mmap_mode='r')
    maps=np.lib.format.open_memmap(OUT/'maps.npy',mode='w+',dtype='float32',shape=(len(frame),4,7,7,768))
    flat=maps.reshape(-1,7,7,768)
    base=load_base().cuda().eval(); base.requires_grad_(False)
    loader=DataLoader(InputImages(frame[MODS].to_numpy().ravel().tolist()),batch_size=32,num_workers=2,pin_memory=True)
    start=time.time(); offset=0
    with torch.inference_mode():
        for i,x in enumerate(loader):
            x=base.patch_embed(x.cuda(non_blocking=True))
            for layer in base.layers[:-1]: x=layer(x)
            x=base.layers[-1].downsample(x)
            flat[offset:offset+len(x)]=x.cpu().numpy(); offset+=len(x)
            if i%100==0: print('CACHE',offset,len(flat),'seconds',round(time.time()-start,1),flush=True)
    maps.flush(); assert offset==4*len(frame)
    # A cached partial forward must reproduce the existing full feature tensor.
    with torch.no_grad(): actual=base.norm(base.layers[-1].blocks(torch.tensor(np.array(flat[:4]),device='cuda'))).mean((1,2)).cpu().numpy()
    with torch.no_grad(): expected=base.forward_head(base.forward_features(torch.stack([InputImages(frame.iloc[0][MODS].tolist())[j] for j in range(4)]).cuda()),pre_logits=True).cpu().numpy()
    assert np.allclose(actual,expected,rtol=1e-4,atol=1e-4)
    save('cache_forward_audit.json',{'passed':True,'max_abs_error':float(np.max(np.abs(actual-expected)))})
    save('cache_complete.json',identity); del base; torch.cuda.empty_cache()
    return maps

def batches(indices,groups,rng,size=16):
    pending=list(rng.permutation(indices))
    while pending:
        deferred=[]; batch=[]; seen=set()
        for idx in pending:
            if groups[idx] in seen: deferred.append(idx); continue
            batch.append(idx); seen.add(groups[idx])
            if len(batch)==size:
                yield np.array(batch); batch=[]; seen=set()
        if batch: yield np.array(batch)
        pending=deferred

def forward_features(tail,maps,indices):
    outputs=[]
    with torch.inference_mode():
        for ids in np.array_split(indices,max(1,int(np.ceil(len(indices)/16)))):
            xx=torch.tensor(np.array(maps[ids]),device='cuda').flatten(0,1)
            outputs.append(tail(xx).reshape(len(ids),4,768).cpu().numpy())
    return np.concatenate(outputs)

def representation(method,maps,frame):
    assert method == 'ce', 'This release trains the proposed CE model only.'
    path=OUT/(method+'_tail.pt')
    base=load_base(); tail=Tail(base).cuda()
    if path.exists(): tail.load_state_dict(torch.load(path,weights_only=True)); tail.eval(); return tail
    torch.manual_seed(SEED); rng=np.random.default_rng(SEED)
    if method=='ce': head=copy.deepcopy(base.head.fc).cuda()
    else: raise ValueError(method)
    del base
    optimizer=torch.optim.AdamW([{'params':tail.parameters(),'lr':1e-5},{'params':head.parameters(),'lr':1e-4}],weight_decay=1e-4)
    tr=np.flatnonzero(frame.split=='train'); groups=frame.group.to_numpy(); labels=frame.label.to_numpy()
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=5)
    history=[]; start=time.time()
    for epoch in range(5):
        tail.train(); head.train(); losses=[]; visited=[]
        for j,ids in enumerate(batches(tr,groups,rng)):
            assert len(set(groups[ids]))==len(ids)
            visited.extend(ids.tolist()); xx=torch.tensor(np.array(maps[ids]),device='cuda').flatten(0,1)
            yy=torch.tensor(labels[ids],device='cuda',dtype=torch.long)
            optimizer.zero_grad(); feature=tail(xx)
            if method=='ce': loss=nn.functional.cross_entropy(head(feature),yy.repeat_interleave(4))
            else: raise ValueError(method)
            assert torch.isfinite(loss)
            loss.backward(); nn.utils.clip_grad_norm_(list(tail.parameters())+list(head.parameters()),1.); optimizer.step(); losses.append(loss.item())
            if j%150==0: print(method,'epoch',epoch+1,'batch',j,'loss',round(loss.item(),4),'seconds',round(time.time()-start,1),flush=True)
        assert sorted(visited)==sorted(tr.tolist())
        history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),'seconds':time.time()-start,'train_pairs_seen':len(visited)})
        pd.DataFrame(history).to_csv(OUT/(method+'_representation_history.csv'),index=False); scheduler.step()
    torch.save(tail.state_dict(),path); torch.save(head.state_dict(),OUT/(method+'_pretraining_head.pt'))
    tail.eval(); return tail

def fit_classifier(features,frame,method,kind):
    torch.manual_seed(SEED)
    x=torch.tensor(features,device='cuda'); labels=frame.label.to_numpy(); y=torch.tensor(labels,device='cuda',dtype=torch.long)
    tr=torch.tensor(np.flatnonzero(frame.split=='train'),device='cuda'); va=torch.tensor(np.flatnonzero(frame.split=='val'),device='cuda')
    model=(nn.Linear(768,4) if kind=='linear' else GatedFusion()).cuda()
    if kind=='gated':
        for p in model.infection.parameters(): p.requires_grad_(False)
    opt=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=3e-4,weight_decay=1e-4)
    history=[]; best=(-1,float('-inf')); patience=0; best_state=None
    for epoch in range(80):
        model.train()
        for ids in tr[torch.randperm(len(tr),device='cuda')].split(128):
            opt.zero_grad()
            if kind=='linear': loss=nn.functional.cross_entropy(model(x[ids].flatten(0,1)),y[ids].repeat_interleave(4))
            else: loss=nn.functional.cross_entropy(model(x[ids])[0],y[ids])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            if kind=='linear': p=torch.softmax(model(x[va]),dim=-1).mean(1)
            else: p=torch.softmax(model(x[va])[0],dim=-1)
            acc=(p.argmax(1)==y[va]).float().mean().item(); vl=-torch.log(p[torch.arange(len(va),device='cuda'),y[va]].clamp_min(1e-12)).mean().item()
        history.append({'epoch':epoch+1,'val_accuracy':acc,'val_loss':vl})
        if (acc,-vl)>best: best=(acc,-vl); best_state=copy.deepcopy(model.state_dict()); patience=0; best_epoch=epoch+1
        else: patience+=1
        if patience>=15: break
    name=method+'_'+kind; torch.save(best_state,OUT/(name+'.pt')); pd.DataFrame(history).to_csv(OUT/(name+'_history.csv'),index=False)
    print('CLASSIFIER',name,'val',best[0],'epoch',best_epoch,flush=True)
    return {'name':name,'method':method,'kind':kind,'val_accuracy':best[0],'val_loss':-best[1],'epoch':best_epoch}

def main():
    global OUT
    ap=argparse.ArgumentParser(description='Retrain proposed model and its two component baselines on CUDA.')
    ap.add_argument('--data-root',type=Path,required=True,help='Directory containing RGB, VNIR, VNIR_800nm and VNIR_1000nm')
    ap.add_argument('--output',type=Path,default=OUT)
    args=ap.parse_args(); OUT=args.output
    if any(OUT.glob('*.pt')): raise SystemExit('Use a new output directory; existing checkpoints are not overwritten.')
    OUT.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available(): raise SystemExit('CUDA is required for this training implementation.')
    torch.set_num_threads(4); torch.manual_seed(SEED)
    frame=pd.read_csv(ROOT/'data/paired_manifest.csv')
    assert frame.groupby('group').split.nunique().max()==1
    for m in MODS: frame[m]=frame[m].map(lambda p:str((args.data_root/p).resolve()))
    missing=[p for p in frame[MODS].to_numpy().ravel() if not Path(p).is_file()]
    if missing: raise SystemExit(f'{len(missing)} images missing. First: {missing[0]}')
    frame.to_csv(OUT/'paired_manifest.csv',index=False)
    maps=cache_maps(frame)
    tv=np.flatnonzero(frame.split!='test'); te=np.flatnonzero(frame.split=='test')
    small=frame.iloc[tv].reset_index(drop=True)
    base=load_base(); initial=Tail(base).cuda().eval(); del base
    frozen=forward_features(initial,maps,tv); del initial
    candidates=[fit_classifier(frozen,small,'frozen','gated')]
    tail=representation('ce',maps,frame)
    feat=forward_features(tail,maps,tv)
    for kind in ['linear','gated']: candidates.append(fit_classifier(feat,small,'ce',kind))
    save('selection_before_test.json',{'selected':max(candidates,key=lambda c:(c['val_accuracy'],-c['val_loss'])),'candidates':candidates})
    feature=forward_features(tail,maps,te)
    from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
    scores={}
    for c in candidates:
        if c['method']=='frozen':
            base=load_base(); initial=Tail(base).cuda().eval(); del base
            x=forward_features(initial,maps,te); del initial
        else: x=feature
        head=(nn.Linear(768,4) if c['kind']=='linear' else GatedFusion()).cuda()
        head.load_state_dict(torch.load(OUT/(c['name']+'.pt'),weights_only=True));head.eval()
        with torch.inference_mode():
            z=torch.tensor(x,device='cuda')
            p=(head(z).softmax(-1).mean(1) if c['kind']=='linear' else head(z)[0].softmax(-1)).cpu().numpy()
        np.save(OUT/(c['name']+'_test_probabilities.npy'),p)
        y=frame.label.to_numpy()[te];pred=p.argmax(1)
        scores[c['name']]={'accuracy':accuracy_score(y,pred),'macro_f1':f1_score(y,pred,average='macro'),'confusion_matrix':confusion_matrix(y,pred).tolist()}
    save('metrics.json',scores)
if __name__=='__main__':main()
