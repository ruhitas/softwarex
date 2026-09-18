"""Reproduce manuscript metrics and cluster bootstrap without images or weights."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,roc_auc_score,average_precision_score

ROOT=Path(__file__).resolve().parents[1]
def score(cm):
    d=cm.sum(0)+cm.sum(1)
    return np.array([np.trace(cm)/cm.sum(),np.divide(2*np.diag(cm),d,out=np.zeros(4),where=d>0).mean()])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'runs/evaluation');a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(ROOT/'data/paired_manifest.csv')
    assert frame.groupby('group').split.nunique().max()==1
    assert frame.groupby('split').size().to_dict()=={'test':1489,'train':7130,'val':1559}
    test=frame[frame.split=='test'].reset_index(drop=True);y=test.label.to_numpy()
    report={};cms={};names=['frozen_gated','ce_linear','ce_gated']
    saved=json.loads((ROOT/'results/reported_metrics.json').read_text())
    for name in names:
        p=np.load(ROOT/'results'/f'{name}_test_probabilities.npy')
        assert p.shape==(len(test),4) and np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6)
        pred=p.argmax(1);cm=confusion_matrix(y,pred,labels=range(4));m=score(cm)
        assert abs(m[0]-saved[name]['accuracy'])<1e-12 and abs(m[1]-saved[name]['macro_f1'])<1e-12
        cms[name]=np.stack([confusion_matrix(y[ids],pred[ids],labels=range(4)) for ids in test.groupby('group',sort=True).indices.values()])
        report[name]={'accuracy':m[0],'macro_f1':m[1],'correct':int(np.trace(cm)),'test_quartets':len(test),'test_plants':test.group.nunique(),'confusion_matrix':cm.tolist(),'roc_auc_per_class':[roc_auc_score(y==k,p[:,k]) for k in range(4)],'average_precision_per_class':[average_precision_score(y==k,p[:,k]) for k in range(4)]}
    rng=np.random.default_rng(20260909);samples={n:[] for n in names};diff=[]
    for _ in range(5000):
        ids=rng.integers(0,50,50);m={n:score(cms[n][ids].sum(0)) for n in names}
        for n in names:samples[n].append(m[n])
        diff.append(m['ce_gated']-m['frozen_gated'])
    for n in names:report[n]['plant_bootstrap_95ci']=np.quantile(samples[n],[.025,.975],axis=0).tolist()
    report['ce_minus_frozen_95ci']=np.quantile(diff,[.025,.975],axis=0).tolist()
    report['bootstrap']={'seed':20260909,'replicates':5000,'ci_columns':['accuracy','macro_f1']}
    (a.output/'metrics.json').write_text(json.dumps(report,indent=2))
    counts=frame.groupby(['split','label']).size().rename('quartets').reset_index()
    for col in ['RGB','VNIR','VNIR_800nm','VNIR_1000nm']:counts[col]=counts.quartets
    counts['total_images']=4*counts.quartets;counts.to_csv(a.output/'split_class_modality_counts.csv',index=False)
    print(json.dumps(report['ce_gated'],indent=2))
if __name__=='__main__':main()
