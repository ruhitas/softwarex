from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,roc_curve,precision_recall_curve,roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/figures';OUT.mkdir(exist_ok=True,parents=True)
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7']
NAMES=['Tomato healthy','Tomato infected','Pepper healthy','Pepper infected']
plt.rcParams.update({'font.size':8,'axes.titlesize':9,'legend.fontsize':7,'svg.fonttype':'none'})
frame=pd.read_csv(ROOT/'data/paired_manifest.csv');test=frame[frame.split=='test']
y=test.label.to_numpy();probs=np.load(ROOT/'results/ce_gated_test_probabilities.npy');pred=probs.argmax(1)
def export(fig,name):
    for ext in ['png','svg']:fig.savefig(OUT/(name+'.'+ext),dpi=300,bbox_inches='tight')
    plt.close(fig)
# Figure 3: confusion and ROC/PR. Each curve is computed from saved probabilities.
fig=plt.figure(figsize=(7.1,5.1)); gs=fig.add_gridspec(2,2,width_ratios=[1.15,1]); ax=fig.add_subplot(gs[:,0]); cm=confusion_matrix(y,pred)
ax.imshow(cm,cmap='Blues'); short=['T-H','T-I','P-H','P-I']; ax.set_xticks(range(4),short); ax.set_yticks(range(4),short); ax.set_xlabel('Predicted class'); ax.set_ylabel('True class'); ax.set_title('(a) Confusion matrix')
for i in range(4):
    for j in range(4): ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
roc=fig.add_subplot(gs[0,1]); pr=fig.add_subplot(gs[1,1]); aucs={}
for i in range(4):
    yy=(y==i).astype(int); fpr,tpr,_=roc_curve(yy,probs[:,i]); precision,recall,_=precision_recall_curve(yy,probs[:,i]); auc=roc_auc_score(yy,probs[:,i]); ap=average_precision_score(yy,probs[:,i]); aucs[NAMES[i]]={'auc':auc,'average_precision':ap}
    roc.plot(fpr,tpr,color=COLORS[i],label=f'{short[i]} {auc:.3f}'); pr.plot(recall,precision,color=COLORS[i],label=f'{short[i]} {ap:.3f}')
roc.plot([0,1],[0,1],':',color='gray'); roc.set(xlabel='False positive rate',ylabel='Recall',title='(b) One-vs-rest ROC'); roc.legend(loc='lower right',fontsize=6)
pr.set(xlabel='Recall',ylabel='Precision',title='(c) Precision–recall'); pr.legend(loc='lower left',fontsize=6); fig.tight_layout(); export(fig,'figure4_confusion_roc_pr_en')
