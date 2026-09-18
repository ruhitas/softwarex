from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/figures';OUT.mkdir(parents=True,exist_ok=True)
fig,ax=plt.subplots(figsize=(11,8)); ax.set(xlim=(0,11),ylim=(0,8)); ax.axis('off')
def box(x,y,w,h,text,color='#e9f0f5'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04',facecolor=color,edgecolor='#4b5964',linewidth=1))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
def arrow(a,b,dashed=False):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,linewidth=1.1,color='#4b5964',linestyle='--' if dashed else '-'))
box(2.9,7,5.2,.7,'Local image archive: 40,716 files\nRGB, VNIR, 800 nm and 1000 nm')
box(2.9,5.8,5.2,.8,'Matching by plant + day + viewing angle\n10,178 complete quartets / 40,712 images\n4 unmatched training images excluded')
arrow((5.5,7),(5.5,6.6))
box(.3,4.3,3.2,.85,'TRAINING\n244 plants / 7,130 quartets\n28,520 images')
box(3.9,4.3,3.2,.85,'VALIDATION\n53 plants / 1,559 quartets\n6,236 images')
box(7.5,4.3,3.2,.85,'TEST\n50 plants / 1,489 quartets\n5,956 images')
for x in [1.9,5.5,9.1]: arrow((5.5,5.8),(x,5.15))
box(.3,2.65,3.2,1.05,'Preprocessing and Swin adaptation\nResize to 255 / crop to 224\nLast two blocks, CE, 5 epochs\nThen train the gated fusion head','#dcece7')
arrow((1.9,4.3),(1.9,3.7))
box(4.0,2.65,3.1,1.05,'Checkpoint selection\nValidation accuracy\nTie-breaker: validation loss','#dcece7')
arrow((3.5,3.15),(4,3.15)); arrow((5.5,4.3),(5.5,3.7))
box(7.5,2.65,3.2,1.05,'Test inference with selected model\nOne prediction per quartet\nFour class probabilities','#dcece7')
arrow((7.1,3.15),(7.5,3.15)); arrow((9.1,4.3),(9.1,3.7))
box(2.5,.8,6,1.0,'Performance evaluation\nClass metrics / confusion matrix / ROC and PR\n5,000 plant-level bootstrap replicates / joint Grad-CAM')
arrow((9.1,2.65),(5.5,1.8))
ax.text(5.5,.25,'All days and viewing angles of a plant remain in the same data split.',ha='center',fontsize=10)
for ext in ['png','svg']: fig.savefig(OUT/f'figure3_experimental_workflow_en.{ext}',dpi=300,bbox_inches='tight')
plt.close(fig)
