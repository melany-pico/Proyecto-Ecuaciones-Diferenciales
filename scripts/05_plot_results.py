import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt,numpy as np
from dynadistill.utils.common import ROOT,config
c=config();s=c['system'];d=np.load(ROOT/f'data/processed/{s}_test.npz');fig,ax=plt.subplots(figsize=(8,4));ax.plot(d['t'],d['clean'][0,:,0],label='True state x0');ax.scatter(d['t'],d['x'][0,:,0],s=5,alpha=.3,label='Noisy observations');ax.set(xlabel='Time',ylabel='State',title=f'{s} held-out trajectory');ax.legend();fig.tight_layout();out=ROOT/'results/figures';out.mkdir(parents=True,exist_ok=True);fig.savefig(out/f'{s}_trajectory.png',dpi=200);print('Saved figure')
