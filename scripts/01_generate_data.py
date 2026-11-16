import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from dynadistill.utils.common import ROOT,config,seed_all
from dynadistill.data.systems import initial,simulate
c=config();seed_all(c['seed']);rng=np.random.default_rng(c['seed']);s=c['system'];d=c['simulation']
t=np.linspace(0,d['t_end'],d['points']); clean=np.stack([simulate(s,t,initial(s,rng)) for _ in range(d['trajectories'])]); noisy=clean+rng.normal(0,d['noise_std'],clean.shape)
ids=rng.permutation(len(clean));ntrain=int(.7*len(ids));nval=int(.15*len(ids));splits={'train':ids[:ntrain],'val':ids[ntrain:ntrain+nval],'test':ids[ntrain+nval:]}
out=ROOT/'data/processed';out.mkdir(parents=True,exist_ok=True)
for split,ix in splits.items(): np.savez_compressed(out/f'{s}_{split}.npz',t=t,x=noisy[ix],clean=clean[ix])
print('Generated',s,clean.shape,'splits:',{k:len(v) for k,v in splits.items()})
