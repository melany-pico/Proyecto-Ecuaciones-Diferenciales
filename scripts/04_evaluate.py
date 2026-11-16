import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,torch
from scipy.integrate import solve_ivp
from sklearn.preprocessing import PolynomialFeatures
from dynadistill.utils.common import ROOT,config
from dynadistill.models.neuralode import VectorField
from dynadistill.data.systems import field
c=config();s=c['system'];test=np.load(ROOT/f'data/processed/{s}_test.npz');ck=torch.load(ROOT/f'results/checkpoints/{s}.pt',map_location='cpu',weights_only=True);m=VectorField(ck['dim'],ck['hidden']);m.load_state_dict(ck['state_dict']);m.eval();eq=np.load(ROOT/f'results/equations/{s}.npz');coef=eq['coef'];poly=PolynomialFeatures(degree=c['discovery']['polynomial_degree'],include_bias=True);poly.fit(test['x'].reshape(-1,ck['dim']));t=test['t'];neural=[];symbolic=[];truth=[]
for x in test['clean']:
    with torch.no_grad():pred=m.trajectory(torch.tensor(x[0:1],dtype=torch.float32),torch.tensor(t,dtype=torch.float32))[0].numpy()
    sol=solve_ivp(lambda _,z:poly.transform(np.asarray(z)[None,:])[0]@coef,(t[0],t[-1]),x[0],t_eval=t,rtol=1e-6,atol=1e-8)
    neural.append(float(np.mean((pred-x)**2)))
    symbolic.append(float(np.mean((sol.y.T-x)**2)) if sol.success and len(sol.t)==len(t) else None)
    truth.append(float(np.mean((poly.transform(x)@coef-field(s,x))**2)))
report={'system':s,'neural_trajectory_mse':float(np.mean(neural)),'symbolic_trajectory_mse':float(np.mean([v for v in symbolic if v is not None])) if any(v is not None for v in symbolic) else None,'symbolic_vectorfield_mse':float(np.mean(truth)),'failed_symbolic_integrations':sum(v is None for v in symbolic),'test_trajectories':len(neural)}
out=ROOT/'results/metrics';out.mkdir(parents=True,exist_ok=True);(out/f'{s}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
