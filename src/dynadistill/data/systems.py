import numpy as np
from scipy.integrate import solve_ivp

def field(name, x):
    x=np.asarray(x)
    if name=='lorenz':
        a,b,c=np.moveaxis(x,-1,0)
        return np.stack([10*(b-a), a*(28-c)-b, a*b-(8/3)*c],axis=-1)
    if name=='vanderpol':
        a,b=np.moveaxis(x,-1,0)
        return np.stack([b, (1-a*a)*b-a],axis=-1)
    if name=='duffing':
        a,b=np.moveaxis(x,-1,0)
        return np.stack([b, -0.2*b+a-a**3],axis=-1)
    if name=='lotka_volterra':
        a,b=np.moveaxis(x,-1,0)
        return np.stack([1.5*a-a*b, a*b-3*b],axis=-1)
    raise ValueError(f'Unknown system: {name}')

def initial(name, rng):
    if name=='lorenz': return rng.uniform([-12,-12,10],[12,12,30])
    if name=='lotka_volterra': return rng.uniform([1,1],[5,5])
    return rng.uniform([-2,-2],[2,2])

def simulate(name,t,x0):
    sol=solve_ivp(lambda _,x:field(name,x), (t[0],t[-1]),x0,t_eval=t,rtol=1e-9,atol=1e-11)
    if not sol.success: raise RuntimeError(sol.message)
    return sol.y.T
