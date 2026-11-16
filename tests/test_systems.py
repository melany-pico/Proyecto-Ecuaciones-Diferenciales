import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from dynadistill.data.systems import field,simulate
from dynadistill.discovery.sparse import stlsq
def test_field():
 assert np.allclose(field('lorenz',np.array([1.,2.,3.])),[10.,23.,-6.])
def test_simulation():
 assert simulate('vanderpol',np.linspace(0,1,10),[1.,0.]).shape==(10,2)
def test_sparse():
 x=np.linspace(-2,2,100)[:,None];y=2*x
 lib,c=stlsq(x,y,degree=1,threshold=.01)
 assert abs(c[1,0]-2)<.01
