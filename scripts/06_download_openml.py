import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sklearn.datasets import fetch_openml
from dynadistill.utils.common import ROOT
# Optional demonstration dataset: not a substitute for continuous-time trajectories.
out=ROOT/'data/external/openml';out.mkdir(parents=True,exist_ok=True)
try:
    dataset=fetch_openml(data_id=61,as_frame=True,parser='auto');dataset.frame.to_csv(out/'iris_openml_61.csv',index=False);print('OpenML sample downloaded (tabular connectivity test only).')
except Exception as e:print('OpenML unavailable:',e,'Use https://www.openml.org/d/61 and place CSV in data/external/openml/')
