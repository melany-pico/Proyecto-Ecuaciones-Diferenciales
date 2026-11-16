import subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
for name in ['01_generate_data.py','02_train_neural_ode.py','03_distill_equations.py','04_evaluate.py','05_plot_results.py']:
 print('\n===',name,'===',flush=True);subprocess.run([sys.executable,str(root/'scripts'/name)],cwd=root,check=True)
