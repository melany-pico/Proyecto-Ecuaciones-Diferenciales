"""Evaluate physical vector fields for v0.5 and baseline checkpoints.
Ground-truth derivatives are used ONLY for evaluation, not training.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import csv
import numpy as np
import torch
from dynadistill.utils.common import ROOT, config
from dynadistill.models.neuralode import VectorField
from dynadistill.data.systems import field


def main():
    c = config(); s = c['system']
    path = ROOT / f'results/checkpoints/{s}_v05.pt'
    if not path.exists(): raise FileNotFoundError(f'First train v0.5: {path}')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = VectorField(checkpoint['dim'], checkpoint['hidden'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    rows = []
    for split in ['train', 'val', 'test']:
        with np.load(ROOT / f'data/processed/{s}_{split}.npz') as data:
            x = data['x'].reshape(-1, checkpoint['dim']).astype(np.float32)
        truth = field(s, x).astype(np.float64)
        with torch.no_grad():
            pred = model.physical_field(torch.from_numpy(x)).numpy().astype(np.float64)
        residual = pred - truth
        mse = float(np.mean(residual**2)); rmse = float(np.sqrt(mse))
        rel = float(np.linalg.norm(residual) / max(np.linalg.norm(truth), 1e-12))
        mean_baseline = np.mean(truth, axis=0, keepdims=True)
        denom = np.sum((truth - mean_baseline)**2)
        skill = float(1 - np.sum(residual**2) / denom) if denom > 0 else float('nan')
        rows.append([split, mse, rmse, rel, skill])
        print(f'{split:5s} | MSE={mse:.6f} RMSE={rmse:.6f} rel_L2={rel:.6f} skill={skill:.6f}')
        if split == 'test':
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, checkpoint['dim'], figsize=(4*checkpoint['dim'], 4))
                axes = np.atleast_1d(axes)
                for j, ax in enumerate(axes):
                    ax.scatter(truth[:, j], pred[:, j], s=3, alpha=.3)
                    lo = min(truth[:, j].min(), pred[:, j].min())
                    hi = max(truth[:, j].max(), pred[:, j].max())
                    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1)
                    ax.set(xlabel='True dx/dt', ylabel='Neural dx/dt', title=f'Component {j+1}')
                fig.tight_layout()
                dest = ROOT / 'results/figures'; dest.mkdir(parents=True, exist_ok=True)
                fig.savefig(dest / f'{s}_vector_field_v05_test.png', dpi=180)
                plt.close(fig)
                print(f'Figure: {dest / (s + "_vector_field_v05_test.png")}')
            except ImportError:
                print('matplotlib unavailable: skipping figure')
    out = ROOT / 'results/metrics'; out.mkdir(parents=True, exist_ok=True)
    target = out / f'{s}_vector_field_v05.csv'
    with target.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f); writer.writerow(['split', 'mse', 'rmse', 'relative_l2', 'skill_vs_mean']); writer.writerows(rows)
    print(f'Metrics: {target}')

if __name__ == '__main__': main()
