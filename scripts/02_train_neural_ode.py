"""v0.5: train a normalized Neural ODE with optional derivative supervision.

Derivative targets are finite differences of noisy observations, never ground truth.
Validation uses fixed, non-overlapping trajectory identities and deterministic windows.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import csv
import copy
import numpy as np
import torch
from dynadistill.utils.common import ROOT, config, seed_all
from dynadistill.models.neuralode import VectorField


def main():
    cfg = config(); seed_all(cfg['seed'])
    system = cfg['system']; d = cfg['training']
    device = torch.device(d.get('device', 'cpu'))
    train = np.load(ROOT / f'data/processed/{system}_train.npz')
    val = np.load(ROOT / f'data/processed/{system}_val.npz')
    xtr = torch.as_tensor(train['x'], dtype=torch.float32, device=device)
    xval = torch.as_tensor(val['x'], dtype=torch.float32, device=device)
    t = torch.as_tensor(train['t'], dtype=torch.float32, device=device)
    tv = torch.as_tensor(val['t'], dtype=torch.float32, device=device)
    if xtr.ndim != 3 or xval.ndim != 3 or len(t) < 3:
        raise ValueError('Expected trajectories [N,T,D] with at least 3 time points')
    if not torch.all(torch.diff(t) > 0) or not torch.all(torch.diff(tv) > 0):
        raise ValueError('Time arrays must be strictly increasing')
    if len(t) != xtr.shape[1] or len(tv) != xval.shape[1]:
        raise ValueError('Time and trajectory lengths disagree')
    if not torch.allclose(t, tv):
        raise ValueError('Train and validation time grids must match')

    enabled = d.get('normalization', {}).get('enabled', True)
    eps = float(d.get('normalization', {}).get('epsilon', 1e-6))
    mean = xtr.mean(dim=(0, 1)) if enabled else torch.zeros(xtr.shape[-1], device=device)
    std = xtr.std(dim=(0, 1), unbiased=False).clamp_min(eps) if enabled else torch.ones(xtr.shape[-1], device=device)
    ztr = (xtr - mean) / std; zval = (xval - mean) / std
    model = VectorField(xtr.shape[-1], d['hidden'], mean=mean, std=std).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(d['lr']))
    sch = d.get('scheduler', {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=float(sch.get('factor', .5)), patience=int(sch.get('patience', 20)))
    loss_cfg = d.get('loss', {})
    wt = float(loss_cfg.get('trajectory_weight', 1.0))
    wd = float(loss_cfg.get('derivative_weight', 0.1))
    if wt < 0 or wd < 0 or wt + wd <= 0:
        raise ValueError('Loss weights must be nonnegative and not both zero')
    rng = np.random.default_rng(cfg['seed'])
    L = min(int(d['segment_length']), len(t))
    if L < 3: raise ValueError('segment_length must be >=3')
    bs = min(int(d['batch_trajectories']), len(ztr))
    n_batches = int(d.get('batches_per_epoch', 8))
    if bs < 1 or n_batches < 1: raise ValueError('Batch settings must be positive')

    def finite_difference(y, times):
        # Nonuniform-grid central differences (interior only).
        left = times[1:-1] - times[:-2]
        right = times[2:] - times[1:-1]
        a = -right / (left * (left + right))
        b = (right - left) / (left * right)
        c = left / (right * (left + right))
        return a[None, :, None] * y[:, :-2] + b[None, :, None] * y[:, 1:-1] + c[None, :, None] * y[:, 2:]

    def losses(batch, times):
        pred = model.trajectory(batch[:, 0], times - times[0])
        trajectory_loss = (pred - batch).square().mean()
        deriv_est = finite_difference(batch, times)
        deriv_pred = model(times[1:-1], batch[:, 1:-1])
        derivative_loss = (deriv_pred - deriv_est).square().mean()
        return trajectory_loss, derivative_loss

    # Fixed windows ensure validation is comparable between epochs.
    starts = sorted(set([0, (len(tv)-L)//2, len(tv)-L]))
    best = float('inf'); best_state = None; patience = 0
    metrics = ROOT / 'results/metrics'; metrics.mkdir(parents=True, exist_ok=True)
    ckpts = ROOT / 'results/checkpoints'; ckpts.mkdir(parents=True, exist_ok=True)
    logpath = metrics / f'{system}_training_v05.csv'
    with logpath.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_total', 'train_traj', 'train_derivative', 'val_total', 'val_traj', 'val_derivative', 'lr'])
        for epoch in range(1, int(d['epochs']) + 1):
            model.train(); totals = np.zeros(3)
            for _ in range(n_batches):
                ids = rng.choice(len(ztr), size=bs, replace=False)
                start = int(rng.integers(0, len(t)-L+1))
                batch = ztr[ids, start:start+L]
                lt, ld = losses(batch, t[start:start+L])
                loss = wt * lt + wd * ld
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                totals += [loss.item(), lt.item(), ld.item()]
            totals /= n_batches
            model.eval()
            with torch.no_grad():
                v = np.zeros(3)
                for start in starts:
                    lt, ld = losses(zval[:, start:start+L], tv[start:start+L])
                    v += [wt*lt.item()+wd*ld.item(), lt.item(), ld.item()]
                v /= len(starts)
            scheduler.step(v[0])
            if v[0] < best - 1e-6:
                best = float(v[0]); best_state = copy.deepcopy(model.state_dict()); patience = 0
            else: patience += 1
            lr = opt.param_groups[0]['lr']
            writer.writerow([epoch, *totals, *v, lr]); f.flush()
            if epoch == 1 or epoch % 10 == 0:
                print(f'epoch {epoch}: train={totals[0]:.6f} val={v[0]:.6f} '
                      f'val_traj={v[1]:.6f} val_der={v[2]:.6f} best={best:.6f} lr={lr:.6g}', flush=True)
            if patience >= int(d.get('early_stopping', {}).get('patience', 60)):
                print(f'Early stopping at epoch {epoch}', flush=True); break
    if best_state is None: raise RuntimeError('No valid checkpoint')
    model.load_state_dict(best_state)
    checkpoint = ckpts / f'{system}_v05.pt'
    torch.save({'state_dict': model.state_dict(), 'dim': xtr.shape[-1], 'hidden': d['hidden'],
                'system': system, 'version': 'v0.5', 'normalized': enabled,
                'best_val': best, 'seed': cfg['seed'], 'training_config': d}, checkpoint)
    print(f'Best checkpoint: {checkpoint} | val_loss={best:.6f}')
    print(f'History: {logpath}')

if __name__ == '__main__': main()
