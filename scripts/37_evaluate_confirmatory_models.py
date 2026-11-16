#!/usr/bin/env python
"""Experiment 32: sealed multisystem evaluation, no model selection.

Pilot: python scripts/32_evaluate_multisystem_scarcity.py --system vanderpol --seed 101
All:   python scripts/32_evaluate_multisystem_scarcity.py --all

Three methods, four observation levels, five seeds and three systems.
Each model is tested on the SAME 5 dense, clean test trajectories and
8 fixed windows per trajectory. Test files are never used to fit/tune.
"""
import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp
from sklearn.preprocessing import PolynomialFeatures

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dynadistill.models.neuralode import VectorField
from src.dynadistill.data.systems import field as true_field

SYSTEMS = ('vanderpol', 'duffing', 'lotka_volterra')
SEEDS = (606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818, 1919, 2020)
LEVELS = ('obs_100', 'obs_050', 'obs_025', 'obs_010')
METHODS = ('neural_ode', 'neural_distillation', 'direct_sindy')
HORIZON = 0.8
WINDOWS = 8
FIELD_SAMPLES = 800  # deterministic points from dense, clean test states


def result_root(system):
    return ROOT / 'results' / 'scarcity' / 'confirmatory_34' / 'evaluation' / system


def model_folder(system, level, seed):
    return ROOT / 'results' / 'scarcity' / system / level / f'seed_{seed}'


def load_clean_test(system, seed):
    """Open test once per system/seed; obs_100 is the shared dense grid."""
    path = ROOT / 'data' / 'scarcity' / system / 'obs_100' / f'seed_{seed}' / 'test.npz'
    with np.load(path, allow_pickle=False) as f:
        x = np.asarray(f['x_clean'], dtype=np.float64)
        t = np.asarray(f['t'], dtype=np.float64)
        ids = np.asarray(f['trajectory_ids'], dtype=np.int64)
    if x.shape != (5, 160, 2) or t.shape != (160,) or len(np.unique(ids)) != 5:
        raise ValueError(f'Unexpected sealed test shape/IDs: {path}')
    if not np.isfinite(x).all() or not np.all(np.diff(t) > 0):
        raise ValueError(f'Invalid sealed test: {path}')
    return x, t, ids


def load_neural(system, level, seed):
    path = model_folder(system, level, seed) / 'neural_ode.pt'
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if (ckpt.get('system') != system or ckpt.get('level') != level
            or int(ckpt.get('seed', -1)) != seed or int(ckpt['dim']) != 2):
        raise ValueError(f'Checkpoint mismatch: {path}')
    model = VectorField(dim=2, hidden=int(ckpt['hidden']))
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    def rhs(_time, state):
        with torch.no_grad():
            y = model.physical_field(torch.as_tensor(state, dtype=torch.float32))
        return y.detach().cpu().numpy().astype(np.float64)
    def batch(states):
        with torch.no_grad():
            return model.physical_field(torch.as_tensor(states, dtype=torch.float32)).cpu().numpy().astype(np.float64)
    return rhs, batch, None


def load_symbolic(system, level, seed, method):
    path = model_folder(system, level, seed) / 'symbolic' / f'{method}_best.npz'
    with np.load(path, allow_pickle=False) as f:
        coef = np.asarray(f['coef'], dtype=np.float64)
        degree = int(f['degree'])
        names = [str(n) for n in f['names']]
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    poly.fit(np.zeros((1, 2)))
    powers = np.asarray(poly.powers_, dtype=np.int32)
    if coef.shape != (len(powers), 2):
        raise ValueError(f'Coefficient shape mismatch in {path}: {coef.shape}')
    if len(names) != len(powers) or not np.isfinite(coef).all():
        raise ValueError(f'Invalid symbolic terms in {path}')
    def batch(states):
        s = np.asarray(states, dtype=np.float64).reshape(-1, 2)
        terms = np.prod(np.power(s[:, None, :], powers[None, :, :]), axis=2)
        return terms @ coef
    def rhs(_time, state):
        return batch(np.asarray(state).reshape(1, 2))[0]
    return rhs, batch, (powers, coef)


def true_support(system):
    # Exponents of x0 and x1 for each equation, from systems.py.
    if system == 'vanderpol':
        return ({(0, 1)}, {(0, 1), (1, 0), (2, 1)})
    if system == 'duffing':
        return ({(0, 1)}, {(0, 1), (1, 0), (3, 0)})
    if system == 'lotka_volterra':
        return ({(1, 0), (1, 1)}, {(1, 1), (0, 1)})
    raise ValueError(system)


def structural_f1(system, symbolic):
    if symbolic is None:
        return float('nan'), 0
    powers, coef = symbolic
    # Small numerical coefficients should not count as recovered terms.
    predicted = [
        {tuple(map(int, powers[i])) for i in range(len(powers)) if abs(coef[i, j]) > 1e-8}
        for j in range(2)
    ]
    truth = true_support(system)
    tp = sum(len(predicted[j] & truth[j]) for j in range(2))
    fp = sum(len(predicted[j] - truth[j]) for j in range(2))
    fn = sum(len(truth[j] - predicted[j]) for j in range(2))
    denominator = 2 * tp + fp + fn
    return (2 * tp / denominator if denominator else 1.0), sum(map(len, predicted))


def fixed_windows(t):
    """Eight nearly equispaced start indices, exact 0.8 physical-time endpoint."""
    starts = np.linspace(float(t[0]), float(t[-1]) - HORIZON, WINDOWS)
    result = []
    for start in starts:
        i = int(np.argmin(np.abs(t - start)))
        if t[i] + HORIZON > t[-1]:
            i = int(np.searchsorted(t, t[-1] - HORIZON, side='right') - 1)
        end = t[i] + HORIZON
        sample_t = t[(t >= t[i]) & (t <= end + 1e-10)]
        # Include exact horizon endpoint (interpolate true trajectory there).
        if abs(sample_t[-1] - end) > 1e-9:
            sample_t = np.r_[sample_t, end]
        result.append((i, np.asarray(sample_t, dtype=np.float64)))
    return result


def reference_at(x, t, sample_t):
    return np.stack([np.interp(sample_t, t, x[:, j]) for j in range(2)], axis=-1)


def integrate(rhs, t_eval, x0):
    """Bound solver work; report failures instead of silently discarding them."""
    def safe_rhs(time, state):
        if not np.isfinite(state).all() or np.max(np.abs(state)) > 1e5:
            raise FloatingPointError('Diverged state')
        y = rhs(time, state)
        if not np.isfinite(y).all() or np.max(np.abs(y)) > 1e7:
            raise FloatingPointError('Diverged derivative')
        return y
    try:
        sol = solve_ivp(safe_rhs, (float(t_eval[0]), float(t_eval[-1])),
                        x0, t_eval=t_eval, method='RK45', rtol=1e-5,
                        atol=1e-7, max_step=0.08)
        if not sol.success or sol.y.shape != (2, len(t_eval)):
            return None
        out = sol.y.T
        if not np.isfinite(out).all():
            return None
        return out
    except (ValueError, OverflowError, FloatingPointError, RuntimeError):
        return None


def evaluate_one(system, seed, level, method, x_test, t_test, ids, windows):
    if method == 'neural_ode':
        rhs, batch, symbolic = load_neural(system, level, seed)
    else:
        rhs, batch, symbolic = load_symbolic(system, level, seed, method)
    f1, terms = structural_f1(system, symbolic)
    # Same deterministic clean test states across all models and scarcity levels.
    states = x_test.reshape(-1, 2)
    sample_idx = np.linspace(0, len(states) - 1, min(FIELD_SAMPLES, len(states)), dtype=int)
    selected = states[sample_idx]
    true_derivative = true_field(system, selected)
    try:
        estimated = batch(selected)
        field_l2 = float(np.linalg.norm(estimated - true_derivative) /
                         max(np.linalg.norm(true_derivative), 1e-12))
        if not np.isfinite(field_l2):
            field_l2 = float('inf')
    except (ValueError, OverflowError, FloatingPointError, RuntimeError):
        field_l2 = float('inf')

    rows = []
    for traj_index, trajectory in enumerate(x_test):
        for window_index, (start, times) in enumerate(windows):
            reference = reference_at(trajectory, t_test, times)
            prediction = integrate(rhs, times, reference[0])
            success = prediction is not None
            error = (float(np.linalg.norm(prediction - reference) /
                           max(np.linalg.norm(reference), 1e-12))
                     if success else float('nan'))
            rows.append(dict(system=system, seed=seed, level=level, method=method,
                             trajectory_id=int(ids[traj_index]), window=window_index,
                             horizon=HORIZON, success=int(success), traj_rel_l2=error))
    errors = [r['traj_rel_l2'] for r in rows if r['success']]
    failures = len(rows) - len(errors)
    # A failed integration makes the model's aggregate trajectory metric infinite.
    # Do not rank a model using only the successful (survivor) windows.
    trajectory_l2 = float(np.mean(errors)) if failures == 0 else float('inf')
    summary = dict(system=system, seed=seed, level=level, method=method,
                   traj_rel_l2=trajectory_l2, field_rel_l2=field_l2,
                   structural_f1=f1, nonzero_terms=terms,
                   windows=len(rows), failures=failures,
                   successful_window_mean=float(np.mean(errors)) if errors else float('nan'))
    return summary, rows


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--system', choices=SYSTEMS)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    if not args.all and (args.system is None or args.seed is None):
        parser.error('Use --all or both --system and --seed')
    torch.set_num_threads(1)
    systems = SYSTEMS if args.all else (args.system,)
    seeds = SEEDS if args.all else (args.seed,)
    for system in systems:
        summaries, detail = [], []
        for seed in seeds:
            x_test, t_test, ids = load_clean_test(system, seed)
            windows = fixed_windows(t_test)
            for level in LEVELS:
                for method in METHODS:
                    summary, rows = evaluate_one(system, seed, level, method,
                                                 x_test, t_test, ids, windows)
                    summaries.append(summary)
                    detail.extend(rows)
                    print(f'{system} seed={seed} {level} {method}: '
                          f'traj_L2={summary["traj_rel_l2"]:.6f} '
                          f'field_L2={summary["field_rel_l2"]:.6f} '
                          f'F1={summary["structural_f1"]:.4f} '
                          f'failures={summary["failures"]}/{summary["windows"]}',
                          flush=True)
        out = result_root(system)
        suffix = '_all' if args.all else f'_seed_{args.seed}'
        write_csv(out / f'test_metrics{suffix}.csv', summaries)
        write_csv(out / f'test_windows{suffix}.csv', detail)
        print(f'SAVED {out} | models={len(summaries)} windows={len(detail)}', flush=True)
        for level in LEVELS:
            print(f'  {system} {level}:', end=' ')
            for method in METHODS:
                values = [r['traj_rel_l2'] for r in summaries
                          if r['level'] == level and r['method'] == method]
                print(f'{method}={np.mean(values):.5f}', end='  ')
            print()
    print('\n=== EXPERIMENT 32 COMPLETE ===')
    print('Sealed test evaluated once, with no parameter fitting or model selection.')


if __name__ == '__main__':
    main()
