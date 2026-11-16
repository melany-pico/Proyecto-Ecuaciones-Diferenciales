"""Experiment 27: sealed, common-grid test of temporal-data scarcity.

Run from project root:
    python scripts/27_evaluate_data_scarcity.py --seed 101
    python scripts/27_evaluate_data_scarcity.py --all

All methods and observation levels use exactly the same CLEAN dense test
trajectories, initial conditions, evaluation times and physical horizon.
No model fitting, tuning or selection takes place in this script.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dynadistill.utils.common import ROOT, config
from dynadistill.models.neuralode import VectorField
from dynadistill.data.systems import field

SYSTEM = 'lorenz'
SEEDS = [101, 202, 303, 404, 505]
LEVELS = ['obs_100', 'obs_050', 'obs_025', 'obs_010']
METHODS = ['neural_ode', 'neural_distillation', 'direct_sindy']
HORIZON = 0.80
N_WINDOWS = 8
N_EVAL_TIMES = 9
RTOL = 1e-5
ATOL = 1e-7
MAX_STEP = 0.05
SUPPORT_TOL = 1e-8


def data_path(level, seed):
    return ROOT / 'data' / 'scarcity' / SYSTEM / level / f'seed_{seed}' / 'test.npz'


def model_dir(level, seed):
    return ROOT / 'results' / 'scarcity' / SYSTEM / level / f'seed_{seed}'


def output_dir():
    return ROOT / 'results' / 'scarcity' / SYSTEM / 'evaluation'


def load_reference_test(seed):
    """Read the dense CLEAN test set, never use noisy test values for targets."""
    path = data_path('obs_100', seed)
    with np.load(path, allow_pickle=False) as f:
        t = np.asarray(f['t'], dtype=np.float64)
        x = np.asarray(f['x_clean'], dtype=np.float64)
        ids = np.asarray(f['trajectory_ids'])
    if x.ndim != 3 or x.shape[-1] != 3 or x.shape[1] != len(t):
        raise ValueError(f'Unexpected reference test dimensions: {path}')
    if not np.all(np.diff(t) > 0) or not np.isfinite(x).all():
        raise ValueError(f'Invalid reference test data: {path}')
    if t[-1] - t[0] < HORIZON:
        raise ValueError('Test trajectories shorter than evaluation horizon')
    # Verify each scarcity level references the same held-out trajectories.
    for level in LEVELS[1:]:
        with np.load(data_path(level, seed), allow_pickle=False) as f:
            other_ids = np.asarray(f['trajectory_ids'])
            other_t = np.asarray(f['t'], dtype=np.float64)
            other_clean = np.asarray(f['x_clean'], dtype=np.float64)
            if 'temporal_indices' not in f.files:
                raise ValueError(f'Missing temporal_indices for {level}, seed {seed}')
            indices = np.asarray(f['temporal_indices'], dtype=np.int64)
        if not np.array_equal(ids, other_ids):
            raise ValueError(f'Test trajectory IDs differ for {level}, seed {seed}')
        if not np.array_equal(t[indices], other_t) or not np.array_equal(x[:, indices, :], other_clean):
            raise ValueError(f'Test trajectories not nested for {level}, seed {seed}')
    return x, t, ids


def evaluation_windows(t):
    """Exact common physical start times and 0.8-unit horizons.

    Initial states are linearly interpolated from clean dense test states,
    identically for all methods. The reference at evaluation times is also
    interpolated from the same dense clean data, without using ground-truth
    derivatives. This avoids differences caused by sparse observation grids.
    """
    starts = np.linspace(float(t[0]), float(t[-1] - HORIZON), N_WINDOWS)
    return [s + np.linspace(0.0, HORIZON, N_EVAL_TIMES) for s in starts]


def interpolate_states(x, t, times):
    return np.stack([np.interp(times, t, x[:, j]) for j in range(3)], axis=1)


def true_support():
    # Lorenz: dx=10y-10x; dy=28x-y-xz; dz=xy-(8/3)z.
    return {(1, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 1),
            (0, 1, 0, 1), (1, 0, 1, 1), (0, 0, 1, 2),
            (1, 1, 0, 2)}


def support_scores(coef, degree):
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    poly.fit(np.zeros((1, 3)))
    if coef.shape != (len(poly.powers_), 3):
        raise ValueError(f'Coefficient shape mismatch: {coef.shape}')
    recovered = set()
    for k, power in enumerate(poly.powers_):
        for eq in range(3):
            if abs(float(coef[k, eq])) > SUPPORT_TOL:
                recovered.add((*map(int, power), eq))
    actual = true_support()
    tp = len(recovered & actual)
    fp = len(recovered - actual)
    fn = len(actual - recovered)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'terms': len(recovered), 'tp': tp, 'fp': fp, 'fn': fn,
            'precision': precision, 'recall': recall, 'f1': f1}


def load_symbolic(level, seed, method):
    path = model_dir(level, seed) / 'symbolic' / f'{method}_best.npz'
    with np.load(path, allow_pickle=False) as f:
        coef = np.asarray(f['coef'], dtype=np.float64)
        degree = int(f['degree'])
        estimator = str(f['estimator'])
        selected_val = float(f['val_traj_l2'])
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    poly.fit(np.zeros((1, 3)))
    powers = poly.powers_.astype(np.int32)
    if coef.shape != (len(powers), 3) or not np.isfinite(coef).all():
        raise ValueError(f'Invalid symbolic model: {path}')
    def vector_field(states):
        states = np.asarray(states, dtype=np.float64)
        return np.prod(np.power(states[..., None, :], powers), axis=-1) @ coef
    def rhs(_t, state):
        return vector_field(state)
    return {'field': vector_field, 'rhs': rhs, 'support': support_scores(coef, degree),
            'degree': degree, 'estimator': estimator, 'val_l2': selected_val}


def load_neural(level, seed):
    path = model_dir(level, seed) / 'neural_ode.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('version') != 'scarcity_paired_v1':
        raise ValueError(f'Unexpected neural checkpoint version: {path}')
    model = VectorField(dim=int(checkpoint['dim']), hidden=int(checkpoint['hidden']))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    def vector_field(states):
        states = np.asarray(states, dtype=np.float32)
        with torch.no_grad():
            return model.physical_field(torch.as_tensor(states)).cpu().numpy().astype(np.float64)
    def rhs(_t, state):
        return vector_field(state)
    return {'field': vector_field, 'rhs': rhs, 'support': None,
            'degree': None, 'estimator': 'neural_field', 'val_l2': float('nan')}


def integrate(rhs, x0, times):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            sol = solve_ivp(rhs, (float(times[0]), float(times[-1])), x0,
                            t_eval=times, method='RK45', rtol=RTOL,
                            atol=ATOL, max_step=MAX_STEP)
        if not sol.success or sol.y.shape != (3, len(times)):
            return None
        prediction = sol.y.T
        if not np.isfinite(prediction).all() or np.max(np.abs(prediction)) > 1e6:
            return None
        return prediction
    except (ValueError, RuntimeError, OverflowError, FloatingPointError):
        return None


def relative_l2(predicted, reference):
    return float(np.linalg.norm(predicted - reference) / max(np.linalg.norm(reference), 1e-12))


def evaluate_model(model, x_clean, t, windows):
    window_errors = []
    failed = 0
    for trajectory in x_clean:
        for times in windows:
            truth = interpolate_states(trajectory, t, times)
            prediction = integrate(model['rhs'], truth[0], times)
            if prediction is None:
                failed += 1
                window_errors.append(float('nan'))
            else:
                window_errors.append(relative_l2(prediction, truth))
    # Field error evaluated at ALL common dense clean test states.
    states = x_clean.reshape(-1, 3)
    reference_field = field(SYSTEM, states)
    try:
        predicted_field = np.concatenate(
            [np.asarray(model['field'](batch), dtype=np.float64)
             for batch in np.array_split(states, max(1, (len(states) + 255) // 256))], axis=0)
        field_l2 = relative_l2(predicted_field, reference_field) if np.isfinite(predicted_field).all() else float('nan')
    except (ValueError, RuntimeError, OverflowError, FloatingPointError):
        field_l2 = float('nan')
    errors = np.asarray(window_errors, dtype=np.float64)
    valid = np.isfinite(errors)
    return {'traj_l2_mean': float(np.mean(errors[valid])) if valid.any() else float('nan'),
            'traj_l2_std_windows': float(np.std(errors[valid], ddof=1)) if valid.sum() > 1 else float('nan'),
            'traj_l2_median': float(np.median(errors[valid])) if valid.any() else float('nan'),
            'field_l2': field_l2, 'n_windows': int(len(errors)),
            'failed_windows': failed, 'failure_rate': failed / len(errors)}


def evaluate_condition(seed, level, x_clean, t, windows):
    rows = []
    for method in METHODS:
        model = load_neural(level, seed) if method == 'neural_ode' else load_symbolic(level, seed, method)
        metrics = evaluate_model(model, x_clean, t, windows)
        row = {'seed': seed, 'level': level, 'observation_percent': int(level.split('_')[1]),
               'method': method, 'degree': model['degree'], 'estimator': model['estimator'],
               'selected_val_l2': model['val_l2'], **metrics}
        if model['support'] is not None:
            row.update(model['support'])
        else:
            row.update({k: '' for k in ['terms', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1']})
        rows.append(row)
        f1 = row['f1'] if method != 'neural_ode' else 'N/A'
        print(f'TEST seed={seed} level={level} method={method} '
              f'traj_L2={row["traj_l2_mean"]:.6f} field_L2={row["field_l2"]:.6f} '
              f'F1={f1} failures={row["failed_windows"]}/{row["n_windows"]}', flush=True)
    return rows


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    summaries = []
    for level in LEVELS:
        for method in METHODS:
            group = [r for r in rows if r['level'] == level and r['method'] == method]
            if not group:
                continue
            summary = {'level': level, 'method': method, 'n_seeds': len(group)}
            for key in ['traj_l2_mean', 'field_l2', 'failure_rate', 'terms', 'precision', 'recall', 'f1']:
                vals = np.array([float(r[key]) for r in group if r[key] != ''], dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                summary[f'{key}_mean'] = float(vals.mean()) if len(vals) else float('nan')
                summary[f'{key}_std'] = float(vals.std(ddof=1)) if len(vals) > 1 else float('nan')
            summaries.append(summary)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=SEEDS, default=101)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--level', choices=LEVELS)
    args = parser.parse_args()
    if config()['system'] != SYSTEM:
        raise ValueError('Experiment 27 requires configs/default.yml system: lorenz')
    torch.set_num_threads(1)
    rows = []
    seeds = SEEDS if args.all else [args.seed]
    levels = [args.level] if args.level else LEVELS
    for seed in seeds:
        x_clean, t, ids = load_reference_test(seed)
        windows = evaluation_windows(t)
        print(f'\n=== SEALED TEST seed={seed}: {len(ids)} trajectories, '
              f'{len(windows)} windows/trajectory, horizon={HORIZON} ===', flush=True)
        for level in levels:
            rows.extend(evaluate_condition(seed, level, x_clean, t, windows))
    out = output_dir()
    suffix = 'all' if args.all else f'seed_{args.seed}'
    if args.level:
        suffix += f'_{args.level}'
    save_csv(out / f'test_metrics_{suffix}.csv', rows)
    summary = summarize(rows)
    save_csv(out / f'test_summary_{suffix}.csv', summary)
    metadata = {'experiment': 27, 'system': SYSTEM, 'seeds': seeds, 'levels': levels,
                'methods': METHODS, 'reference': 'obs_100/test.npz:x_clean',
                'horizon': HORIZON, 'windows_per_trajectory': N_WINDOWS,
                'evaluation_points_per_window': N_EVAL_TIMES,
                'field_reference': 'analytical Lorenz vector field',
                'trajectory_reference': 'linear interpolation of clean dense test',
                'selection': 'fixed in experiment 26; no tuning on test',
                'failure_policy': 'failures reported separately; failed windows excluded from mean; compare failure_rate'}
    (out / f'test_protocol_{suffix}.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('\n=== EXPERIMENT 27 COMPLETE ===')
    print(f'Metrics: {out / f"test_metrics_{suffix}.csv"}')
    print(f'Summary: {out / f"test_summary_{suffix}.csv"}')
    print('No training, validation selection, or hyperparameter tuning was performed on test.')
    if any(r['failed_windows'] for r in rows):
        print('WARNING: Some rollouts failed. Compare failure rates alongside valid-window errors.')


if __name__ == '__main__':
    main()
