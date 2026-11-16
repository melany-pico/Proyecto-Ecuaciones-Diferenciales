
"""Experiment 31: Neural distillation vs direct SINDy under temporal scarcity.

Run from project root:
    python scripts/31_discover_multisystem_scarcity.py --system vanderpol --seed 101
    python scripts/31_discover_multisystem_scarcity.py --all

Uses train and validation only; never opens test data. Existing best models
are skipped. All validation rollouts span a common physical time horizon.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import warnings
from scipy.integrate import solve_ivp
from sklearn.preprocessing import PolynomialFeatures

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dynadistill.utils.common import ROOT, config
from dynadistill.models.neuralode import VectorField
from dynadistill.discovery.sparse import stlsq

SYSTEMS = ("vanderpol", "duffing", "lotka_volterra")
SEEDS = [101, 202, 303, 404, 505]
LEVELS = ['obs_100', 'obs_050', 'obs_025', 'obs_010']
DEGREES = [2, 3]
THRESHOLDS = [0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50]
RIDGE = 1e-5
HORIZON = 0.80
WINDOWS_PER_TRAJECTORY = 8
COMPLEXITY_PENALTY = 0.005
DERIVATIVE_WINDOWS = [5, 7, 9, 11, 15, 21]


def data_path(system, level, seed, split):
    return ROOT / 'data' / 'scarcity' / system / level / f'seed_{seed}' / f'{split}.npz'


def result_dir(system, level, seed):
    return ROOT / 'results' / 'scarcity' / system / level / f'seed_{seed}' / 'symbolic'


def read_split(system, level, seed, split):
    if split not in ('train', 'val'):
        raise ValueError('Experiment 31 may only read train/val')
    with np.load(data_path(system, level, seed, split), allow_pickle=False) as f:
        x = np.asarray(f['x'], dtype=np.float64)
        t = np.asarray(f['t'], dtype=np.float64)
    if x.ndim != 3 or x.shape[-1] != 2 or x.shape[1] != len(t):
        raise ValueError('Invalid trajectory dimensions')
    if not np.isfinite(x).all() or not np.isfinite(t).all() or not np.all(np.diff(t) > 0):
        raise ValueError('Invalid values or time grid')
    return x, t


def local_polynomial_derivative(x, t, window, degree=2):
    """Local least-squares derivative using actual, possibly irregular, times."""
    n_traj, n_times, dim = x.shape
    if window > n_times or window % 2 != 1:
        raise ValueError('Invalid local polynomial window')
    result = np.empty_like(x)
    half = window // 2
    for i in range(n_times):
        left = max(0, min(i - half, n_times - window))
        ids = np.arange(left, left + window)
        dt = t[ids] - t[i]
        scale = max(float(np.max(np.abs(dt))), 1e-12)
        u = dt / scale
        design = np.vander(u, N=degree + 1, increasing=True)
        ys = x[:, ids, :].transpose(1, 0, 2).reshape(window, -1)
        beta = np.linalg.lstsq(design, ys, rcond=None)[0]
        result[:, i, :] = (beta[1] / scale).reshape(n_traj, dim)
    return result


def estimate_derivative(x, t, method):
    if method == 'gradient':
        return np.gradient(x, t, axis=1, edge_order=2)
    window = int(method.split('_w')[-1])
    return local_polynomial_derivative(x, t, window)


def validation_windows(t):
    """Choose windows using physical time, respecting observed time points."""
    start_times = np.linspace(t[0], t[-1] - HORIZON, WINDOWS_PER_TRAJECTORY)
    windows = []
    for nominal in start_times:
        start = int(np.argmin(np.abs(t - nominal)))
        if t[start] + HORIZON > t[-1]:
            start = max(0, int(np.searchsorted(t, t[-1] - HORIZON, side='right') - 1))
        end = int(np.searchsorted(t, t[start] + HORIZON, side='right') - 1)
        if end <= start:
            end = start + 1
        if end >= len(t):
            raise ValueError('No valid validation window')
        windows.append((start, end))
    return windows


def rollout_error(coef, degree, x_val, t_val, windows):
    library = PolynomialFeatures(degree=degree, include_bias=True)
    library.fit(np.zeros((1, x_val.shape[-1])))
    powers = library.powers_.astype(np.int32)

    def rhs(_t, state):
        if not np.isfinite(state).all() or np.max(np.abs(state)) > 1e5:
            raise FloatingPointError('Candidate trajectory diverged')
        monomials = np.prod(np.power(state[None, :], powers), axis=1)
        return monomials @ coef

    errors = []
    for trajectory in x_val:
        for start, end in windows:
            times = t_val[start:end + 1]
            observed = trajectory[start:end + 1]
            try:
                sol = solve_ivp(
                    rhs, (float(times[0]), float(times[-1])),
                    observed[0], t_eval=times, method='RK45',
                    rtol=1e-5, atol=1e-7, max_step=0.08
                )
                if not sol.success or sol.y.shape != (x_val.shape[-1], len(times)):
                    return float('inf')
                predicted = sol.y.T
                if not np.isfinite(predicted).all():
                    return float('inf')
                numerator = np.linalg.norm(predicted - observed)
                denominator = max(np.linalg.norm(observed), 1e-12)
                errors.append(numerator / denominator)
            except (ValueError, OverflowError, FloatingPointError, RuntimeError):
                return float('inf')
    return float(np.mean(errors)) if errors else float('inf')


def load_neural_model(system, level, seed):
    path = ROOT / 'results' / 'scarcity' / system / level / f'seed_{seed}' / 'neural_ode.pt'
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if (checkpoint.get('version') != 'scarcity_multisystem_paired_v1'
            or checkpoint.get('system') != system
            or checkpoint.get('seed') != seed
            or checkpoint.get('level') != level):
        raise ValueError(f'Unexpected checkpoint version: {path}')
    if int(checkpoint['dim']) != 2:
        raise ValueError(f'Expected 2D checkpoint: {path}')
    model = VectorField(dim=int(checkpoint['dim']), hidden=int(checkpoint['hidden']))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def neural_targets(model, x_train, max_points, seed):
    states = x_train.reshape(-1, x_train.shape[-1])
    if len(states) > max_points:
        rng = np.random.default_rng(900000 + seed)
        states = states[np.sort(rng.choice(len(states), max_points, replace=False))]
    with torch.no_grad():
        derivative = model.physical_field(
            torch.tensor(states, dtype=torch.float32)
        ).numpy()
    return states, np.asarray(derivative, dtype=np.float64)


def fit_grid(method, train_x, train_t, val_x, val_t, model, seed):
    windows = validation_windows(val_t)
    if method == 'neural_distillation':
        x, dx = neural_targets(model, train_x, 3000, seed)
        estimators = [('neural_field', x, dx)]
    else:
        estimators = []
        methods = ['gradient'] + [
            f'localpoly_w{w}' for w in DERIVATIVE_WINDOWS if w <= len(train_t)
        ]
        for estimator in methods:
            dx = estimate_derivative(train_x, train_t, estimator)
            estimators.append((
                estimator,
                train_x.reshape(-1, train_x.shape[-1]),
                dx.reshape(-1, dx.shape[-1])
            ))

    records = []
    best = None
    for estimator, states, derivatives in estimators:
        for degree in DEGREES:
            for threshold in THRESHOLDS:
                try:
                    library, coef = stlsq(
                        states, derivatives, degree=degree,
                        threshold=threshold, ridge=RIDGE
                    )
                    terms = int(np.count_nonzero(coef))
                    val_l2 = rollout_error(coef, degree, val_x, val_t, windows)
                    score = val_l2 + COMPLEXITY_PENALTY * terms
                except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                    continue

                row = {
                    'method': method,
                    'estimator': estimator,
                    'degree': degree,
                    'threshold': threshold,
                    'terms': terms,
                    'val_traj_l2': val_l2,
                    'score': score,
                }
                records.append(row)
                if np.isfinite(score) and (
                    best is None
                    or (score, terms) < (best['score'], best['terms'])
                ):
                    best = dict(row)
                    best['coef'] = coef.copy()
                    best['names'] = np.asarray(
                        library.get_feature_names_out(), dtype=str
                    )
    if best is None:
        raise RuntimeError(f'No valid candidates for {method}')
    return best, records


def save_best(destination, method, best, records):
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination / f'{method}_best.npz',
        coef=best['coef'], names=best['names'],
        degree=np.int64(best['degree']),
        threshold=np.float64(best['threshold']),
        estimator=np.asarray(best['estimator']),
        terms=np.int64(best['terms']),
        val_traj_l2=np.float64(best['val_traj_l2']),
        score=np.float64(best['score']),
    )
    with (destination / f'{method}_grid.csv').open(
        'w', newline='', encoding='utf-8'
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summary = {
        k: v for k, v in best.items() if k not in ('coef', 'names')
    }
    summary['candidate_count'] = len(records)
    (destination / f'{method}_summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8'
    )


def run_condition(system, seed, level):
    destination = result_dir(system, level, seed)
    methods = ['neural_distillation', 'direct_sindy']
    existing = [
        m for m in methods
        if (destination / f'{m}_best.npz').exists()
    ]
    if len(existing) == len(methods):
        print(
            f'SKIPPED system={system} seed={seed} level={level}: '
            'both symbolic models exist', flush=True
        )
        return

    print(
        f'\n=== DISCOVERY system={system} seed={seed} level={level} ===',
        flush=True
    )
    x_train, t_train = read_split(system, level, seed, 'train')
    x_val, t_val = read_split(system, level, seed, 'val')
    model = load_neural_model(
        system, level, seed
    ) if 'neural_distillation' not in existing else None

    for method in methods:
        if method in existing:
            print(f'SKIPPED {method}: best model exists', flush=True)
            continue
        best, grid = fit_grid(
            method, x_train, t_train, x_val, t_val, model, seed
        )
        save_best(destination, method, best, grid)
        print(
            f'BEST {method} system={system} seed={seed} level={level} '
            f'estimator={best["estimator"]} degree={best["degree"]} '
            f'threshold={best["threshold"]} terms={best["terms"]} '
            f'val_L2={best["val_traj_l2"]:.6f} '
            f'score={best["score"]:.6f} candidates={len(grid)}',
            flush=True
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--system', choices=SYSTEMS)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--level', choices=LEVELS)
    args = parser.parse_args()

    if not args.all and (args.system is None or args.seed is None):
        parser.error('Use --all or provide both --system and --seed')

    torch.set_num_threads(1)
    for system in (SYSTEMS if args.all else (args.system,)):
        for seed in (SEEDS if args.all else (args.seed,)):
            for level in ([args.level] if args.level else LEVELS):
                run_condition(system, seed, level)

    print('\n=== EXPERIMENT 31 COMPLETE ===')
    print('Only train and validation data were accessed. Test remains sealed.')


if __name__ == '__main__':
    main()
