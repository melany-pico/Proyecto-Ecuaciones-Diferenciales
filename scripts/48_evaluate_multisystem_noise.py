#!/usr/bin/env python
"""Experiment 44: sealed observational-noise evaluation, no model selection.

Pilot: python scripts/48_evaluate_multisystem_noise.py --system vanderpol --seed 606 --noise noise_000
All:   python scripts/48_evaluate_multisystem_noise.py --all

Three methods, five noise levels, fifteen seeds and three systems.
Each model is tested on the SAME 5 dense, clean test trajectories and
8 fixed windows per trajectory. Both clean and independently noisy starts.
Test files are never used to fit/tune.
"""
import argparse
import hashlib
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
SEEDS = (606,707,808,909,1010,1111,1212,1313,1414,1515,1616,1717,1818,1919,2020)
LEVELS = ('noise_000', 'noise_001', 'noise_005', 'noise_010', 'noise_020')
SIGMAS = dict(zip(LEVELS, (0.0, 0.01, 0.05, 0.10, 0.20)))
POLICIES = ('clean_start', 'noisy_start')
METHODS = ('neural_ode', 'neural_distillation', 'direct_sindy')
HORIZON = 0.8
WINDOWS = 8
FIELD_SAMPLES = 800  # deterministic points from dense, clean test states


def result_root(system):
    return ROOT / 'results' / 'robustness_multisystem_44' / 'evaluation' / system


def model_folder(system, level, seed):
    return ROOT / 'results' / 'robustness_multisystem_44' / system / level / f'seed_{seed}'


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
    if (ckpt.get('version') != 'robustness_multisystem_44_paired_v1' or ckpt.get('system') != system or ckpt.get('level') != level
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


def perturbation(system, seed, level, trajectory_id, window_index):
    """Independent, paired across methods, deterministic per test window."""
    token = f'exp44|test_start|{system}|{seed}|{level}|{trajectory_id}|{window_index}'
    digest = hashlib.sha256(token.encode('utf-8')).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
    return rng.normal(0.0, SIGMAS[level], size=2) if SIGMAS[level] else np.zeros(2)


def evaluate_one(system, seed, level, method, policy, x_test, t_test, ids, windows):
    if method == 'neural_ode':
        rhs, batch, symbolic = load_neural(system, level, seed)
    else:
        rhs, batch, symbolic = load_symbolic(system, level, seed, method)
    f1, terms = structural_f1(system, symbolic)
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
        tid = int(ids[traj_index])
        for window_index, (start, times) in enumerate(windows):
            reference = reference_at(trajectory, t_test, times)
            offset = (perturbation(system, seed, level, tid, window_index)
                      if policy == 'noisy_start' else np.zeros(2))
            prediction = integrate(rhs, times, reference[0] + offset)
            success = prediction is not None
            error = (float(np.linalg.norm(prediction - reference) /
                           max(np.linalg.norm(reference), 1e-12))
                     if success else float('nan'))
            rows.append(dict(system=system, seed=seed, level=level, noise_std=SIGMAS[level],
                             start_policy=policy, method=method, trajectory_id=tid,
                             window=window_index, horizon=HORIZON, success=int(success),
                             traj_rel_l2=error))
    errors = [r['traj_rel_l2'] for r in rows if r['success']]
    failures = len(rows) - len(errors)
    trajectory_l2 = float(np.mean(errors)) if failures == 0 else float('inf')
    summary = dict(system=system, seed=seed, level=level, noise_std=SIGMAS[level],
                   start_policy=policy, method=method, traj_rel_l2=trajectory_l2,
                   field_rel_l2=field_l2, structural_f1=f1, nonzero_terms=terms,
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
    parser.add_argument('--noise', choices=LEVELS)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    if args.all and any(x is not None for x in (args.system, args.seed, args.noise)):
        parser.error('--all cannot be combined with individual selectors')
    if not args.all and any(x is None for x in (args.system, args.seed, args.noise)):
        parser.error('Specify --all or --system, --seed and --noise')
    protocol_path = ROOT / 'results' / 'robustness_multisystem_44' / 'protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    if (protocol.get('experiment') != 44 or tuple(protocol['systems']) != SYSTEMS
            or tuple(protocol['seeds']) != SEEDS or tuple(protocol['noise_labels']) != LEVELS):
        raise ValueError('Frozen protocol mismatch')
    torch.set_num_threads(1)
    systems = SYSTEMS if args.all else (args.system,)
    seeds = SEEDS if args.all else (args.seed,)
    levels = LEVELS if args.all else (args.noise,)
    # Preflight output paths BEFORE opening any test data. Never overwrite.
    destinations = {}
    for system in systems:
        out = result_root(system)
        suffix = 'all' if args.all else f'seed_{args.seed}_{args.noise}'
        paths = (out / f'test_metrics_{suffix}.csv', out / f'test_windows_{suffix}.csv')
        if any(path.exists() for path in paths):
            raise FileExistsError(f'Refusing to overwrite existing evaluation: {paths}')
        destinations[system] = paths
    for system in systems:
        summaries, detail = [], []
        for seed in seeds:
            x_test, t_test, ids = load_clean_test(system, seed)
            windows = fixed_windows(t_test)
            for level in levels:
                for policy in POLICIES:
                    for method in METHODS:
                        summary, rows = evaluate_one(system, seed, level, method,
                                                     policy, x_test, t_test, ids, windows)
                        summaries.append(summary)
                        detail.extend(rows)
                        print(f'{system} seed={seed} {level} {policy} {method}: '
                              f'traj_L2={summary["traj_rel_l2"]:.6f} '
                              f'field_L2={summary["field_rel_l2"]:.6f} '
                              f'F1={summary["structural_f1"]:.4f} '
                              f'failures={summary["failures"]}/{summary["windows"]}',
                              flush=True)
        metrics_path, windows_path = destinations[system]
        write_csv(metrics_path, summaries)
        write_csv(windows_path, detail)
        print(f'SAVED {metrics_path} | models={len(summaries)} windows={len(detail)}', flush=True)
    print('\n=== EXPERIMENT 44 SEALED EVALUATION COMPLETE ===')
    print('No test-based fitting or model selection performed.')


if __name__ == '__main__':
    main()
