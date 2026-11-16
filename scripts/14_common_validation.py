"""Common trajectory-validation protocol for direct SINDy candidates.

Fit each candidate to derivatives estimated from TRAIN observations, then
rank all candidates on the SAME held-out validation trajectory segments.
The previously selected neural-distillation model is evaluated as a fixed
reference, never retuned. No test data or exact physical derivatives used.
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
from sklearn.preprocessing import PolynomialFeatures

from dynadistill.utils.common import ROOT, config, seed_all
from dynadistill.discovery.sparse import stlsq

DEGREES = (2, 3)
THRESHOLDS = (0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50)
ESTIMATORS = [('gradient', None)] + [('savgol', w) for w in (5, 7, 9, 11, 15, 21)]
COMPLEXITY_PENALTY = 0.005
SEGMENT_POINTS = 9  # 8 time intervals, approximately 0.40 Lorenz time units
SEGMENT_STRIDE = 16
MAX_ABS_STATE = 1e5


def load_split(system, split):
    path = ROOT / 'data' / 'processed' / f'{system}_{split}.npz'
    with np.load(path) as data:
        x = np.asarray(data['x'], dtype=np.float64)
        t = np.asarray(data['t'], dtype=np.float64) if 't' in data else None
    if x.ndim != 3 or not np.isfinite(x).all():
        raise ValueError(f'Invalid observations: {path}')
    if t is not None:
        t = t.reshape(-1)
        if len(t) != x.shape[1]:
            raise ValueError(f'Invalid time grid in {path}')
    return x, t


def time_grid(x, saved_t, cfg):
    if saved_t is not None:
        return saved_t
    return np.linspace(0.0, float(cfg['simulation']['t_end']), x.shape[1])


def estimate(x, dt, method, window):
    if method == 'gradient':
        return np.gradient(x, dt, axis=1, edge_order=2)
    return savgol_filter(x, window_length=window, polyorder=3,
                         deriv=1, delta=dt, axis=1, mode='interp')


def segments(x, t):
    if len(t) < SEGMENT_POINTS:
        raise ValueError('Validation trajectories are shorter than segment length')
    # Use non-overlapping-ish fixed starts, same segments for every candidate.
    starts = list(range(0, len(t) - SEGMENT_POINTS + 1, SEGMENT_STRIDE))
    return [(x[i, start:start + SEGMENT_POINTS],
             t[start:start + SEGMENT_POINTS] - t[start])
            for i in range(len(x)) for start in starts]


def predict_segment(library, coef, x0, times):
    def rhs(_t, state):
        with np.errstate(over='ignore', invalid='ignore'):
            result = library.transform(np.asarray(state).reshape(1, -1))[0] @ coef
        if not np.isfinite(result).all() or np.max(np.abs(state)) > MAX_ABS_STATE:
            raise FloatingPointError('Divergent polynomial field')
        return result
    try:
        sol = solve_ivp(rhs, (float(times[0]), float(times[-1])), x0,
                        t_eval=times, method='RK45', rtol=1e-5, atol=1e-7)
        if not sol.success or sol.y.shape[1] != len(times):
            return None
        pred = sol.y.T
        if not np.isfinite(pred).all() or np.max(np.abs(pred)) > MAX_ABS_STATE:
            return None
        return pred
    except (ValueError, RuntimeError, OverflowError, FloatingPointError):
        return None


def trajectory_error(library, coef, validation_segments):
    squared_error = 0.0
    squared_reference = 0.0
    failures = 0
    for observed, times in validation_segments:
        pred = predict_segment(library, coef, observed[0], times)
        if pred is None:
            failures += 1
            continue
        squared_error += float(np.sum((pred - observed) ** 2))
        squared_reference += float(np.sum(observed ** 2))
    if failures:
        return float('inf'), failures
    return float(np.sqrt(squared_error / max(squared_reference, 1e-12))), 0


def load_distilled(path, dim):
    with np.load(path, allow_pickle=True) as data:
        coef = np.asarray(data['coef'], dtype=np.float64)
        names = np.asarray(data['names']).astype(str)
        degree = int(data['degree'])
    library = PolynomialFeatures(degree=degree, include_bias=True)
    library.fit(np.zeros((1, dim)))
    expected = library.get_feature_names_out([f'x{i}' for i in range(dim)])
    if not np.array_equal(names, expected):
        raise ValueError('Distilled model feature names do not match polynomial library')
    if coef.shape != (len(expected), dim):
        raise ValueError('Invalid distilled coefficient shape')
    return library, coef


def main():
    cfg = config()
    seed_all(cfg['seed'])
    system = cfg['system']
    train, train_t_saved = load_split(system, 'train')
    val, val_t_saved = load_split(system, 'val')
    t_train = time_grid(train, train_t_saved, cfg)
    t_val = time_grid(val, val_t_saved, cfg)
    dt = float(np.mean(np.diff(t_train)))
    if dt <= 0 or not np.allclose(np.diff(t_train), dt, rtol=1e-5, atol=1e-9):
        raise ValueError('Training time grid must be uniformly spaced')
    if not np.allclose(t_val, t_train):
        raise ValueError('Train and validation time grids must match')
    if train.shape[2] != val.shape[2]:
        raise ValueError('Train and validation state dimensions differ')
    validation_segments = segments(val, t_val)
    x_all = train.reshape(-1, train.shape[-1])
    n = min(len(x_all), int(cfg['discovery']['sample_points']))
    rng = np.random.default_rng(cfg['seed'])
    indices = rng.choice(len(x_all), size=n, replace=False)
    x_train = x_all[indices]
    output = ROOT / 'results' / 'baselines'
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    best = None

    print('\n=== COMMON TRAJECTORY VALIDATION ===')
    print(f'System: {system} | segments: {len(validation_segments)} | '
          f'points per segment: {SEGMENT_POINTS} | dt: {dt:.6f}')
    print('Selection: trajectory relative L2 + 0.005 * number of terms')
    print('No test data and no oracle derivatives are used.\n')

    for method, window in ESTIMATORS:
        label = method if window is None else f'{method}_w{window}'
        y_all = estimate(train, dt, method, window).reshape(-1, train.shape[-1])
        y_train = y_all[indices]
        case_best = None
        for degree in DEGREES:
            for threshold in THRESHOLDS:
                library, coef = stlsq(x_train, y_train, degree=degree,
                                      threshold=threshold,
                                      ridge=float(cfg['discovery']['ridge']))
                terms = int(np.count_nonzero(np.abs(coef) > 1e-10))
                error, failures = trajectory_error(library, coef, validation_segments)
                score = error + COMPLEXITY_PENALTY * terms
                row = dict(estimator=label, degree=degree, threshold=threshold,
                           n_terms=terms, trajectory_relative_l2=error,
                           failed_segments=failures, score=score)
                rows.append(row)
                if case_best is None or score < case_best['score']:
                    case_best = row
                if best is None or score < best['row']['score']:
                    best = dict(row=row.copy(), coef=coef.copy(),
                                names=library.get_feature_names_out(
                                    [f'x{i}' for i in range(x_train.shape[1])]))
        print(f'{label:14s} degree={case_best["degree"]} '
              f'threshold={case_best["threshold"]:.2f} '
              f'terms={case_best["n_terms"]:2d} '
              f'traj_L2={case_best["trajectory_relative_l2"]:.6f} '
              f'score={case_best["score"]:.6f}')

    if best is None or not np.isfinite(best['row']['score']):
        raise RuntimeError('All direct SINDy candidates failed trajectory validation')

    distilled_path = (ROOT / 'results' / 'equations' /
                      f'{system}_v05_symbolic_tuned.npz')
    distilled_library, distilled_coef = load_distilled(distilled_path, train.shape[-1])
    distilled_error, distilled_failures = trajectory_error(
        distilled_library, distilled_coef, validation_segments)
    distilled_terms = int(np.count_nonzero(np.abs(distilled_coef) > 1e-10))
    distilled_score = distilled_error + COMPLEXITY_PENALTY * distilled_terms

    print('\n=== BEST DIRECT SINDy (COMMON VALIDATION) ===')
    print(json.dumps(best['row'], indent=2))
    print('\n=== FROZEN NEURAL DISTILLATION REFERENCE ===')
    print(f'terms={distilled_terms} traj_L2={distilled_error:.6f} '
          f'failed_segments={distilled_failures} score={distilled_score:.6f}')
    print('Note: the neural model was selected previously; this is a diagnostic comparison,')
    print('not a fresh, unbiased model-selection competition.')

    csv_path = output / f'{system}_common_validation_grid.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    model_path = output / f'{system}_sindy_common_validation_best.npz'
    np.savez(model_path, coef=best['coef'], names=np.asarray(best['names'], dtype=str),
             degree=best['row']['degree'], threshold=best['row']['threshold'],
             estimator=best['row']['estimator'])
    summary = dict(system=system, seed=int(cfg['seed']),
                   segment_points=SEGMENT_POINTS, segment_stride=SEGMENT_STRIDE,
                   validation_segments=len(validation_segments),
                   complexity_penalty=COMPLEXITY_PENALTY,
                   best_direct_sindy=best['row'],
                   frozen_neural_distillation=dict(
                       n_terms=distilled_terms, trajectory_relative_l2=distilled_error,
                       failed_segments=distilled_failures, score=distilled_score),
                   caution='Validation reused during prior development; no final generalization claim.')
    summary_path = output / f'{system}_common_validation_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'\nGrid: {csv_path}\nBest SINDy model: {model_path}\nSummary: {summary_path}')


if __name__ == '__main__':
    main()
