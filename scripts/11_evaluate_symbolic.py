"""Independent test evaluation of a frozen symbolic Neural ODE distillation.

Run from the repository root: python scripts/11_evaluate_symbolic.py
Does not fit, tune, or modify the symbolic model.
"""
import sys
import csv
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
import torch
from scipy.integrate import solve_ivp
from sklearn.preprocessing import PolynomialFeatures

from dynadistill.utils.common import ROOT, config, seed_all
from dynadistill.data.systems import field
from dynadistill.models.neuralode import VectorField


def metrics(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    delta = pred - target
    return {
        'mse': float(np.mean(delta ** 2)),
        'rmse': float(np.sqrt(np.mean(delta ** 2))),
        'relative_l2': float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(target.ravel()), 1e-12)),
    }


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    cfg = config()
    seed_all(cfg['seed'])
    system = cfg['system']
    if system != 'lorenz':
        raise ValueError('This script currently includes Lorenz-specific support metrics.')

    test_path = ROOT / 'data' / 'processed' / f'{system}_test.npz'
    symbol_path = ROOT / 'results' / 'equations' / f'{system}_v05_symbolic_tuned.npz'
    checkpoint_path = ROOT / 'results' / 'checkpoints' / f'{system}_v05.pt'
    for p in (test_path, symbol_path, checkpoint_path):
        if not p.exists():
            raise FileNotFoundError(f'Missing required file: {p}')

    with np.load(test_path) as data:
        t = np.asarray(data['t'], dtype=np.float64)
        observed = np.asarray(data['x'], dtype=np.float64)
        # Use clean held-out states if available, avoiding measurement-noise bias.
        clean = np.asarray(data['clean'], dtype=np.float64) if 'clean' in data.files else None
    if clean is None:
        print('WARNING: clean test states unavailable; field errors use noisy observed states.')
    states = clean if clean is not None else observed
    if states.ndim != 3 or states.shape[-1] != 3:
        raise ValueError('Expected test states with shape (trajectories, time, 3).')
    if len(t) != states.shape[1] or np.any(np.diff(t) <= 0):
        raise ValueError('Invalid test time grid.')

    with np.load(symbol_path, allow_pickle=True) as data:
        coefficients = np.asarray(data['coef'], dtype=np.float64)
        saved_names = list(map(str, data['names']))
        degree = int(data['degree']) if 'degree' in data.files else 2
        threshold = float(data['threshold']) if 'threshold' in data.files else float('nan')

    library = PolynomialFeatures(degree=degree, include_bias=True)
    library.fit(np.zeros((1, 3)))
    names = list(library.get_feature_names_out(['x0', 'x1', 'x2']))
    if names != saved_names or coefficients.shape != (len(names), 3):
        raise ValueError('Saved symbolic library and coefficients are incompatible.')

    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = VectorField(dim=ck['dim'], hidden=ck['hidden'])
    model.load_state_dict(ck['state_dict'])
    model.eval()

    def symbolic_field(x):
        x = np.asarray(x, dtype=np.float64)
        return library.transform(x.reshape(-1, 3)) @ coefficients

    flat = states.reshape(-1, 3)
    true_dx = field(system, flat)
    with torch.no_grad():
        neural_dx = np.concatenate([
            model.physical_field(torch.tensor(batch, dtype=torch.float32)).cpu().numpy()
            for batch in np.array_split(flat, max(1, int(np.ceil(len(flat) / 1024))))
        ]).astype(np.float64)
    symbolic_dx = symbolic_field(flat)

    field_rows = []
    for label, prediction, target in (
        ('neural_vs_true', neural_dx, true_dx),
        ('symbolic_vs_true', symbolic_dx, true_dx),
        ('symbolic_vs_neural', symbolic_dx, neural_dx),
    ):
        result = {'comparison': label, **metrics(prediction, target)}
        field_rows.append(result)

    # Ground-truth support is used ONLY to report test-time recovery, not to tune.
    true_coefficients = np.zeros_like(coefficients)
    expected = {
        (0, 'x0'): -10.0, (0, 'x1'): 10.0,
        (1, 'x0'): 28.0, (1, 'x1'): -1.0, (1, 'x0 x2'): -1.0,
        (2, 'x2'): -8.0 / 3.0, (2, 'x0 x1'): 1.0,
    }
    for (output, term), value in expected.items():
        true_coefficients[names.index(term), output] = value
    actual_support = np.abs(true_coefficients) > 1e-10
    predicted_support = np.abs(coefficients) > 1e-10
    tp = int(np.count_nonzero(actual_support & predicted_support))
    fp = int(np.count_nonzero(~actual_support & predicted_support))
    fn = int(np.count_nonzero(actual_support & ~predicted_support))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    support = {
        'selected_terms': int(predicted_support.sum()),
        'true_terms': int(actual_support.sum()),
        'true_positives': tp, 'false_positives': fp, 'false_negatives': fn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'coefficient_relative_l2': float(np.linalg.norm(coefficients - true_coefficients) / np.linalg.norm(true_coefficients)),
    }

    # Evaluate only short prediction windows. Long-horizon pointwise error is
    # intrinsically sensitive to small perturbations in chaotic systems.
    horizon = 1.0  # physical time units; fixed before inspecting test outcomes
    relative_t = t - t[0]
    window_indices = np.where(relative_t <= horizon + 1e-12)[0]
    tw = relative_t[window_indices]
    if len(tw) < 2:
        raise ValueError('Too few test samples in short trajectory window.')

    trajectory_rows = []
    example = None
    for i, reference in enumerate(states):
        x0 = reference[0].copy()
        target = reference[window_indices]
        with torch.no_grad():
            neural_traj = model.physical_trajectory(
                torch.tensor(x0[None, :], dtype=torch.float32),
                torch.tensor(tw, dtype=torch.float32),
            ).cpu().numpy()[0].astype(np.float64)
        sol = solve_ivp(
            lambda _t, x: symbolic_field(x)[0],
            (float(tw[0]), float(tw[-1])), x0, t_eval=tw,
            rtol=1e-7, atol=1e-9,
        )
        if not sol.success or sol.y.shape[1] != len(tw):
            raise RuntimeError(f'Symbolic integration failed on test trajectory {i}: {sol.message}')
        symbolic_traj = sol.y.T
        for label, prediction in (('neural', neural_traj), ('symbolic', symbolic_traj)):
            trajectory_rows.append({
                'trajectory': i, 'model': label, 'horizon': float(tw[-1]),
                **metrics(prediction, target),
            })
        if i == 0:
            example = (tw, target, neural_traj, symbolic_traj)

    metrics_dir = ROOT / 'results' / 'metrics'
    figures_dir = ROOT / 'results' / 'figures'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    write_csv(metrics_dir / f'{system}_symbolic_v05_test_field.csv', field_rows)
    write_csv(metrics_dir / f'{system}_symbolic_v05_test_trajectories.csv', trajectory_rows)

    summary = {
        'system': system, 'split': 'test', 'n_trajectories': int(len(states)),
        'n_test_states': int(len(flat)), 'states_source': 'clean' if clean is not None else 'observed_noisy',
        'degree': degree, 'threshold': threshold,
        'field_metrics': field_rows, 'support_metrics': support,
        'trajectory_horizon': float(tw[-1]),
        'trajectory_summary': {
            label: {
                key: float(np.mean([r[key] for r in trajectory_rows if r['model'] == label]))
                for key in ('rmse', 'relative_l2')
            }
            for label in ('neural', 'symbolic')
        },
        'note': 'Short-horizon trajectory scores use held-out reference states and their initial conditions.',
    }
    json_path = metrics_dir / f'{system}_symbolic_v05_test_summary.json'
    json_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        tw, target, neural_traj, symbolic_traj = example
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        for j, ax in enumerate(axes):
            ax.plot(tw, target[:, j], label='Lorenz reference', linewidth=2)
            ax.plot(tw, neural_traj[:, j], '--', label='Neural ODE')
            ax.plot(tw, symbolic_traj[:, j], ':', label='Symbolic')
            ax.set_ylabel(f'x{j}')
            ax.grid(alpha=0.25)
        axes[0].legend()
        axes[-1].set_xlabel('Time since test initial condition')
        fig.suptitle('Held-out test: short-horizon trajectories')
        fig.tight_layout()
        fig_path = figures_dir / f'{system}_symbolic_v05_test_trajectories.png'
        fig.savefig(fig_path, dpi=180)
        plt.close(fig)
        print(f'Figure: {fig_path}')
    except ImportError:
        print('Matplotlib not installed: skipping figure.')

    print('\n=== INDEPENDENT TEST: VECTOR FIELD ===')
    for row in field_rows:
        print(f"{row['comparison']:>20}: RMSE={row['rmse']:.6f}, relative_L2={row['relative_l2']:.6f}")
    print('\n=== SUPPORT RECOVERY ===')
    print(f"TP={tp}, FP={fp}, FN={fn}, precision={precision:.4f}, recall={recall:.4f}, F1={f1:.4f}")
    print(f"Coefficient relative L2={support['coefficient_relative_l2']:.6f}")
    print('\n=== SHORT-HORIZON TEST TRAJECTORIES ===')
    for label, values in summary['trajectory_summary'].items():
        print(f"{label}: mean RMSE={values['rmse']:.6f}, mean relative_L2={values['relative_l2']:.6f}")
    print(f'CSV: {metrics_dir}')
    print(f'JSON: {json_path}')


if __name__ == '__main__':
    main()
