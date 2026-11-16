"""Experiment 28: paired statistical analysis of sealed scarcity test results.

Run from the project root:
    python scripts/28_statistical_analysis_scarcity.py
    python scripts/28_statistical_analysis_scarcity.py --bootstrap 20000

Input: results/scarcity/lorenz/evaluation/test_metrics_all.csv (Experiment 27).
Output: results/scarcity/lorenz/statistics/ (CSV, JSON, PNG, PDF).

This is POST-HOC analysis only. No fitting, model selection, or test reruns.
Statistical unit: independent seed, NOT individual rollout windows.
Exact sign-flip tests assess paired mean differences; with n=5, the
smallest possible two-sided p-value is 0.0625.
"""
import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = 'lorenz'
SEEDS = [101, 202, 303, 404, 505]
LEVELS = ['obs_100', 'obs_050', 'obs_025', 'obs_010']
METHODS = ['neural_ode', 'neural_distillation', 'direct_sindy']
METRICS = ['traj_l2_mean', 'field_l2', 'f1']
COMPARISONS = [('neural_distillation', 'direct_sindy'),
               ('neural_distillation', 'neural_ode')]


def read_rows(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing Experiment 27 results: {path}\nRun: python scripts/27_evaluate_data_scarcity.py --all')
    with path.open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f'Empty input CSV: {path}')
    required = {'seed', 'level', 'method', 'traj_l2_mean', 'field_l2',
                'f1', 'n_windows', 'failed_windows', 'failure_rate'}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f'Input CSV missing columns: {sorted(missing)}')
    indexed = {}
    for row in rows:
        key = (int(row['seed']), row['level'], row['method'])
        if key in indexed:
            raise ValueError(f'Duplicate record: {key}')
        indexed[key] = row
    expected = set(itertools.product(SEEDS, LEVELS, METHODS))
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise ValueError(f'Expected 60 complete records. Missing={missing}, extra={extra}')
    for key, row in indexed.items():
        n = int(row['n_windows'])
        failed = int(row['failed_windows'])
        rate = float(row['failure_rate'])
        if n <= 0 or failed < 0 or failed > n or not np.isclose(rate, failed / n, atol=1e-10):
            raise ValueError(f'Invalid failure accounting: {key}')
        for metric in METRICS:
            if metric == 'f1' and key[2] == 'neural_ode':
                continue
            if row[metric] == '' or not np.isfinite(float(row[metric])):
                raise ValueError(f'Missing/nonfinite {metric} at {key}')
        if key[2] != 'neural_ode' and not (0 <= float(row['f1']) <= 1):
            raise ValueError(f'Invalid F1 at {key}')
    return indexed


def values(indexed, level, method, metric):
    return np.array([float(indexed[(seed, level, method)][metric]) for seed in SEEDS], dtype=float)


def bootstrap_ci(values_array, rng, iterations, alpha=0.05):
    """Percentile paired-seed bootstrap CI for a sample mean."""
    a = np.asarray(values_array, dtype=float)
    draws = rng.integers(0, len(a), size=(iterations, len(a)))
    estimates = a[draws].mean(axis=1)
    return tuple(float(x) for x in np.quantile(estimates, [alpha / 2, 1 - alpha / 2]))


def exact_sign_flip_p(differences):
    """Exact two-sided paired sign-flip randomization test of mean difference.

    Requires exchangeability/symmetry of signs under the null; not a claim
    of population-level certainty with five independent seeds.
    """
    d = np.asarray(differences, dtype=float)
    observed = abs(float(d.mean()))
    permutations = np.array(list(itertools.product([-1., 1.], repeat=len(d))))
    null_values = abs((permutations * d).mean(axis=1))
    return float(np.mean(null_values >= observed - 1e-12))


def holm_adjust(pvalues):
    """Holm-Bonferroni adjusted p-values, preserving original order."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running = 0.
    for rank, index in enumerate(order):
        running = max(running, min(1., (len(p) - rank) * p[index]))
        adjusted[index] = running
    return adjusted


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows to write: {path}')
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def descriptive(indexed, rng, iterations):
    rows = []
    for level in LEVELS:
        for method in METHODS:
            for metric in METRICS:
                if metric == 'f1' and method == 'neural_ode':
                    continue
                x = values(indexed, level, method, metric)
                low, high = bootstrap_ci(x, rng, iterations)
                rows.append(dict(level=level, observation_percent=int(level.split('_')[1]),
                                 method=method, metric=metric, n_seeds=len(x),
                                 mean=float(x.mean()), sd=float(x.std(ddof=1)),
                                 median=float(np.median(x)), min=float(x.min()), max=float(x.max()),
                                 bootstrap_ci95_low=low, bootstrap_ci95_high=high))
    return rows


def paired(indexed, rng, iterations):
    rows = []
    for left, right in COMPARISONS:
        for metric in METRICS:
            if metric == 'f1' and 'neural_ode' in (left, right):
                continue
            block = []
            for level in LEVELS:
                a = values(indexed, level, left, metric)
                b = values(indexed, level, right, metric)
                delta = a - b  # negative means first method has lower value
                ci_low, ci_high = bootstrap_ci(delta, rng, iterations)
                # For error metrics, improvement > 0 means first method better.
                # For F1, larger is better.
                benefit = -delta if metric != 'f1' else delta
                # Percent change of means, not mean of unstable per-seed ratios.
                percent_change = 100 * float(delta.mean()) / float(b.mean()) if b.mean() != 0 else float('nan')
                row = dict(level=level, observation_percent=int(level.split('_')[1]),
                           metric=metric, method_a=left, method_b=right,
                           n_pairs=len(delta), mean_a=float(a.mean()), mean_b=float(b.mean()),
                           mean_delta_a_minus_b=float(delta.mean()),
                           sd_delta=float(delta.std(ddof=1)),
                           delta_ci95_low=ci_low, delta_ci95_high=ci_high,
                           percent_change_of_means=percent_change,
                           wins_a=int(np.sum(benefit > 1e-12)),
                           ties=int(np.sum(abs(benefit) <= 1e-12)),
                           wins_b=int(np.sum(benefit < -1e-12)),
                           exact_signflip_p=exact_sign_flip_p(delta),
                           holm_p=float('nan'),
                           cohens_dz=float(delta.mean() / delta.std(ddof=1)) if delta.std(ddof=1) > 0 else float('nan'))
                block.append(row)
            adjusted = holm_adjust([r['exact_signflip_p'] for r in block])
            for row, p in zip(block, adjusted):
                row['holm_p'] = float(p)
            rows.extend(block)
    return rows


def plot_metric(indexed, metric, out):
    labels = ['100%', '50%', '25%', '10%']
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for method in METHODS:
        if metric == 'f1' and method == 'neural_ode':
            continue
        matrix = np.array([values(indexed, level, method, metric) for level in LEVELS])
        means = matrix.mean(axis=1)
        sds = matrix.std(axis=1, ddof=1)
        ax.errorbar(np.arange(len(LEVELS)), means, yerr=sds, marker='o', capsize=4,
                    linewidth=1.7, label=method.replace('_', ' ').title())
    ax.set_xticks(np.arange(len(LEVELS)), labels)
    ax.set_xlabel('Available temporal observations')
    ax.set_ylabel({'traj_l2_mean': 'Relative trajectory L2',
                   'field_l2': 'Relative vector-field L2',
                   'f1': 'Structural F1'}[metric])
    ax.set_title('Lorenz — held-out test (mean ± SD, n=5 seeds)')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'figure_{metric}.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_paired(indexed, metric, out):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    positions = np.arange(len(LEVELS))
    for i, seed in enumerate(SEEDS):
        delta = [float(indexed[(seed, level, 'neural_distillation')][metric]) -
                 float(indexed[(seed, level, 'direct_sindy')][metric]) for level in LEVELS]
        ax.plot(positions, delta, marker='o', alpha=0.6, label=f'Seed {seed}')
    mean_d = [np.mean([float(indexed[(seed, level, 'neural_distillation')][metric]) -
                       float(indexed[(seed, level, 'direct_sindy')][metric]) for seed in SEEDS])
              for level in LEVELS]
    ax.plot(positions, mean_d, marker='D', linewidth=2.5, color='black', label='Paired mean')
    ax.axhline(0, linestyle='--', color='gray', linewidth=1)
    ax.set_xticks(positions, ['100%', '50%', '25%', '10%'])
    ax.set_xlabel('Available temporal observations')
    ax.set_ylabel('Distillation − Direct SINDy')
    ax.set_title(f'Paired differences: {metric} (n=5 seeds)')
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'paired_{metric}.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path,
                        default=ROOT / 'results' / 'scarcity' / SYSTEM / 'evaluation' / 'test_metrics_all.csv')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'results' / 'scarcity' / SYSTEM / 'statistics')
    parser.add_argument('--bootstrap', type=int, default=20000)
    parser.add_argument('--random-seed', type=int, default=28042)
    args = parser.parse_args()
    if args.bootstrap < 1000:
        parser.error('--bootstrap must be at least 1000')
    indexed = read_rows(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)
    descriptive_rows = descriptive(indexed, rng, args.bootstrap)
    paired_rows = paired(indexed, rng, args.bootstrap)
    write_csv(args.output / 'descriptive_statistics.csv', descriptive_rows)
    write_csv(args.output / 'paired_comparisons.csv', paired_rows)
    for metric in METRICS:
        plot_metric(indexed, metric, args.output)
        plot_paired(indexed, metric, args.output)
    failure_count = sum(int(row['failed_windows']) for row in indexed.values())
    window_count = sum(int(row['n_windows']) for row in indexed.values())
    protocol = dict(experiment=28, source=str(args.input), n_models=len(indexed),
                    seeds=SEEDS, levels=LEVELS, methods=METHODS,
                    statistical_unit='seed', bootstrap_iterations=args.bootstrap,
                    bootstrap_rng_seed=args.random_seed,
                    bootstrap_ci='percentile 95% paired resampling of seeds',
                    hypothesis_test='exact two-sided paired sign-flip test of mean differences',
                    multiplicity='Holm adjustment across four scarcity levels, separately for each metric and method comparison',
                    inference_caution='Five seeds: exploratory intervals and low-power tests; no causal or broad-population claims',
                    test_usage='post-hoc reporting only; no training, selection, or tuning',
                    total_failed_windows=failure_count, total_evaluated_windows=window_count,
                    failure_policy='Failed windows excluded by Experiment 27; report failure counts and do not compare valid-only errors if failures differ')
    (args.output / 'analysis_protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    print('\n=== EXPERIMENT 28: PAIRED STATISTICAL ANALYSIS ===')
    print(f'Records: {len(indexed)}; independent seeds: {len(SEEDS)}; failures: {failure_count}/{window_count}')
    print('\nTrajectory L2: Neural Distillation vs Direct SINDy (paired differences)')
    for row in paired_rows:
        if row['metric'] == 'traj_l2_mean' and row['method_b'] == 'direct_sindy':
            print(f"{row['level']:>7}  ND={row['mean_a']:.6f} SINDy={row['mean_b']:.6f} "
                  f"delta={row['mean_delta_a_minus_b']:+.6f} "
                  f"CI95=[{row['delta_ci95_low']:+.6f},{row['delta_ci95_high']:+.6f}] "
                  f"wins={row['wins_a']}/{row['n_pairs']} "
                  f"p_exact={row['exact_signflip_p']:.5f} p_Holm={row['holm_p']:.5f}")
    print('\nStructural F1: Neural Distillation vs Direct SINDy')
    for row in paired_rows:
        if row['metric'] == 'f1' and row['method_b'] == 'direct_sindy':
            print(f"{row['level']:>7}  ND={row['mean_a']:.6f} SINDy={row['mean_b']:.6f} "
                  f"delta={row['mean_delta_a_minus_b']:+.6f} "
                  f"wins={row['wins_a']}/{row['n_pairs']} "
                  f"p_exact={row['exact_signflip_p']:.5f} p_Holm={row['holm_p']:.5f}")
    print(f'\nOutputs: {args.output}')
    print('Files: descriptive_statistics.csv, paired_comparisons.csv, analysis_protocol.json, 12 figures (PNG/PDF).')
    print('No training, hyperparameter selection, or test re-evaluation was performed.')


if __name__ == '__main__':
    main()
