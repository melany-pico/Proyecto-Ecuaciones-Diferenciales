#!/usr/bin/env python
"""Experiment 33: post-hoc paired multisystem analysis (no test reruns).

Pilot: python scripts/33_statistical_analysis_multisystem.py --system vanderpol
All:   python scripts/33_statistical_analysis_multisystem.py --all

Unit of inference: independent seed (n=5). A failed rollout yields infinite
aggregate error; report failures and finite-pair sensitivity separately.
"""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ('vanderpol', 'duffing', 'lotka_volterra')
SEEDS = (101, 202, 303, 404, 505)
LEVELS = ('obs_100', 'obs_050', 'obs_025', 'obs_010')
METHODS = ('neural_ode', 'neural_distillation', 'direct_sindy')
METRICS = ('traj_rel_l2', 'field_rel_l2', 'structural_f1')
COMPARISONS = (('neural_distillation', 'direct_sindy'),
               ('neural_distillation', 'neural_ode'))


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_data(system):
    path = ROOT / 'results' / 'scarcity' / system / 'evaluation' / 'test_metrics_all.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing experiment 32 output: {path}')
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    required = {'system', 'seed', 'level', 'method', *METRICS,
                'windows', 'failures', 'successful_window_mean'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f'Unexpected CSV columns in {path}')
    data = {}
    for r in rows:
        if r['system'] != system:
            raise ValueError(f'System mismatch: {r}')
        key = (int(r['seed']), r['level'], r['method'])
        if key in data:
            raise ValueError(f'Duplicate record: {key}')
        for col in ('windows', 'failures'):
            r[col] = int(r[col])
        for col in METRICS + ('successful_window_mean',):
            r[col] = float(r[col])
        if r['windows'] != 40 or not 0 <= r['failures'] <= r['windows']:
            raise ValueError(f'Invalid window/failure accounting: {key}')
        if r['failures'] and not np.isinf(r['traj_rel_l2']):
            raise ValueError(f'Failed rollouts must imply infinite model error: {key}')
        if not r['failures'] and not np.isfinite(r['traj_rel_l2']):
            raise ValueError(f'Nonfinite trajectory error without recorded failure: {key}')
        if r['method'] != 'neural_ode' and not (0 <= r['structural_f1'] <= 1):
            raise ValueError(f'Invalid structural F1: {key}')
        data[key] = r
    expected = set(itertools.product(SEEDS, LEVELS, METHODS))
    if set(data) != expected:
        raise ValueError(f'{system}: missing={sorted(expected-set(data))}, extra={sorted(set(data)-expected)}')
    return data


def bootstrap_mean_ci(x, rng, n_boot):
    x = np.asarray(x, dtype=float)
    if not len(x) or not np.isfinite(x).all():
        return float('nan'), float('nan')
    sampled = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return tuple(float(z) for z in np.quantile(sampled, [.025, .975]))


def exact_sign_flip_p(diff):
    """Two-sided exhaustive paired sign-flip test, testing mean difference 0."""
    diff = np.asarray(diff, dtype=float)
    if len(diff) < 2 or not np.isfinite(diff).all():
        return float('nan')
    observed = abs(float(diff.mean()))
    null = np.array([abs(np.mean(diff * signs)) for signs in itertools.product((-1, 1), repeat=len(diff))])
    return float(np.mean(null >= observed - 1e-12))


def holm_adjust(pvals):
    """Holm correction within a predefined family; nan stays nan."""
    result = [float('nan')] * len(pvals)
    valid = sorted((i for i, p in enumerate(pvals) if np.isfinite(p)), key=lambda i: pvals[i])
    running = 0.
    for rank, i in enumerate(valid):
        running = max(running, min(1., float(pvals[i]) * (len(valid) - rank)))
        result[i] = running
    return result


def safe_mean(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()) if len(x) else float('nan')


def analyze(systems, bootstrap):
    rng = np.random.default_rng(330042)
    indexed = {system: read_data(system) for system in systems}
    descriptives, comparisons, failures = [], [], []
    for system, data in indexed.items():
        for level in LEVELS:
            for method in METHODS:
                rs = [data[(seed, level, method)] for seed in SEEDS]
                failure_count = sum(r['failures'] for r in rs)
                failures.append(dict(system=system, level=level, method=method,
                                     models=len(rs), windows=sum(r['windows'] for r in rs),
                                     failures=failure_count,
                                     failure_rate=failure_count/sum(r['windows'] for r in rs),
                                     models_with_failure=sum(r['failures'] > 0 for r in rs)))
                for metric in METRICS:
                    if metric == 'structural_f1' and method == 'neural_ode':
                        continue
                    vals = np.array([r[metric] for r in rs])
                    finite = vals[np.isfinite(vals)]
                    ci = bootstrap_mean_ci(finite, rng, bootstrap)
                    descriptives.append(dict(system=system, level=level, method=method,
                        metric=metric, n_seeds=len(vals), n_finite=len(finite),
                        n_nonfinite=len(vals)-len(finite),
                        mean_all=safe_mean(vals), mean_finite=safe_mean(finite),
                        sd_finite=float(np.std(finite, ddof=1)) if len(finite)>1 else float('nan'),
                        finite_mean_ci_low=ci[0], finite_mean_ci_high=ci[1]))
            for a, b in COMPARISONS:
                for metric in METRICS:
                    if metric == 'structural_f1' and b == 'neural_ode':
                        continue
                    x = np.array([data[(seed, level, a)][metric] for seed in SEEDS])
                    y = np.array([data[(seed, level, b)][metric] for seed in SEEDS])
                    finite = np.isfinite(x) & np.isfinite(y)
                    d = x[finite] - y[finite]
                    ci = bootstrap_mean_ci(d, rng, bootstrap)
                    wins = int(np.sum(x < y))  # finite always beats +inf
                    losses = int(np.sum(x > y))
                    ties = int(np.sum(x == y))
                    comparisons.append(dict(system=system, level=level, metric=metric,
                        method_a=a, method_b=b, n_paired=len(x), n_finite_pairs=int(finite.sum()),
                        n_nonfinite_pairs=int((~finite).sum()),
                        wins_a=wins, wins_b=losses, ties=ties,
                        mean_a=safe_mean(x), mean_b=safe_mean(y),
                        mean_difference_finite=safe_mean(d),
                        difference_ci_low=ci[0], difference_ci_high=ci[1],
                        p_exact_finite=exact_sign_flip_p(d), p_holm=float('nan'),
                        analysis_note='finite-pair sensitivity only' if not finite.all() else 'complete paired analysis'))
    # Correction family: for each metric and method contrast, across ALL selected
    # systems and observation levels. F1 only has ND vs SINDy.
    for metric in METRICS:
        for a, b in COMPARISONS:
            group = [i for i, r in enumerate(comparisons)
                     if r['metric'] == metric and r['method_a'] == a and r['method_b'] == b]
            corrected = holm_adjust([comparisons[i]['p_exact_finite'] for i in group])
            for i, p in zip(group, corrected):
                comparisons[i]['p_holm'] = p
    return descriptives, comparisons, failures


def figures(out, descriptives, systems):
    for system in systems:
        fig, ax = plt.subplots(figsize=(8.5, 5))
        for method in METHODS:
            y = [next(r['mean_all'] for r in descriptives if
                      r['system']==system and r['level']==level and
                      r['method']==method and r['metric']=='traj_rel_l2')
                 for level in LEVELS]
            ax.plot(range(len(LEVELS)), [v if np.isfinite(v) else np.nan for v in y],
                    marker='o', linewidth=2, label=method)
        ax.set_xticks(range(len(LEVELS)), ['100%', '50%', '25%', '10%'])
        ax.set_ylabel('Test trajectory relative L2 (mean of 5 seeds)')
        ax.set_xlabel('Observed time points')
        ax.set_title(f'{system}: predictive error (nonfinite points omitted)')
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        for ext in ('png','pdf'):
            fig.savefig(out / f'{system}_trajectory.{ext}', dpi=180)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--system', choices=SYSTEMS)
    p.add_argument('--all', action='store_true')
    p.add_argument('--bootstrap', type=int, default=20000)
    args = p.parse_args()
    if not args.all and args.system is None:
        p.error('Use --all or --system SYSTEM')
    if args.bootstrap < 1000:
        p.error('--bootstrap must be >= 1000')
    systems = SYSTEMS if args.all else (args.system,)
    desc, comp, failures = analyze(systems, args.bootstrap)
    out = ROOT / 'results' / 'scarcity' / 'multisystem' / 'statistics'
    if not args.all:
        out = out / args.system
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out/'descriptive_statistics.csv', desc)
    write_csv(out/'paired_comparisons.csv', comp)
    write_csv(out/'integration_failures.csv', failures)
    figures(out, desc, systems)
    meta = dict(experiment=33, systems=systems, seeds=SEEDS, levels=LEVELS,
                bootstrap_replicates=args.bootstrap, inferential_unit='independent seed',
                comparison='paired difference A minus B (negative favors A for errors)',
                ci='percentile bootstrap, resampled paired seeds; exploratory with n=5',
                test='two-sided exact paired sign-flip, exhaustive 2^n sign configurations',
                holm_family='each metric x method contrast across all selected systems and levels',
                failure_policy='infinite aggregate trajectory error on any failed window; finite-pair sensitivity is not full-data inference',
                warning='If running a single-system pilot, Holm correction is pilot-only; use --all for final corrected results.')
    (out/'analysis_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print('\n=== EXPERIMENT 33: PAIRED MULTISYSTEM STATISTICS ===')
    print(f'Systems={list(systems)}; models={len(systems)*60}; bootstrap={args.bootstrap}')
    for system in systems:
        print(f'\n[{system}] Neural Distillation vs Direct SINDy: test trajectory L2')
        for level in LEVELS:
            r = next(z for z in comp if z['system']==system and z['level']==level
                     and z['metric']=='traj_rel_l2' and z['method_b']=='direct_sindy')
            print(f"  {level}: ND={r['mean_a']:.6f} SINDy={r['mean_b']:.6f} "
                  f"wins={r['wins_a']}/5 finite_pairs={r['n_finite_pairs']}/5 "
                  f"delta_finite={r['mean_difference_finite']:+.6f} "
                  f"CI95=[{r['difference_ci_low']:+.6f},{r['difference_ci_high']:+.6f}] "
                  f"p={r['p_exact_finite']:.5f} p_Holm={r['p_holm']:.5f}")
        print('  Integration failures:',
              sum(r['failures'] for r in failures if r['system']==system), '/2400')
    print(f'\nSAVED: {out}')
    print('POST-HOC ONLY: no model training, tuning, or test reintegration.')

if __name__ == '__main__':
    main()
