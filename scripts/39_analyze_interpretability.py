
#!/usr/bin/env python3
"""Experiment 39: descriptive interpretability and robustness analysis.

Reads sealed confirmatory model-level test metrics; does not fit, tune, or
re-evaluate any model. All analyses use seed-level observations.
"""
from pathlib import Path
import csv
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'results' / 'scarcity' / 'confirmatory_34' / 'evaluation'
OUTPUT = ROOT / 'results' / 'scarcity' / 'confirmatory_34' / 'interpretability_39'

SYSTEMS = ('vanderpol', 'duffing', 'lotka_volterra')
LEVELS = ('obs_100', 'obs_050', 'obs_025', 'obs_010')
METHODS = ('neural_ode', 'neural_distillation', 'direct_sindy')
SEEDS = (606,707,808,909,1010,1111,1212,1313,1414,
         1515,1616,1717,1818,1919,2020)
METRICS = ('traj_rel_l2', 'field_rel_l2',
           'structural_f1', 'nonzero_terms')


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def summarize(vals):
    valid = [v for v in vals if math.isfinite(v)]
    return dict(
        n_finite=len(valid),
        n_nonfinite=len(vals)-len(valid),
        mean=statistics.mean(valid) if valid else float('nan'),
        median=statistics.median(valid) if valid else float('nan'),
        sd=statistics.stdev(valid) if len(valid)>1 else float('nan'),
        minimum=min(valid) if valid else float('nan'),
        maximum=max(valid) if valid else float('nan')
    )


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(
            f, fieldnames=fields, extrasaction='ignore'
        )
        w.writeheader()
        w.writerows(rows)


def main():
    records = {}

    for system in SYSTEMS:
        path = INPUT / system / 'test_metrics_all.csv'
        if not path.exists():
            raise SystemExit(f'MISSING: {path}')

        with path.open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                key = (
                    row['system'],
                    row['level'],
                    int(row['seed']),
                    row['method']
                )
                if key in records:
                    raise SystemExit(f'DUPLICATE: {key}')
                records[key] = row

    expected = {
        (s, l, seed, m)
        for s in SYSTEMS
        for l in LEVELS
        for seed in SEEDS
        for m in METHODS
    }

    if set(records) != expected:
        raise SystemExit(
            f'INVALID RECORD SET: '
            f'missing={len(expected-set(records))}, '
            f'extra={len(set(records)-expected)}'
        )

    for key, row in records.items():
        w = num(row['windows'])
        failures = num(row['failures'])
        if not (
            w == 40
            and math.isfinite(failures)
            and 0 <= failures <= 40
        ):
            raise SystemExit(
                f'INVALID WINDOW COUNT: {key}: {w}, {failures}'
            )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    summary = []

    for s in SYSTEMS:
        for l in LEVELS:
            for m in METHODS:
                group = [
                    records[s, l, seed, m]
                    for seed in SEEDS
                ]

                out = dict(
                    system=s,
                    level=l,
                    method=m,
                    n_seeds=len(group),
                    total_failed_windows=sum(
                        int(num(r['failures'])) for r in group
                    ),
                    seeds_with_failures=sum(
                        num(r['failures']) > 0 for r in group
                    )
                )

                for metric in METRICS:
                    st = summarize([
                        num(r[metric]) for r in group
                    ])
                    for stat, value in st.items():
                        out[f'{metric}_{stat}'] = value

                summary.append(out)

    write_csv(
        OUTPUT / 'method_summary.csv',
        summary,
        list(summary[0])
    )

    paired = []

    for s in SYSTEMS:
        for l in LEVELS:
            for seed in SEEDS:
                a = records[s, l, seed, 'neural_distillation']
                b = records[s, l, seed, 'direct_sindy']

                out = dict(
                    system=s,
                    level=l,
                    seed=seed,
                    nd_failures=int(num(a['failures'])),
                    sindy_failures=int(num(b['failures']))
                )

                for metric in METRICS:
                    x, y = num(a[metric]), num(b[metric])
                    out[f'nd_{metric}'] = x
                    out[f'sindy_{metric}'] = y
                    out[f'delta_{metric}'] = (
                        x-y if math.isfinite(x)
                        and math.isfinite(y)
                        else float('nan')
                    )

                paired.append(out)

    write_csv(
        OUTPUT / 'paired_by_seed.csv',
        paired,
        list(paired[0])
    )

    comparisons = []

    for s in SYSTEMS:
        for l in LEVELS:
            group = [
                r for r in paired
                if r['system'] == s and r['level'] == l
            ]

            out = dict(
                system=s,
                level=l,
                n_seeds=len(group)
            )

            for metric in METRICS:
                vals = [
                    r[f'delta_{metric}'] for r in group
                ]
                st = summarize(vals)

                out[f'{metric}_finite_pairs'] = st['n_finite']
                out[f'{metric}_mean_delta'] = st['mean']
                out[f'{metric}_median_delta'] = st['median']
                out[f'{metric}_sd_delta'] = st['sd']

                out[f'{metric}_nd_lower_count'] = sum(
                    v < 0 for v in vals if math.isfinite(v)
                )
                out[f'{metric}_nd_higher_count'] = sum(
                    v > 0 for v in vals if math.isfinite(v)
                )
                out[f'{metric}_ties_count'] = sum(
                    v == 0 for v in vals if math.isfinite(v)
                )

            # For F1, higher is better.
            # For field error, lower is better.
            out['nd_f1_higher_count'] = (
                out['structural_f1_nd_higher_count']
            )
            out['nd_field_error_lower_count'] = (
                out['field_rel_l2_nd_lower_count']
            )

            out['nd_f1_and_field_better_count'] = sum(
                math.isfinite(r['delta_structural_f1'])
                and math.isfinite(r['delta_field_rel_l2'])
                and r['delta_structural_f1'] > 0
                and r['delta_field_rel_l2'] < 0
                for r in group
            )

            out['nd_prediction_and_field_better_count'] = sum(
                math.isfinite(r['delta_traj_rel_l2'])
                and math.isfinite(r['delta_field_rel_l2'])
                and r['delta_traj_rel_l2'] < 0
                and r['delta_field_rel_l2'] < 0
                for r in group
            )

            comparisons.append(out)

    write_csv(
        OUTPUT / 'paired_descriptive_summary.csv',
        comparisons,
        list(comparisons[0])
    )

    metadata = {
        'experiment': 39,
        'scope': 'confirmatory 15 seeds only',
        'sources': [
            str(INPUT / s / 'test_metrics_all.csv')
            for s in SYSTEMS
        ],
        'n_models': len(records),
        'n_paired_seed_comparisons': len(paired),
        'analysis': (
            'descriptive only; no new hypothesis testing '
            'or model selection'
        ),
        'f1': (
            'Not applicable for neural_ode; NaN values '
            'are excluded and counted, never replaced with zero'
        ),
        'nonfinite': (
            'Nonfinite values excluded from finite descriptive '
            'statistics and counted explicitly'
        ),
        'nonzero_terms': (
            'A lower count alone does not establish a better equation'
        ),
        'note': (
            'Structural F1 does not assess coefficient accuracy; '
            'field_rel_l2 is a complementary fidelity measure.'
        )
    }

    (OUTPUT / 'metadata.json').write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )

    print(
        '=== EXPERIMENT 39: '
        'INTERPRETABILITY AND ROBUSTNESS ==='
    )
    print(
        f'Validated {len(records)} model rows; '
        f'{len(paired)} paired seed comparisons.'
    )

    for r in comparisons:
        print(
            f"{r['system']:14s} {r['level']} "
            f"F1_delta={r['structural_f1_mean_delta']:+.4f} "
            f"field_delta={r['field_rel_l2_mean_delta']:+.5f} "
            f"terms_delta={r['nonzero_terms_mean_delta']:+.2f} "
            f"ND_field_wins={r['nd_field_error_lower_count']}/15 "
            f"ND_prediction_and_field_wins="
            f"{r['nd_prediction_and_field_better_count']}/15"
        )

    print('SAVED', OUTPUT)


if __name__ == '__main__':
    main()
