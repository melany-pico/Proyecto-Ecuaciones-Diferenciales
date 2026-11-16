#!/usr/bin/env python
"""Experiment 44 post-hoc analysis. Reads saved CSVs; never reruns test.
Run: python scripts/49_analyze_multisystem_noise.py
"""
import csv
import itertools
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results' / 'robustness_multisystem_44'
METHODS = ('neural_ode', 'neural_distillation', 'direct_sindy')
POLICIES = ('clean_start', 'noisy_start')
METRICS = ('traj_rel_l2', 'field_rel_l2', 'structural_f1')

def write_csv(path, rows):
    if not rows: return
    with path.open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def exact_p(d):
    d = np.asarray(d, dtype=float)
    obs = abs(d.mean())
    # 2^15 exhaustive sign flips, no asymptotic approximation.
    masks = np.arange(1 << len(d), dtype=np.uint32)[:, None]
    signs = 1 - 2 * ((masks >> np.arange(len(d), dtype=np.uint32)) & 1).astype(np.int8)
    vals = np.abs((signs @ d) / len(d))
    return float(np.mean(vals >= obs - 1e-12))

def ci(d, seed):
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.mean(d[rng.integers(0, len(d), (20000, len(d)))], axis=1)
    return tuple(float(v) for v in np.quantile(means, [0.025, 0.975]))

def main():
    p = json.loads((BASE / 'protocol.json').read_text(encoding='utf-8'))
    systems, seeds, levels = p['systems'], p['seeds'], p['noise_labels']
    expected = set(itertools.product(systems, seeds, levels, POLICIES, METHODS))
    indexed = {}
    for system in systems:
        path = BASE / 'evaluation' / system / 'test_metrics_all.csv'
        with path.open(encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                key = (r['system'], int(r['seed']), r['level'], r['start_policy'], r['method'])
                if key in indexed: raise ValueError(f'Duplicate: {key}')
                if r['system'] != system: raise ValueError(f'Wrong system: {key}')
                r['failures'] = int(r['failures']); r['windows'] = int(r['windows'])
                for m in METRICS: r[m] = float(r[m])
                if r['windows'] != 40 or r['failures'] != 0 or not np.isfinite(r['traj_rel_l2']):
                    raise ValueError(f'Unexpected failure/nonfinite trajectory: {key}')
                if r['method'] != 'neural_ode' and not (0 <= r['structural_f1'] <= 1):
                    raise ValueError(f'Invalid F1: {key}')
                indexed[key] = r
    if set(indexed) != expected:
        raise ValueError(f'Missing={len(expected-set(indexed))}; extra={len(set(indexed)-expected)}')
    # Cross-policy invariance: same learned vector field and structural support.
    for system, seed, level, method in itertools.product(systems, seeds, levels, METHODS):
        a = indexed[(system, seed, level, 'clean_start', method)]
        b = indexed[(system, seed, level, 'noisy_start', method)]
        for metric in ('field_rel_l2', 'structural_f1'):
            if not (a[metric] == b[metric] or (np.isnan(a[metric]) and np.isnan(b[metric]))):
                raise ValueError(f'Unexpected policy dependence {metric}: {system,seed,level,method}')
        if level == 'noise_000' and a['traj_rel_l2'] != b['traj_rel_l2']:
            raise ValueError(f'Zero noise mismatch: {system,seed,method}')
    desc, contrasts = [], []
    for system, level, policy in itertools.product(systems, levels, POLICIES):
        for method in METHODS:
            records = [indexed[(system, seed, level, policy, method)] for seed in seeds]
            for metric in METRICS:
                if metric == 'structural_f1' and method == 'neural_ode': continue
                vals = np.array([r[metric] for r in records])
                if not np.isfinite(vals).all(): raise ValueError(f'Nonfinite {metric}: {system,level,policy,method}')
                desc.append(dict(system=system, level=level, noise_std=records[0]['noise_std'],
                    start_policy=policy, method=method, metric=metric, n_seeds=len(seeds),
                    mean=float(vals.mean()), sd=float(vals.std(ddof=1)), median=float(np.median(vals)),
                    min=float(vals.min()), max=float(vals.max())))
        for metric in METRICS:
            for a, b in (('neural_distillation', 'direct_sindy'), ('neural_distillation', 'neural_ode')):
                if metric == 'structural_f1' and b == 'neural_ode': continue
                x = np.array([indexed[(system,s,level,policy,a)][metric] for s in seeds])
                y = np.array([indexed[(system,s,level,policy,b)][metric] for s in seeds])
                d = x-y
                low, high = ci(d, 490044 + systems.index(system)*1000 + levels.index(level)*100 + POLICIES.index(policy)*10 + METRICS.index(metric))
                contrasts.append(dict(system=system,level=level,start_policy=policy,metric=metric,
                    method_a=a,method_b=b,n_seeds=len(seeds),mean_a=float(x.mean()),mean_b=float(y.mean()),
                    mean_difference=float(d.mean()),median_difference=float(np.median(d)),
                    ci_95_low=low,ci_95_high=high,wins_a=int(np.sum(d<0)),wins_b=int(np.sum(d>0)),
                    ties=int(np.sum(d==0)),p_exact_two_sided=exact_p(d),p_holm_primary=''))
    primary = [r for r in contrasts if r['level']=='noise_020' and r['start_policy']=='clean_start'
        and r['metric']=='traj_rel_l2' and r['method_a']=='neural_distillation'
        and r['method_b']=='direct_sindy']
    assert len(primary)==3
    ordered = sorted(primary, key=lambda r: r['p_exact_two_sided'])
    running=0.0
    for rank,r in enumerate(ordered):
        running=max(running,min(1.0,(len(ordered)-rank)*r['p_exact_two_sided']))
        r['p_holm_primary']=running
    out = BASE / 'analysis_49'
    out.mkdir(parents=True,exist_ok=True)
    # Never overwrite existing analysis: explicit reruns require a new directory/version.
    for name in ('descriptive.csv','paired_contrasts.csv','primary_hypotheses.csv','analysis_summary.json'):
        if (out/name).exists(): raise FileExistsError(f'Refusing overwrite: {out/name}')
    write_csv(out/'descriptive.csv',desc)
    write_csv(out/'paired_contrasts.csv',contrasts)
    write_csv(out/'primary_hypotheses.csv',primary)
    summary={'experiment':44,'evaluation_rows':len(indexed),'seeds_per_system':len(seeds),
        'primary_family':'three two-sided clean-start ND vs SINDy comparisons at sigma=0.20',
        'primary':primary,'bootstrap':'20000 paired-seed resamples, percentile 95% CI',
        'exact_test':'exhaustive 2^15 paired sign flips of seed-level mean differences',
        'multiple_testing':'Holm over exactly three primary hypotheses',
        'interpretation_note':'Experiment 44 reuses trajectories/seeds from experiment 34; not an independent replication.'}
    (out/'analysis_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print('AUDIT OK: 1350 model-policy evaluations; no integration failures; zero-noise policies match.')
    print('PRIMARY: clean_start, noise_020, Neural Distillation minus Direct SINDy')
    for r in primary:
        print(f"{r['system']}: mean_diff={r['mean_difference']:.6f} CI95=[{r['ci_95_low']:.6f},{r['ci_95_high']:.6f}] wins_ND={r['wins_a']}/15 p_exact={r['p_exact_two_sided']:.6g} p_Holm={r['p_holm_primary']:.6g}")
    print('SAVED:',out)
    print('Only existing test CSVs read; no model refitting or test re-evaluation.')

if __name__=='__main__': main()
