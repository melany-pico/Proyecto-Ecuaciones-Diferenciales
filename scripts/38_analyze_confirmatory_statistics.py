"""Experiment 38: preregistered confirmatory paired analysis (15 new seeds).
Run: python scripts/38_analyze_confirmatory_statistics.py
Requires numpy and scipy; reads ONLY saved test metrics, never refits models.
"""
from pathlib import Path
import csv
import json
import math
import itertools
import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results' / 'scarcity' / 'confirmatory_34'
OUT = BASE / 'statistics'
SYSTEMS = ('vanderpol', 'duffing', 'lotka_volterra')
LEVELS = ('obs_100', 'obs_050', 'obs_025', 'obs_010')
SEEDS = (606,707,808,909,1010,1111,1212,1313,1414,1515,1616,1717,1818,1919,2020)
METHODS = ('neural_ode', 'neural_distillation', 'direct_sindy')
BOOTSTRAPS = 20000
BOOT_SEED = 380042

def read_rows(system):
    path = BASE / 'evaluation' / system / 'test_metrics_all.csv'
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 180:
        raise ValueError(f'{path}: expected 180 rows, got {len(rows)}')
    data = {}
    for r in rows:
        key = (r['system'], int(r['seed']), r['level'], r['method'])
        if key in data: raise ValueError(f'duplicate row {key}')
        if key[0] != system or key[1] not in SEEDS or key[2] not in LEVELS or key[3] not in METHODS:
            raise ValueError(f'unexpected row {key}')
        val = float(r['traj_rel_l2'])
        failures = int(r['failures'])
        windows = int(r['windows'])
        if math.isnan(val) or val < 0 or failures < 0 or windows != 40 or failures > windows:
            raise ValueError(f'invalid metric/failures/windows: {key}, {val}, {failures}, {windows}')
        if failures and math.isfinite(val):
            raise ValueError(f'failure not represented as infinite L2: {key}')
        data[key] = dict(l2=val, failures=failures, windows=windows,
                         field_l2=float(r['field_rel_l2']),
                         structural_f1=float(r['structural_f1']))
    expected = {(system,s,l,m) for s in SEEDS for l in LEVELS for m in METHODS}
    if set(data) != expected: raise ValueError(f'missing keys: {expected - set(data)}')
    return data

def exact_signflip_p(deltas):
    """Exact two-sided randomization p for the mean signed paired difference."""
    d = np.asarray(deltas, dtype=np.float64)
    if not len(d): return float('nan')
    observed = abs(float(np.sum(d)))
    # Enumerate all 2^n sign assignments, including zero differences.
    total = 1 << len(d)
    extreme = 0
    for mask in range(total):
        signed_sum = sum(v if (mask >> j) & 1 else -v for j,v in enumerate(d))
        if abs(signed_sum) >= observed - 1e-12 * max(1.0, observed):
            extreme += 1
    return extreme / total

def bootstrap_ci(deltas, rng):
    d = np.asarray(deltas, dtype=np.float64)
    if not len(d): return (float('nan'), float('nan'))
    means = np.empty(BOOTSTRAPS)
    for i in range(BOOTSTRAPS):
        means[i] = np.mean(d[rng.integers(0, len(d), size=len(d))])
    return tuple(float(x) for x in np.quantile(means, [.025, .975]))

def write_csv(path, rows):
    if not rows: raise ValueError('no rows')
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    all_data = {}
    for system in SYSTEMS: all_data.update(read_rows(system))
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(BOOT_SEED)
    comparisons = []
    per_seed = []
    for system in SYSTEMS:
        for level in LEVELS:
            finite_deltas = []
            wins = ties = losses = 0
            failures_nd = failures_sindy = 0
            n_nonfinite_pairs = 0
            for seed in SEEDS:
                a = all_data[(system,seed,level,'neural_distillation')]
                b = all_data[(system,seed,level,'direct_sindy')]
                x,y = a['l2'],b['l2']
                failures_nd += a['failures']
                failures_sindy += b['failures']
                if x < y: wins += 1
                elif x > y: losses += 1
                else: ties += 1
                finite = math.isfinite(x) and math.isfinite(y)
                if finite: finite_deltas.append(x-y)
                else: n_nonfinite_pairs += 1
                per_seed.append(dict(system=system,level=level,seed=seed,nd_l2=x,sindy_l2=y,
                                     difference_nd_minus_sindy=(x-y if finite else float('nan')),
                                     nd_failures=a['failures'],sindy_failures=b['failures'],finite_pair=int(finite)))
            lo,hi = bootstrap_ci(finite_deltas, rng)
            # The finite-pair sign flip is NOT an unconditional confirmatory test if failures occurred.
            p = exact_signflip_p(finite_deltas) if n_nonfinite_pairs == 0 else float('nan')
            n_eff = wins + losses
            p_wins = binomtest(wins, n_eff, .5).pvalue if n_eff else 1.0
            comparisons.append(dict(system=system,level=level,n_seeds=len(SEEDS),n_finite_pairs=len(finite_deltas),
                nonfinite_pairs=n_nonfinite_pairs,nd_wins=wins,sindy_wins=losses,ties=ties,
                nd_failure_windows=failures_nd,sindy_failure_windows=failures_sindy,
                mean_paired_delta_finite=float(np.mean(finite_deltas)) if finite_deltas else float('nan'),
                ci95_finite_low=lo,ci95_finite_high=hi,
                exact_signflip_p=p,win_loss_sign_test_p=p_wins,
                inference_note=('complete finite pairs' if n_nonfinite_pairs==0 else
                    'nonfinite pairs: sign-flip and CI on finite pairs are descriptive only; win/loss sign test includes all seeds')))
    # Holm family correction only when all 12 exact mean tests are defined.
    # If any nonfinite comparisons occur, preserve valid per-comparison tests and mark
    # the full 12-test mean-based family as not estimable, rather than silently dropping hypotheses.
    finite_ps = [(i,r['exact_signflip_p']) for i,r in enumerate(comparisons) if math.isfinite(r['exact_signflip_p'])]
    if len(finite_ps)==len(comparisons):
        ordered=sorted(finite_ps,key=lambda t:t[1]); prev=0.0
        for rank,(idx,p) in enumerate(ordered):
            prev=max(prev,min(1.0,(len(ordered)-rank)*p))
            comparisons[idx]['holm_p_exact_signflip']=prev
    else:
        for r in comparisons: r['holm_p_exact_signflip']=float('nan')
    # Prespecified all-seed failure-robust win/loss sign test (12-comparison Holm).
    ordered=sorted(enumerate(comparisons),key=lambda t:t[1]['win_loss_sign_test_p'])
    prev=0.0
    for rank,(idx,r) in enumerate(ordered):
        prev=max(prev,min(1.0,(len(ordered)-rank)*r['win_loss_sign_test_p']))
        comparisons[idx]['holm_p_win_loss_sign_test']=prev
    write_csv(OUT/'confirmatory_15_paired.csv', comparisons)
    write_csv(OUT/'confirmatory_15_per_seed.csv', per_seed)
    report = {
      'design': '15 independent new seeds; 3 systems; 4 scarcity levels; ND minus direct SINDy',
      'analysis': 'exact two-sided sign-flip of paired mean on all-finite groups; 20,000 paired bootstrap resamples; Holm over 12 hypotheses; failure-robust win/loss sign test on all 15 seeds',
      'caution': 'If nonfinite pairs occur, mean-difference tests on finite subset are descriptive and not confirmatory. Sign test uses all seeds but tests wins rather than mean error.',
      'seed_count': len(SEEDS), 'comparisons': len(comparisons),
      'nonfinite_comparisons': sum(r['nonfinite_pairs']>0 for r in comparisons),
      'total_failed_windows_nd': sum(r['nd_failure_windows'] for r in comparisons),
      'total_failed_windows_sindy': sum(r['sindy_failure_windows'] for r in comparisons),
      'files': ['confirmatory_15_paired.csv','confirmatory_15_per_seed.csv']
    }
    (OUT/'analysis_metadata.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('=== EXPERIMENT 38: CONFIRMATORY STATISTICS ===')
    print('All 540 model rows validated. 12 paired comparisons. 15 seeds each.')
    for r in comparisons:
        print(f"{r['system']:14} {r['level']} wins_ND={r['nd_wins']:2d}/15 "
              f"finite={r['n_finite_pairs']:2d} delta={r['mean_paired_delta_finite']:+.6f} "
              f"p_signflip={r['exact_signflip_p']:.6g} p_holm_sign={r['holm_p_win_loss_sign_test']:.6g} "
              f"failures_ND/SINDy={r['nd_failure_windows']}/{r['sindy_failure_windows']}")
    print('SAVED', OUT)

if __name__ == '__main__': main()
