"""Supplementary audit of Experiment 34's three prespecified primary hypotheses.

Reads frozen per-seed results from Experiment 38; does not train, evaluate,
change, or overwrite any existing experiment artifact.

Run: python scripts/50_audit_confirmatory_primary.py
Requires: numpy (already in dynadistill).
"""
from pathlib import Path
import csv
import hashlib
import itertools
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results' / 'scarcity' / 'confirmatory_34'
INPUT = BASE / 'statistics' / 'confirmatory_15_per_seed.csv'
PROTOCOL = BASE / 'protocol.json'
OUT = BASE / 'statistics' / 'primary_audit_50'
SYSTEMS = ('vanderpol', 'duffing', 'lotka_volterra')
LEVEL = 'obs_010'
BOOTSTRAPS = 20000


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_left_signflip(d):
    """One-sided H1: mean(ND - SINDy) < 0, exhaustive 2^15 flips."""
    observed = float(np.sum(d))
    n = len(d)
    extreme = 0
    for signs in itertools.product((-1, 1), repeat=n):
        value = sum(v * s for v, s in zip(d, signs))
        if value <= observed + 1e-12 * max(1.0, abs(observed)):
            extreme += 1
    return extreme / (1 << n)


def holm(p_values):
    out = [None] * len(p_values)
    running = 0.0
    for rank, i in enumerate(sorted(range(len(p_values)), key=lambda k: p_values[k])):
        running = max(running, min(1.0, p_values[i] * (len(p_values) - rank)))
        out[i] = running
    return out


def main():
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    assert protocol['experiment'] == 34
    assert tuple(protocol['confirmatory_seeds']) == tuple(range(606, 2021, 101))
    assert tuple(protocol['systems']) == SYSTEMS
    assert len(protocol['primary_hypotheses']) == 3
    for h in protocol['primary_hypotheses']:
        assert h['level'] == LEVEL and h['direction'] == 'neural_distillation < direct_sindy'
    assert {h['system'] for h in protocol['primary_hypotheses']} == set(SYSTEMS)

    with INPUT.open(newline='', encoding='utf-8-sig') as f:
        raw = list(csv.DictReader(f))
    expected = {(s, LEVEL, seed) for s in SYSTEMS for seed in protocol['confirmatory_seeds']}
    indexed = {}
    for r in raw:
        key = (r['system'], r['level'], int(r['seed']))
        if key in indexed:
            raise ValueError(f'Duplicate row: {key}')
        indexed[key] = r
    if len(raw) != 180 or {k for k in indexed if k[1] == LEVEL} != expected:
        raise ValueError('Unexpected number of rows or missing primary comparisons')

    results = []
    for i, system in enumerate(SYSTEMS):
        records = [indexed[(system, LEVEL, seed)] for seed in protocol['confirmatory_seeds']]
        if any(int(r['nd_failures']) or int(r['sindy_failures']) or int(r['finite_pair']) != 1 for r in records):
            raise ValueError(f'Nonfinite/failure pairs in {system}; requires prespecified failure-aware analysis')
        d = np.array([float(r['nd_l2']) - float(r['sindy_l2']) for r in records], dtype=float)
        if not np.isfinite(d).all():
            raise ValueError(f'Nonfinite primary difference: {system}')
        rng = np.random.default_rng(500034 + i)
        means = d[rng.integers(0, len(d), size=(BOOTSTRAPS, len(d)))].mean(axis=1)
        low, high = np.quantile(means, (0.025, 0.975))
        results.append(dict(system=system, level=LEVEL, n_seeds=len(d),
                            mean_delta_nd_minus_sindy=float(d.mean()),
                            ci95_bootstrap_low=float(low), ci95_bootstrap_high=float(high),
                            nd_wins=int(np.sum(d < 0)), sindy_wins=int(np.sum(d > 0)),
                            ties=int(np.sum(d == 0)),
                            p_exact_one_sided=exact_left_signflip(d)))
    adjusted = holm([r['p_exact_one_sided'] for r in results])
    for r, p in zip(results, adjusted):
        r['p_holm_three_primary'] = p
    # Versioned directory, exclusive creation: no historical overwrites.
    OUT.mkdir(parents=True, exist_ok=False)
    with (OUT / 'primary_three_hypotheses.csv').open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    manifest = dict(analysis='Supplementary primary-hypothesis audit, experiment 34',
                    status='Retrospective implementation of frozen directional primary hypotheses',
                    protocol_sha256=sha256(PROTOCOL), input_sha256=sha256(INPUT),
                    hypotheses='Three one-sided paired mean sign-flip tests at obs_010; negative favors ND',
                    multiplicity='Holm across exactly three primary hypotheses',
                    bootstrap='20000 paired-seed resamples; percentile 95% descriptive CIs',
                    note='Original Experiment 38 two-sided 12-comparison analysis remains unchanged.',
                    results=results)
    (OUT / 'audit_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('PRIMARY AUDIT 50: 3 one-sided tests, 15 seeds per system, no test reruns')
    for r in results:
        print(f"{r['system']}: delta={r['mean_delta_nd_minus_sindy']:+.6f}, "
              f"wins_ND={r['nd_wins']}/15, p={r['p_exact_one_sided']:.8f}, "
              f"p_Holm={r['p_holm_three_primary']:.8f}")
    print('SAVED:', OUT)


if __name__ == '__main__':
    main()
