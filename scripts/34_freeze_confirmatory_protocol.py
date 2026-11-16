"""Experiment 34: freeze a prospective, auditable multisystem replication plan.

Run from project root: python scripts/34_freeze_confirmatory_protocol.py
Does not train models, open test data, or change prior experiments.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

SYSTEMS = ('vanderpol', 'duffing', 'lotka_volterra')
LEVELS = ('obs_100', 'obs_050', 'obs_025', 'obs_010')
OLD_SEEDS = (101, 202, 303, 404, 505)
NEW_SEEDS = tuple(range(606, 2021, 101))
SCRIPT_NAMES = (
    '29_prepare_multisystem_scarcity.py',
    '30_train_multisystem_scarcity.py',
    '31_discover_multisystem_scarcity.py',
    '32_evaluate_multisystem_scarcity.py',
    '33_statistical_analysis_multisystem.py',
)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check', action='store_true', help='Verify frozen files without changing anything')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    dest = root / 'results' / 'scarcity' / 'confirmatory_34'
    manifest = dest / 'protocol.json'
    rows_path = dest / 'new_seed_manifest.csv'
    files = [root / 'scripts' / s for s in SCRIPT_NAMES] + [root / 'configs' / 'default.yml']
    missing = [str(f.relative_to(root)) for f in files if not f.is_file()]
    if missing:
        raise SystemExit('Missing required files; cannot freeze protocol: ' + ', '.join(missing))
    hashes = {str(f.relative_to(root)).replace('\\', '/'): sha256(f) for f in files}
    if args.check:
        if not manifest.is_file() or not rows_path.is_file():
            raise SystemExit('No frozen protocol found. Run without --check first.')
        old = json.loads(manifest.read_text(encoding='utf-8'))
        if old['source_sha256'] != hashes:
            raise SystemExit('SOURCE DRIFT: one or more scripts/config files changed since protocol freeze.')
        if old['manifest_sha256'] != sha256(rows_path):
            raise SystemExit('MANIFEST DRIFT: new_seed_manifest.csv has changed.')
        print('PASS: protocol, manifest and source hashes match the frozen versions.')
        return
    if manifest.exists() or rows_path.exists():
        raise SystemExit('Protocol already exists; refusing to overwrite. Use --check.')
    # Ensure new runs cannot accidentally overwrite previously generated conditions.
    conflicts = []
    for system in SYSTEMS:
        for level in LEVELS:
            for seed in NEW_SEEDS:
                for base in ('data/scarcity', 'results/scarcity'):
                    path = root / base / system / level / f'seed_{seed}'
                    if path.exists():
                        conflicts.append(str(path.relative_to(root)))
    if conflicts:
        raise SystemExit('Existing new-seed directories; investigate before freezing:\n' + '\n'.join(conflicts[:20]))
    dest.mkdir(parents=True, exist_ok=True)
    with rows_path.open('x', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['system', 'seed', 'level', 'phase', 'status'])
        for system in SYSTEMS:
            for seed in NEW_SEEDS:
                for level in LEVELS:
                    w.writerow([system, seed, level, 'new_independent_replication', 'pending'])
    protocol = {
        'experiment': 34,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Prospective replication of multisystem temporal-scarcity comparisons',
        'exploratory_seeds_excluded_from_confirmatory_primary_test': list(OLD_SEEDS),
        'confirmatory_seeds': list(NEW_SEEDS),
        'systems': list(SYSTEMS),
        'observation_levels': list(LEVELS),
        'methods': ['neural_ode', 'neural_distillation', 'direct_sindy'],
        'primary_endpoint': 'mean test short-horizon trajectory relative L2 per independent seed',
        'primary_contrast': 'neural_distillation minus direct_sindy; negative favors distillation',
        'primary_hypotheses': [
            {'system': s, 'level': 'obs_010', 'direction': 'neural_distillation < direct_sindy'}
            for s in SYSTEMS
        ],
        'multiplicity': 'Holm adjustment over three primary system-specific hypotheses',
        'secondary_endpoints': ['field relative L2', 'symbolic structural F1', 'integration failure rate', 'obs_025 trajectory relative L2'],
        'inference_unit': 'seed, not trajectory or window',
        'missing_and_infinite_policy': (
            'Report all integration failures; never silently drop or cap infinities. '
            'Report complete-case finite error comparisons as sensitivity analyses only. '
            'For primary confirmatory inference, use a prespecified failure-aware paired comparison '
            'where a failed method ranks worse than a finite method; if both fail, treat as tied. '
            'Also report finite-only estimates and their denominators.'
        ),
        'design_constraints': [
            'No modification to original 5-seed results',
            'No model selection on test',
            'Use the frozen training/discovery/evaluation protocol for new seeds',
            'Run a new-seed pilot before full replication',
            'Do not treat exploratory and confirmatory seeds as one independent confirmatory dataset',
        ],
        'planned_new_conditions': len(SYSTEMS) * len(NEW_SEEDS) * len(LEVELS),
        'planned_new_neural_ode_models': len(SYSTEMS) * len(NEW_SEEDS) * len(LEVELS),
        'planned_new_symbolic_models': 2 * len(SYSTEMS) * len(NEW_SEEDS) * len(LEVELS),
        'planned_new_test_model_evaluations': 3 * len(SYSTEMS) * len(NEW_SEEDS) * len(LEVELS),
        'source_sha256': hashes,
        'manifest_sha256': sha256(rows_path),
    }
    manifest.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('=== EXPERIMENT 34: CONFIRMATORY PROTOCOL FROZEN ===')
    print('Existing seeds:', OLD_SEEDS)
    print('New seeds:', NEW_SEEDS)
    print('New conditions:', protocol['planned_new_conditions'])
    print('Frozen manifest:', manifest)
    print('Seed manifest:', rows_path)
    print('Next: inspect CLI compatibility of scripts 29-32 before generating new datasets.')

if __name__ == '__main__':
    main()
