
#!/usr/bin/env python
"""Experiment 29: paired, nested temporal scarcity datasets for three systems.

Usage:
 python scripts/29_prepare_multisystem_scarcity.py --system vanderpol --seed 101
 python scripts/29_prepare_multisystem_scarcity.py --all
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dynadistill.data.systems import initial, simulate

SYSTEMS = ('vanderpol', 'duffing', 'lotka_volterra')
SEEDS = (606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818, 1919, 2020)
LEVELS = {
    'obs_100': 160,
    'obs_050': 80,
    'obs_025': 40,
    'obs_010': 16,
}

N_TRAJ = 24
N_POINTS = 160
T_END = 8.0
NOISE_STD = 0.01


def nested_indices(n_points):
    """Deterministic largest-gap midpoint selection, including both endpoints."""
    chosen = [0, n_points - 1]
    snapshots = {2: np.array(chosen, dtype=np.int64)}

    for count in range(3, n_points + 1):
        ordered = sorted(chosen)
        gaps = [
            (ordered[j + 1] - ordered[j], ordered[j], ordered[j + 1])
            for j in range(len(ordered) - 1)
        ]
        _, left, right = max(gaps, key=lambda g: (g[0], -g[1]))
        middle = (left + right) // 2

        if middle in chosen:
            raise RuntimeError(
                'Duplicate temporal index during nested sampling'
            )

        chosen.append(middle)

        if count in LEVELS.values():
            snapshots[count] = np.asarray(
                sorted(chosen), dtype=np.int64
            )

    return snapshots


def make_dataset(system, seed, overwrite=False):
    out_root = ROOT / 'data' / 'scarcity' / system

    targets = [
        out_root / level / f'seed_{seed}' / f'{part}.npz'
        for level in LEVELS
        for part in ('train', 'val', 'test')
    ]

    meta_file = (
        ROOT / 'results' / 'scarcity' / system
        / f'seed_{seed}' / 'metadata.json'
    )

    existing = [p for p in targets if p.exists()]

    if existing and not overwrite:
        if len(existing) == len(targets) and meta_file.exists():
            print(
                f'SKIPPED system={system} seed={seed}: '
                'all files exist'
            )
            return

        raise FileExistsError(
            f'Partial dataset for {system}, seed {seed}; '
            'use --overwrite after checking files'
        )

    rng = np.random.default_rng(seed)
    t_full = np.linspace(
        0.0, T_END, N_POINTS, dtype=np.float64
    )

    clean = []

    for trajectory_id in range(N_TRAJ):
        x0 = initial(system, rng)
        x = np.asarray(
            simulate(system, t_full, x0),
            dtype=np.float64
        )

        if x.shape[0] != N_POINTS or not np.isfinite(x).all():
            raise RuntimeError(
                f'Invalid simulation: {system}, '
                f'seed={seed}, trajectory={trajectory_id}'
            )

        clean.append(x)

    clean = np.stack(clean, axis=0)
    noisy = clean + rng.normal(
        0.0, NOISE_STD, size=clean.shape
    )

    # Disjoint trajectory partition shared across scarcity levels.
    permutation = rng.permutation(N_TRAJ)

    partitions = {
        'train': permutation[:16],
        'val': permutation[16:19],
        'test': permutation[19:24],
    }

    assert all(
        len(partitions[p]) == n
        for p, n in [
            ('train', 16),
            ('val', 3),
            ('test', 5),
        ]
    )

    assert len(
        set(np.concatenate(list(partitions.values())).tolist())
    ) == N_TRAJ

    indices_by_size = nested_indices(N_POINTS)

    for level, count in LEVELS.items():
        indices = indices_by_size[count]
        times = t_full[indices]

        assert len(indices) == count
        assert indices[0] == 0
        assert indices[-1] == N_POINTS - 1

        for part, ids in partitions.items():
            destination = (
                out_root / level / f'seed_{seed}' / f'{part}.npz'
            )
            destination.parent.mkdir(
                parents=True, exist_ok=True
            )

            np.savez_compressed(
                destination,
                x=noisy[ids][:, indices, :],
                x_clean=clean[ids][:, indices, :],
                t=times,
                trajectory_ids=ids.astype(np.int64),
                temporal_indices=indices,
                noise_std=np.float64(NOISE_STD),
                seed=np.int64(seed),
            )

        print(
            f'CREATED system={system} seed={seed} '
            f'level={level} '
            f'train=(16,{count},{clean.shape[-1]}) '
            f'val=(3,{count},{clean.shape[-1]}) '
            f'test=(5,{count},{clean.shape[-1]})'
        )

    meta_file.parent.mkdir(
        parents=True, exist_ok=True
    )

    metadata = {
        'experiment': 29,
        'system': system,
        'seed': seed,
        'simulation': {
            'trajectories': N_TRAJ,
            'points': N_POINTS,
            't_end': T_END,
            'noise_std': NOISE_STD,
        },
        'levels': LEVELS,
        'partition_ids': {
            key: val.tolist()
            for key, val in partitions.items()
        },
        'temporal_indices': {
            level: indices_by_size[count].tolist()
            for level, count in LEVELS.items()
        },
        'sampling': (
            'nested largest-gap midpoint, endpoints preserved'
        ),
        'test_policy': (
            'test data generated and sealed; '
            'do not use for model selection'
        ),
    }

    meta_file.write_text(
        json.dumps(metadata, indent=2),
        encoding='utf-8'
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        '--system', choices=SYSTEMS
    )
    parser.add_argument(
        '--seed', type=int, choices=SEEDS
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate all three systems and five seeds',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
    )

    args = parser.parse_args()

    if not args.all and (
        args.system is None or args.seed is None
    ):
        parser.error(
            'Use --all or specify both --system and --seed'
        )

    systems = (
        SYSTEMS if args.all else (args.system,)
    )
    seeds = (
        SEEDS if args.all else (args.seed,)
    )

    for system in systems:
        for seed in seeds:
            make_dataset(
                system, seed,
                overwrite=args.overwrite
            )

    print('\n=== EXPERIMENT 29 COMPLETE ===')
    print(
        'Test partitions generated but not used '
        'for model selection.'
    )


if __name__ == '__main__':
    main()
