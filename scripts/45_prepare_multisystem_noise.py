#!/usr/bin/env python
"""Experiment 45: generate paired observational-noise datasets for frozen protocol 44.

Pilot:
  python scripts/45_prepare_multisystem_noise.py --system vanderpol --seed 606 --noise noise_000
Then other noise levels, followed by --all once the pilot has been inspected.
No original file is modified. Existing complete conditions are verified and skipped.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / 'results/robustness_multisystem_44/protocol.json'
SPLITS = ('train', 'val', 'test')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rng_for(system, seed, label, split):
    token = f'exp44|gaussian|{system}|{seed}|{label}|{split}'.encode('ascii')
    entropy = np.frombuffer(hashlib.sha256(token).digest(), dtype='<u4').tolist()
    return np.random.default_rng(np.random.SeedSequence(entropy))


def prepare(protocol, system, seed, label, sigma):
    src = ROOT / protocol['source_data_root'] / system / 'obs_100' / f'seed_{seed}'
    dst = ROOT / protocol['data_root'] / system / label / f'seed_{seed}'
    outputs = [dst / f'{split}.npz' for split in SPLITS]
    manifest = dst / 'manifest.json'
    existing = [p for p in outputs + [manifest] if p.exists()]
    if existing:
        if len(existing) != 4:
            raise FileExistsError(f'Partial condition, refusing to overwrite: {dst}')
        info = json.loads(manifest.read_text(encoding='utf-8'))
        if (info.get('system'), info.get('seed'), info.get('noise_label'), info.get('noise_std')) != (system, seed, label, sigma):
            raise ValueError(f'Manifest mismatch: {manifest}')
        if info.get('protocol_sha256') != digest(PROTOCOL_PATH):
            raise ValueError(f'Protocol hash mismatch: {manifest}')
        for split in SPLITS:
            if info['output_sha256'][split] != digest(dst / f'{split}.npz'):
                raise ValueError(f'Output hash mismatch: {dst / f"{split}.npz"}')
            if info['source_sha256'][split] != digest(src / f'{split}.npz'):
                raise ValueError(f'Source hash mismatch: {src / f"{split}.npz"}')
        print(f'VERIFIED/SKIPPED {system} seed={seed} {label}', flush=True)
        return

    prepared = {}
    source_hashes = {}
    ids_by_split = {}
    for split, ntraj in (('train', 16), ('val', 3), ('test', 5)):
        source = src / f'{split}.npz'
        if not source.is_file():
            raise FileNotFoundError(source)
        source_hashes[split] = digest(source)
        with np.load(source, allow_pickle=False) as data:
            clean = np.asarray(data['x_clean'], dtype=np.float64)
            t = np.asarray(data['t'], dtype=np.float64)
            ids = np.asarray(data['trajectory_ids'], dtype=np.int64)
            indices = np.asarray(data['temporal_indices'], dtype=np.int64)
            stored_seed = int(data['seed'])
        if (clean.shape != (ntraj, 160, 2) or t.shape != (160,) or
                ids.shape != (ntraj,) or indices.shape != (160,) or
                stored_seed != seed or len(set(ids.tolist())) != ntraj or
                not np.array_equal(indices, np.arange(160)) or
                not np.isfinite(clean).all() or not np.isfinite(t).all() or
                not np.all(np.diff(t) > 0)):
            raise ValueError(f'Invalid frozen source: {source}')
        ids_by_split[split] = set(ids.tolist())
        noise = (np.zeros_like(clean) if sigma == 0 else
                 rng_for(system, seed, label, split).normal(0, sigma, size=clean.shape))
        observed = clean + noise
        if not np.isfinite(observed).all():
            raise ValueError(f'Nonfinite generated observations: {system}/{seed}/{label}/{split}')
        prepared[split] = dict(x=observed, x_clean=clean, t=t,
                               trajectory_ids=ids, temporal_indices=indices,
                               noise_std=np.float64(sigma), seed=np.int64(seed))
    if any(ids_by_split[a] & ids_by_split[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError(f'Overlapping trajectory partitions: {system}/{seed}')
    if not np.array_equal(prepared['train']['t'], prepared['val']['t']) or not np.array_equal(prepared['train']['t'], prepared['test']['t']):
        raise ValueError('Split time grids do not match')

    dst.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        with (dst / f'{split}.npz').open('xb') as handle:
            np.savez_compressed(handle, **prepared[split])
    output_hashes = {split: digest(dst / f'{split}.npz') for split in SPLITS}
    info = dict(experiment=44, system=system, seed=seed, noise_label=label,
                noise_std=sigma, protocol_sha256=digest(PROTOCOL_PATH),
                source_sha256=source_hashes, output_sha256=output_hashes,
                rng='numpy.default_rng(SeedSequence(SHA256-derived uint32 words))',
                rng_token_format='exp44|gaussian|system|seed|noise_label|split',
                noise_generated_from='x_clean',
                test_policy='Generated and sealed; never use test for training or model selection',
                trajectory_ids={split: sorted(ids_by_split[split]) for split in SPLITS})
    with manifest.open('x', encoding='utf-8') as handle:
        json.dump(info, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(f'CREATED {system} seed={seed} {label} sigma={sigma:g} | train=16 val=3 test=5', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--noise')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    if not PROTOCOL_PATH.is_file():
        raise FileNotFoundError(PROTOCOL_PATH)
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding='utf-8'))
    if protocol.get('experiment') != 44:
        raise ValueError('Not an experiment 44 protocol')
    systems, seeds = protocol['systems'], protocol['seeds']
    labels = protocol['noise_labels']
    sigmas = protocol['noise_standard_deviations_absolute']
    if len(labels) != len(sigmas) or len(set(labels)) != len(labels):
        raise ValueError('Invalid noise mapping in protocol')
    if args.all:
        if any(x is not None for x in (args.system, args.seed, args.noise)):
            parser.error('Use --all alone or specify --system, --seed and --noise')
        conditions = ((s, seed, label, sigma) for s in systems for seed in seeds for label, sigma in zip(labels, sigmas))
    else:
        if args.system not in systems or args.seed not in seeds or args.noise not in labels:
            parser.error(f'Provide --system {systems}, --seed from frozen protocol and --noise {labels}')
        conditions = [(args.system, args.seed, args.noise, sigmas[labels.index(args.noise)])]
    for system, seed, label, sigma in conditions:
        prepare(protocol, system, seed, label, float(sigma))
    print('DATA PREPARATION COMPLETE. No model training or test evaluation performed.', flush=True)


if __name__ == '__main__':
    main()
