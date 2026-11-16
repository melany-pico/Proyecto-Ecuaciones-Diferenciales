#!/usr/bin/env python
"""Experiment 44: paired Neural ODE training for observational noise.

Run:
  python scripts/46_train_multisystem_noise.py --system vanderpol --seed 606 --noise noise_000
  python scripts/46_train_multisystem_noise.py --all

Uses ONLY train.npz and val.npz. Never opens test.npz.
"""
import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dynadistill.models.neuralode import VectorField
from src.dynadistill.utils.common import config

SYSTEMS = ('vanderpol', 'duffing', 'lotka_volterra')
SEEDS = (606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818, 1919, 2020)
LEVELS = ('noise_000', 'noise_001', 'noise_005', 'noise_010', 'noise_020')
WINDOW_DURATION = 0.80
VALIDATION_WINDOWS = 8
LOG_EVERY = 25


def hash_state(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode('utf-8'))
        digest.update(str(value.shape).encode('ascii'))
        digest.update(value.tobytes())
    return digest.hexdigest()


def read_partition(system, level, seed, partition):
    path = ROOT / 'data' / 'robustness_multisystem_44' / system / level / f'seed_{seed}' / f'{partition}.npz'
    if not path.exists():
        raise FileNotFoundError(f'Missing dataset: {path}; run experiment 29 first')
    with np.load(path, allow_pickle=False) as data:
        x = np.asarray(data['x'], dtype=np.float32)
        t = np.asarray(data['t'], dtype=np.float32)
        ids = np.asarray(data['trajectory_ids'], dtype=np.int64)
    if x.ndim != 3 or t.ndim != 1 or x.shape[1] != len(t) or x.shape[2] != 2:
        raise ValueError(f'Invalid data dimensions: {path}, x={x.shape}, t={t.shape}')
    if not np.isfinite(x).all() or not np.isfinite(t).all() or not np.all(np.diff(t) > 0):
        raise ValueError(f'Invalid finite values or time grid: {path}')
    return x, t, ids


def make_windows(t, duration=WINDOW_DURATION):
    """One window per possible start, with end nearest start+duration."""
    windows = []
    for start in range(len(t) - 1):
        if float(t[start]) + duration > float(t[-1]) + 1e-5:
            continue
        end = start + 1 + int(np.argmin(np.abs(t[start + 1:] - (t[start] + duration))))
        windows.append((start, end))
    if not windows:
        raise ValueError('No usable training windows')
    return windows


def common_schedule(seed, epochs, batches, batch_size):
    """Identical trajectory indices and relative window positions across levels."""
    rng = np.random.default_rng(500000 + int(seed))
    traj_indices = rng.integers(0, 16, size=(epochs, batches, batch_size), dtype=np.int64)
    fractions = rng.random((epochs, batches))
    digest = hashlib.sha256(traj_indices.tobytes() + fractions.tobytes()).hexdigest()
    return traj_indices, fractions, digest


def initial_model(system, seed, hidden):
    # Stable distinct initialization per system; paired across observation levels.
    system_offset = SYSTEMS.index(system) * 100000
    init_seed = 700000 + system_offset + int(seed)
    torch.manual_seed(init_seed)
    model = VectorField(dim=2, hidden=hidden)
    return model, init_seed, hash_state(model.state_dict())


def configure_normalization(model, x, epsilon):
    flat = x.reshape(-1, x.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0)
    std = np.maximum(flat.std(axis=0), epsilon)
    with torch.no_grad():
        model.state_mean.copy_(torch.from_numpy(mean.astype(np.float32)))
        model.state_std.copy_(torch.from_numpy(std.astype(np.float32)))


def val_loss(model, x_val, t_val, windows):
    selected = np.unique(np.linspace(0, len(windows) - 1, min(VALIDATION_WINDOWS, len(windows)), dtype=int))
    total, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for idx in selected:
            start, end = windows[int(idx)]
            time = t_val[start:end + 1] - t_val[start]
            x0 = x_val[:, start, :]
            target = x_val[:, start:end + 1, :]
            z0 = (x0 - model.state_mean) / model.state_std
            z_target = (target - model.state_mean) / model.state_std
            predicted = model.trajectory(z0, time)
            loss = torch.mean((predicted - z_target) ** 2)
            if not torch.isfinite(loss):
                return float('inf')
            total += float(loss.item())
            count += 1
    return total / max(count, 1)


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['epoch', 'train_mse', 'val_mse', 'best_val', 'lr'])
        writer.writeheader()
        writer.writerows(rows)


def train_one(system, seed, level, cfg, skip_completed=False):
    output = ROOT / 'results' / 'robustness_multisystem_44' / system / level / f'seed_{seed}'
    checkpoint_path = output / 'neural_ode.pt'
    history_path = output / 'training_history.csv'
    summary_path = output / 'training_summary.json'
    artifacts = (checkpoint_path, history_path, summary_path)
    if any(p.exists() for p in artifacts):
        if not skip_completed:
            raise FileExistsError(f'Refusing to overwrite existing training artifact in {output}')
        if not all(p.is_file() and p.stat().st_size > 0 for p in artifacts):
            raise RuntimeError(f'Incomplete training artifacts; manual review required: {output}')
        try:
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
            with history_path.open('r', encoding='utf-8', newline='') as handle:
                history = list(csv.DictReader(handle))
            expected_init_seed = 700000 + SYSTEMS.index(system) * 100000 + int(seed)
            valid = (
                summary.get('system') == system and summary.get('seed') == seed
                and summary.get('level') == level
                and summary.get('version') == 'robustness_multisystem_44_paired_v1'
                and summary.get('initialization_seed') == expected_init_seed
                and isinstance(summary.get('initial_weights_sha256'), str)
                and len(summary['initial_weights_sha256']) == 64
                and isinstance(summary.get('minibatch_schedule_sha256'), str)
                and len(summary['minibatch_schedule_sha256']) == 64
                and summary.get('training_config') == dict(cfg['training'])
                and isinstance(summary.get('best_epoch'), int)
                and 1 <= summary['best_epoch'] <= len(history)
                and summary.get('epochs_completed') == len(history)
                and len(history) > 0
                and all(row.get('epoch') == str(i) for i, row in enumerate(history, 1))
            )
            if not valid:
                raise ValueError('Summary or history does not match current training protocol')
        except Exception as exc:
            raise RuntimeError(f'Cannot safely skip existing training artifacts in {output}: {exc}') from exc
        print(f'VERIFIED/SKIPPED system={system} seed={seed} level={level}', flush=True)
        return


    x_train_np, t_train_np, ids_train = read_partition(system, level, seed, 'train')
    x_val_np, t_val_np, ids_val = read_partition(system, level, seed, 'val')
    if len(ids_train) != 16 or len(ids_val) != 3 or set(ids_train) & set(ids_val):
        raise ValueError(f'Invalid train/validation partition for {system}/{level}/seed_{seed}')
    if not np.array_equal(t_train_np, t_val_np):
        raise ValueError('Training and validation time grids differ')

    train_cfg = cfg['training']
    epochs = int(train_cfg['epochs'])
    hidden = int(train_cfg['hidden'])
    batch_size = int(train_cfg['batch_trajectories'])
    batches = int(train_cfg.get('batches_per_epoch', 8))
    lr = float(train_cfg['lr'])
    eps = float(train_cfg.get('normalization', {}).get('epsilon', 1e-6))
    sched_cfg = train_cfg.get('scheduler', {})
    stop_cfg = train_cfg.get('early_stopping', {})
    patience = int(stop_cfg.get('patience', 60))
    device = torch.device(train_cfg.get('device', 'cpu'))
    if batch_size < 1 or epochs < 1 or batches < 1:
        raise ValueError('Invalid training hyperparameters')

    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))
    model, init_seed, init_hash = initial_model(system, seed, hidden)
    configure_normalization(model, x_train_np, eps)
    model = model.to(device)

    x_train = torch.from_numpy(x_train_np).to(device)
    x_val = torch.from_numpy(x_val_np).to(device)
    t_train = torch.from_numpy(t_train_np).to(device)
    t_val = torch.from_numpy(t_val_np).to(device)
    windows_train = make_windows(t_train_np)
    windows_val = make_windows(t_val_np)
    trajectory_schedule, window_schedule, schedule_hash = common_schedule(seed, epochs, batches, batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=float(sched_cfg.get('factor', 0.5)),
        patience=int(sched_cfg.get('patience', 20)),
    )

    best_val = float('inf')
    best_epoch = 0
    best_state = None
    rows = []
    stale = 0
    print(f'=== TRAIN system={system} seed={seed} level={level} windows={len(windows_train)} ===', flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.0
        for batch in range(batches):
            ids = torch.as_tensor(trajectory_schedule[epoch - 1, batch], dtype=torch.long, device=device)
            fraction = float(window_schedule[epoch - 1, batch])
            window_idx = min(int(fraction * len(windows_train)), len(windows_train) - 1)
            start, end = windows_train[window_idx]
            times = t_train[start:end + 1] - t_train[start]
            target = x_train[ids, start:end + 1, :]
            z0 = (target[:, 0, :] - model.state_mean) / model.state_std
            z_target = (target - model.state_mean) / model.state_std
            optimizer.zero_grad(set_to_none=True)
            prediction = model.trajectory(z0, times)
            loss = torch.mean((prediction - z_target) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite train loss: {system} seed={seed} level={level} epoch={epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_total += float(loss.item())

        training_mse = train_total / batches
        validation_mse = val_loss(model, x_val, t_val, windows_val)
        if not np.isfinite(validation_mse):
            raise FloatingPointError(f'Non-finite validation loss: {system} seed={seed} level={level} epoch={epoch}')
        scheduler.step(validation_mse)
        if validation_mse < best_val - 1e-10:
            best_val = validation_mse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        current_lr = float(optimizer.param_groups[0]['lr'])
        rows.append({'epoch': epoch, 'train_mse': training_mse, 'val_mse': validation_mse,
                     'best_val': best_val, 'lr': current_lr})
        if epoch == 1 or epoch % LOG_EVERY == 0 or epoch == epochs:
            print(f'epoch={epoch:03d} train={training_mse:.6f} val={validation_mse:.6f} '
                  f'best={best_val:.6f} lr={current_lr:.6g}', flush=True)
        if stale >= patience:
            print(f'EARLY STOP epoch={epoch} best_epoch={best_epoch}', flush=True)
            break

    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'state_dict': best_state,
        'dim': 2,
        'hidden': hidden,
        'system': system,
        'seed': seed,
        'level': level,
        'version': 'robustness_multisystem_44_paired_v1',
        'normalized': True,
        'best_val': best_val,
        'best_epoch': best_epoch,
        'initialization_seed': init_seed,
        'initial_weights_sha256': init_hash,
        'minibatch_schedule_sha256': schedule_hash,
        'window_duration': WINDOW_DURATION,
        'training_config': dict(train_cfg),
    }
    # Write to temporary files first, then commit only when all three artifacts exist.
    temp_checkpoint = output / 'neural_ode.pt.tmp'
    temp_history = output / 'training_history.csv.tmp'
    temp_summary = output / 'training_summary.json.tmp'
    if any(p.exists() for p in (temp_checkpoint, temp_history, temp_summary)):
        raise FileExistsError(f'Incomplete previous write detected in {output}')
    torch.save(checkpoint, temp_checkpoint)
    write_csv(temp_history, rows)
    summary = {k: v for k, v in checkpoint.items() if k != 'state_dict'}
    summary.update({'epochs_completed': len(rows), 'training_windows': len(windows_train),
                    'validation_windows': min(VALIDATION_WINDOWS, len(windows_val)),
                    'train_trajectories': len(ids_train), 'val_trajectories': len(ids_val)})
    summary['protocol_sha256'] = hashlib.sha256((ROOT / 'results' / 'robustness_multisystem_44' / 'protocol.json').read_bytes()).hexdigest()
    temp_summary.write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    temp_checkpoint.rename(checkpoint_path)
    temp_history.rename(history_path)
    temp_summary.rename(summary_path)
    print(f'SAVED system={system} seed={seed} level={level} best_val={best_val:.6f} '
          f'best_epoch={best_epoch} checkpoint={checkpoint_path}', flush=True)


def verify_protocol():
    path = ROOT / 'results' / 'robustness_multisystem_44' / 'protocol.json'
    protocol = json.loads(path.read_text(encoding='utf-8'))
    if protocol.get('experiment') != 44 or tuple(protocol['systems']) != SYSTEMS or tuple(protocol['seeds']) != SEEDS or tuple(protocol['noise_labels']) != LEVELS:
        raise RuntimeError('Experiment 44 protocol does not match training design')
    if protocol.get('data_root') != 'data/robustness_multisystem_44' or protocol.get('results_root') != 'results/robustness_multisystem_44':
        raise RuntimeError('Experiment 44 protocol paths differ from training script')
    for relative, expected in protocol['frozen_source_sha256'].items():
        file = ROOT / relative
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Frozen source hash mismatch: {relative}')
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system', choices=SYSTEMS)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--noise', choices=LEVELS)
    args = parser.parse_args()
    if not args.all and (args.system is None or args.seed is None or args.noise is None):
        parser.error('Use --all or provide --system, --seed and --noise')
    if args.all and any(v is not None for v in (args.system, args.seed, args.noise)):
        parser.error('--all cannot be combined with --system, --seed or --noise')
    verify_protocol()
    cfg = config()
    systems = SYSTEMS if args.all else (args.system,)
    seeds = SEEDS if args.all else (args.seed,)
    for system in systems:
        for seed in seeds:
            for level in (LEVELS if args.all else (args.noise,)):
                train_one(system, seed, level, cfg, skip_completed=args.all)
    print('\n=== EXPERIMENT 44 TRAINING COMPLETE ===', flush=True)
    print('Only train and validation were accessed; test remains sealed.', flush=True)


if __name__ == '__main__':
    main()
