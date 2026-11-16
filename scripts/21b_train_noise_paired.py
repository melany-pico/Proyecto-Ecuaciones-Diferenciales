
"""Experiment 21b: Paired Neural ODE training across noise levels.

For each seed, all noise conditions use:
- Identical initial neural network weights.
- Identical minibatch trajectory indices and starting positions.
- Identical optimizer and training hyperparameters.

Only training/validation observations are accessed.

Examples:
    python scripts/21b_train_noise_paired.py --seed 101 --noise noise_000
    python scripts/21b_train_noise_paired.py --seed 101
    python scripts/21b_train_noise_paired.py --all
"""

import sys
import json
import copy
import random
import argparse
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from dynadistill.utils.common import ROOT, config
from dynadistill.models.neuralode import VectorField


SEEDS = [101, 202, 303, 404, 505]
NOISES = [
    "noise_000",
    "noise_001",
    "noise_005",
    "noise_010",
    "noise_020",
]

VAL_HORIZON = 9
VAL_WINDOWS = 8
PROTOCOL = "paired_v1"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(system, noise, seed, split):
    path = (
        ROOT / "data" / "robustness"
        / system / noise / f"seed_{seed}"
        / f"{split}.npz"
    )

    with np.load(path) as data:
        x = np.asarray(data["x"], dtype=np.float32)
        t = np.asarray(data["t"], dtype=np.float32)

    if x.ndim != 3 or t.ndim != 1:
        raise ValueError(f"Invalid dimensions: {path}")

    if x.shape[1] != len(t):
        raise ValueError(f"Time dimension mismatch: {path}")

    if not np.isfinite(x).all() or not np.isfinite(t).all():
        raise ValueError(f"Non-finite values: {path}")

    dt = np.diff(t)

    if np.any(dt <= 0):
        raise ValueError(f"Non-increasing time grid: {path}")

    if not np.allclose(dt, dt[0], rtol=1e-5, atol=1e-7):
        raise ValueError(f"Nonuniform time grid: {path}")

    return x, t


def make_validation_segments(x):
    n_traj, n_times, dim = x.shape

    starts = np.unique(
        np.linspace(
            0,
            n_times - VAL_HORIZON,
            VAL_WINDOWS,
            dtype=int,
        )
    )

    return np.asarray(
        [
            x[i, start:start + VAL_HORIZON]
            for i in range(n_traj)
            for start in starts
        ],
        dtype=np.float32,
    )


def prepare_schedule(seed, n_traj, n_times, cfg):
    """Generate minibatch indices independently of noise."""

    training = cfg["training"]

    epochs = int(training["epochs"])
    batches = int(training["batches_per_epoch"])
    batch_size = min(
        int(training["batch_trajectories"]),
        n_traj,
    )
    segment_length = int(training["segment_length"])

    if segment_length < 2 or segment_length > n_times:
        raise ValueError("Invalid training segment length")

    rng = np.random.default_rng(seed * 1000 + 17)

    trajectory_indices = np.empty(
        (epochs, batches, batch_size),
        dtype=np.int64,
    )

    start_indices = np.empty_like(trajectory_indices)

    for epoch in range(epochs):
        for batch in range(batches):
            trajectory_indices[epoch, batch] = rng.choice(
                n_traj,
                size=batch_size,
                replace=False,
            )

            start_indices[epoch, batch] = rng.integers(
                0,
                n_times - segment_length + 1,
                size=batch_size,
            )

    return trajectory_indices, start_indices


def digest_arrays(*arrays):
    digest = hashlib.sha256()

    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.tobytes())

    return digest.hexdigest()


def train_condition(system, noise, seed, cfg, overwrite=False):
    output = (
        ROOT / "results" / "robustness_paired"
        / system / noise / f"seed_{seed}"
    )

    output.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output / "neural_ode.pt"
    history_path = output / "neural_ode_history.json"

    if checkpoint_path.exists() and not overwrite:
        print(
            f"\nSKIPPED {noise} seed={seed}: "
            "checkpoint already exists"
        )
        return None

    train_x, train_t = load_split(
        system, noise, seed, "train"
    )

    val_x, val_t = load_split(
        system, noise, seed, "val"
    )

    if not np.allclose(train_t, val_t):
        raise ValueError("Train and validation grids differ")

    training = cfg["training"]

    epochs = int(training["epochs"])
    hidden = int(training["hidden"])
    lr = float(training["lr"])
    batches = int(training["batches_per_epoch"])
    segment_length = int(training["segment_length"])

    device = torch.device(training.get("device", "cpu"))

    if device.type != "cpu":
        raise ValueError(
            "Paired experiment currently requires CPU "
            "for the intended reproducibility protocol"
        )

    # Same initialization seed across noise levels.
    initialization_seed = seed * 1000 + 11
    seed_everything(initialization_seed)

    # Training normalization only.
    mean = train_x.mean(axis=(0, 1))
    std = train_x.std(axis=(0, 1))

    epsilon = float(
        training.get("normalization", {}).get(
            "epsilon", 1e-6
        )
    )
    std = np.maximum(std, epsilon)

    train_z = (train_x - mean) / std
    val_z = (val_x - mean) / std

    train_tensor = torch.tensor(
        train_z,
        dtype=torch.float32,
        device=device,
    )

    val_segments = torch.tensor(
        make_validation_segments(val_z),
        dtype=torch.float32,
        device=device,
    )

    dt = float(np.mean(np.diff(train_t)))

    train_times = (
        torch.arange(
            segment_length,
            dtype=torch.float32,
            device=device,
        ) * dt
    )

    val_times = (
        torch.arange(
            VAL_HORIZON,
            dtype=torch.float32,
            device=device,
        ) * dt
    )

    model = VectorField(
        dim=train_x.shape[-1],
        hidden=hidden,
        mean=mean,
        std=std,
    ).to(device)

    # Hash trainable weights only. Normalization buffers
    # differ slightly across noise conditions by design.
    initial_weights = [
        parameter.detach().cpu().numpy().copy()
        for parameter in model.parameters()
    ]
    initial_hash = digest_arrays(*initial_weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    scheduler_cfg = training.get("scheduler", {})

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(scheduler_cfg.get("factor", 0.5)),
        patience=int(scheduler_cfg.get("patience", 20)),
    )

    patience = int(
        training.get("early_stopping", {}).get(
            "patience", 60
        )
    )

    trajectory_schedule, start_schedule = prepare_schedule(
        seed,
        train_x.shape[0],
        train_x.shape[1],
        cfg,
    )

    schedule_hash = digest_arrays(
        trajectory_schedule,
        start_schedule,
    )

    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []

    print(
        f"\n=== PAIRED NEURAL ODE TRAINING ===\n"
        f"noise={noise} seed={seed}\n"
        f"train={train_x.shape} val={val_x.shape}\n"
        f"epochs={epochs} hidden={hidden} "
        f"batches={batches}\n"
        f"initial_weights_sha256={initial_hash[:16]}\n"
        f"minibatch_schedule_sha256={schedule_hash[:16]}"
    )

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for batch in range(batches):
            indices = trajectory_schedule[epoch, batch]
            starts = start_schedule[epoch, batch]

            segments = torch.stack(
                [
                    train_tensor[
                        int(i),
                        int(start):int(start) + segment_length,
                        :,
                    ]
                    for i, start in zip(indices, starts)
                ],
                dim=0,
            )

            optimizer.zero_grad(set_to_none=True)

            prediction = model.trajectory(
                segments[:, 0, :],
                train_times,
            )

            loss = torch.mean(
                (prediction - segments) ** 2
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss at "
                    f"seed={seed}, noise={noise}, "
                    f"epoch={epoch + 1}"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            train_losses.append(float(loss.item()))

        model.eval()

        with torch.no_grad():
            val_prediction = model.trajectory(
                val_segments[:, 0, :],
                val_times,
            )

            val_loss = float(
                torch.mean(
                    (val_prediction - val_segments) ** 2
                ).item()
            )

        if not np.isfinite(val_loss):
            raise RuntimeError(
                f"Non-finite validation loss: {noise}, {seed}"
            )

        scheduler.step(val_loss)

        current_epoch = epoch + 1
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_loss = float(np.mean(train_losses))

        history.append({
            "epoch": current_epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr,
        })

        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_epoch = current_epoch
            stale_epochs = 0

            torch.save({
                "state_dict": copy.deepcopy(
                    model.state_dict()
                ),
                "dim": int(train_x.shape[-1]),
                "hidden": hidden,
                "system": system,
                "seed": seed,
                "noise_label": noise,
                "version": PROTOCOL,
                "normalized": True,
                "best_val": best_val,
                "best_epoch": best_epoch,
                "initialization_seed": initialization_seed,
                "initial_weights_sha256": initial_hash,
                "minibatch_schedule_sha256": schedule_hash,
                "training_config": training,
            }, checkpoint_path)

        else:
            stale_epochs += 1

        if (
            current_epoch == 1
            or current_epoch % 25 == 0
            or current_epoch == epochs
        ):
            print(
                f"epoch={current_epoch:3d} "
                f"train={train_loss:.6f} "
                f"val={val_loss:.6f} "
                f"best={best_val:.6f} "
                f"lr={current_lr:.6g}"
            )

        if stale_epochs >= patience:
            print(
                f"Early stopping: epoch {current_epoch}"
            )
            break

    history_path.write_text(
        json.dumps({
            "protocol": PROTOCOL,
            "system": system,
            "noise_label": noise,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val": best_val,
            "initial_weights_sha256": initial_hash,
            "minibatch_schedule_sha256": schedule_hash,
            "history": history,
        }, indent=2),
        encoding="utf-8",
    )

    print(
        f"BEST noise={noise} seed={seed} "
        f"epoch={best_epoch} val={best_val:.6f}"
    )
    print(f"Checkpoint: {checkpoint_path}")

    return {
        "seed": seed,
        "noise": noise,
        "best_epoch": best_epoch,
        "best_val": best_val,
        "initial_hash": initial_hash,
        "schedule_hash": schedule_hash,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=101,
    )

    parser.add_argument(
        "--noise",
        choices=NOISES,
        default=None,
    )

    parser.add_argument(
        "--all",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    cfg = config()
    system = cfg["system"]

    if args.all:
        selected_seeds = SEEDS
        selected_noises = NOISES
    else:
        if args.seed not in SEEDS:
            parser.error(f"Choose a seed from {SEEDS}")

        selected_seeds = [args.seed]
        selected_noises = (
            [args.noise]
            if args.noise is not None
            else NOISES
        )

    completed = []

    for seed in selected_seeds:
        for noise in selected_noises:
            result = train_condition(
                system,
                noise,
                seed,
                cfg,
                overwrite=args.overwrite,
            )

            if result is not None:
                completed.append(result)

    print("\n=== EXPERIMENT 21b COMPLETE ===")

    for result in completed:
        print(
            f"seed={result['seed']} "
            f"noise={result['noise']} "
            f"best_epoch={result['best_epoch']} "
            f"best_val={result['best_val']:.6f}"
        )

    print(
        "\nAll models used paired initialization "
        "and minibatch schedules within each seed."
    )
    print("Test data was not accessed.")


if __name__ == "__main__":
    main()
