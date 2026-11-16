
"""Experiment 21: Train Neural ODEs for noise robustness.

Uses only training and validation observations.
Never reads test data.

Examples:
    python scripts/21_train_noise_robustness.py --seed 101
    python scripts/21_train_noise_robustness.py --seed 101 --noise noise_005
    python scripts/21_train_noise_robustness.py --all
"""

import sys
import json
import random
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from dynadistill.utils.common import ROOT, config
from dynadistill.models.neuralode import VectorField


SEEDS = [101, 202, 303, 404, 505]
NOISE_LABELS = [
    "noise_000",
    "noise_001",
    "noise_005",
    "noise_010",
    "noise_020",
]

VAL_HORIZON = 9
VAL_SEGMENTS_PER_TRAJECTORY = 8


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_data(system, noise, seed, split):
    path = (
        ROOT / "data" / "robustness"
        / system / noise / f"seed_{seed}"
        / f"{split}.npz"
    )

    with np.load(path) as data:
        x = np.asarray(data["x"], dtype=np.float32)
        t = np.asarray(data["t"], dtype=np.float32)

    if x.ndim != 3 or t.ndim != 1:
        raise ValueError(f"Invalid dataset: {path}")

    if x.shape[1] != len(t):
        raise ValueError(f"Time mismatch: {path}")

    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite data: {path}")

    if np.any(np.diff(t) <= 0):
        raise ValueError(f"Invalid time grid: {path}")

    return x, t


def build_segments(x, length, starts=None):
    n_trajectories, n_points, dim = x.shape

    if length > n_points:
        raise ValueError("Segment length exceeds trajectory")

    if starts is None:
        starts = np.arange(n_points - length + 1)

    segments = np.asarray(
        [
            x[trajectory, start:start + length, :]
            for trajectory in range(n_trajectories)
            for start in starts
        ],
        dtype=np.float32,
    )

    return segments


def validation_segments(x):
    starts = np.unique(
        np.linspace(
            0,
            x.shape[1] - VAL_HORIZON,
            VAL_SEGMENTS_PER_TRAJECTORY,
            dtype=int,
        )
    )
    return build_segments(x, VAL_HORIZON, starts)


def train_one(system, noise, seed, cfg, overwrite=False):
    output = (
        ROOT / "results" / "robustness"
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
        return

    # Different random initialization for each condition,
    # but fully deterministic across repeated executions.
    noise_id = NOISE_LABELS.index(noise)
    training_seed = seed * 100 + noise_id
    set_seed(training_seed)

    train_x, train_t = read_data(
        system, noise, seed, "train"
    )
    val_x, val_t = read_data(
        system, noise, seed, "val"
    )

    if not np.allclose(train_t, val_t):
        raise ValueError("Train/validation time grids differ")

    training_cfg = cfg["training"]

    epochs = int(training_cfg["epochs"])
    hidden = int(training_cfg["hidden"])
    lr = float(training_cfg["lr"])
    batch_trajectories = int(
        training_cfg["batch_trajectories"]
    )
    segment_length = int(
        training_cfg["segment_length"]
    )
    batches_per_epoch = int(
        training_cfg["batches_per_epoch"]
    )

    if segment_length < 2:
        raise ValueError("segment_length must be >= 2")

    device = torch.device(
        training_cfg.get("device", "cpu")
    )

    # Train-only normalization. Validation statistics
    # and test statistics are never used.
    mean = train_x.mean(axis=(0, 1))
    std = train_x.std(axis=(0, 1))

    epsilon = float(
        training_cfg.get("normalization", {})
        .get("epsilon", 1e-6)
    )
    std = np.maximum(std, epsilon)

    train_z = (train_x - mean) / std
    val_z = (val_x - mean) / std

    train_tensor = torch.tensor(
        train_z, dtype=torch.float32, device=device
    )

    val_segments = torch.tensor(
        validation_segments(val_z),
        dtype=torch.float32,
        device=device,
    )

    dt = float(np.mean(np.diff(train_t)))

    segment_times = torch.arange(
        segment_length,
        dtype=torch.float32,
        device=device,
    ) * dt

    val_times = torch.arange(
        VAL_HORIZON,
        dtype=torch.float32,
        device=device,
    ) * dt

    model = VectorField(
        dim=train_x.shape[-1],
        hidden=hidden,
        mean=mean,
        std=std,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr
    )

    scheduler_cfg = training_cfg.get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(scheduler_cfg.get("factor", 0.5)),
        patience=int(scheduler_cfg.get("patience", 20)),
    )

    patience = int(
        training_cfg.get("early_stopping", {})
        .get("patience", 60)
    )

    rng = np.random.default_rng(training_seed)

    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []

    n_train, n_points, _ = train_tensor.shape
    batch_size = min(batch_trajectories, n_train)

    print(
        f"\n=== TRAIN NOISE ROBUSTNESS ===\n"
        f"noise={noise} seed={seed}\n"
        f"train={train_x.shape} val={val_x.shape}\n"
        f"epochs={epochs} hidden={hidden} "
        f"batches_per_epoch={batches_per_epoch}\n"
        f"device={device}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for _ in range(batches_per_epoch):
            trajectory_indices = rng.choice(
                n_train, size=batch_size, replace=False
            )

            start_indices = rng.integers(
                0,
                n_points - segment_length + 1,
                size=batch_size,
            )

            segments = torch.stack(
                [
                    train_tensor[int(i), int(start):
                                 int(start) + segment_length]
                    for i, start in zip(
                        trajectory_indices, start_indices
                    )
                ],
                dim=0,
            )

            optimizer.zero_grad(set_to_none=True)

            prediction = model.trajectory(
                segments[:, 0, :],
                segment_times,
            )

            loss = torch.mean(
                (prediction - segments) ** 2
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss: "
                    f"{noise}, seed={seed}, epoch={epoch}"
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()

            losses.append(float(loss.item()))

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

        scheduler.step(val_loss)

        train_loss = float(np.mean(losses))
        current_lr = float(
            optimizer.param_groups[0]["lr"]
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr,
        })

        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_epoch = epoch
            stale_epochs = 0

            torch.save({
                "state_dict": model.state_dict(),
                "dim": int(train_x.shape[-1]),
                "hidden": hidden,
                "system": system,
                "seed": seed,
                "noise_label": noise,
                "version": "robustness_v1",
                "normalized": True,
                "best_val": best_val,
                "best_epoch": best_epoch,
                "training_seed": training_seed,
                "training_config": training_cfg,
            }, checkpoint_path)

        else:
            stale_epochs += 1

        if (
            epoch == 1
            or epoch % 25 == 0
            or epoch == epochs
        ):
            print(
                f"epoch={epoch:3d} "
                f"train={train_loss:.6f} "
                f"val={val_loss:.6f} "
                f"best={best_val:.6f} "
                f"lr={current_lr:.6g}"
            )

        if stale_epochs >= patience:
            print(
                f"Early stopping at epoch {epoch}"
            )
            break

    history_path.write_text(
        json.dumps({
            "system": system,
            "noise_label": noise,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val": best_val,
            "history": history,
        }, indent=2),
        encoding="utf-8",
    )

    print(
        f"BEST noise={noise} seed={seed} "
        f"epoch={best_epoch} val={best_val:.6f}"
    )
    print(f"Checkpoint: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", type=int, default=101
    )
    parser.add_argument(
        "--noise",
        choices=NOISE_LABELS,
        default=None,
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all 25 conditions",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing checkpoints",
    )

    args = parser.parse_args()
    cfg = config()
    system = cfg["system"]

    if args.all:
        selected_seeds = SEEDS
        selected_noises = NOISE_LABELS
    else:
        if args.seed not in SEEDS:
            parser.error(
                f"Seed must be one of {SEEDS}"
            )

        selected_seeds = [args.seed]
        selected_noises = (
            [args.noise]
            if args.noise is not None
            else NOISE_LABELS
        )

    for seed in selected_seeds:
        for noise in selected_noises:
            train_one(
                system, noise, seed, cfg,
                overwrite=args.overwrite,
            )

    print("\n=== EXPERIMENT 21 COMPLETE ===")
    print("Test data was not accessed.")


if __name__ == "__main__":
    main()
