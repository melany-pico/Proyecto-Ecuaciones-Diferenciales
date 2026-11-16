
"""Experiment 25: Paired Neural ODE training under temporal data scarcity.

Run:
    python scripts/25_train_data_scarcity.py --seed 101
    python scripts/25_train_data_scarcity.py --all

Training and validation only. Test data is never opened.
Existing checkpoints are skipped, never overwritten.
"""

import argparse
import csv
import hashlib
import io
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torchdiffeq import odeint

from dynadistill.utils.common import ROOT, config
from dynadistill.models.neuralode import VectorField


SEEDS = [101, 202, 303, 404, 505]
LEVELS = ["obs_100", "obs_050", "obs_025", "obs_010"]
SYSTEM = "lorenz"

# Physical time units, not observation counts.
WINDOW_DURATION = 0.80
VALIDATION_WINDOWS = 8

# One training minibatch contains several trajectories
# observed over the same relative time grid.
LOG_EVERY = 25


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def data_path(level, seed, split):
    return (
        ROOT / "data" / "scarcity" / SYSTEM
        / level / f"seed_{seed}" / f"{split}.npz"
    )


def output_dir(level, seed):
    return (
        ROOT / "results" / "scarcity"
        / SYSTEM / level / f"seed_{seed}"
    )


def load_split(level, seed, split):
    path = data_path(level, seed, split)

    with np.load(path, allow_pickle=False) as data:
        x = np.asarray(data["x"], dtype=np.float32)
        t = np.asarray(data["t"], dtype=np.float64)

    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError(f"Invalid trajectories: {path}")

    if len(t) != x.shape[1] or np.any(np.diff(t) <= 0):
        raise ValueError(f"Invalid times: {path}")

    if not np.isfinite(x).all() or not np.isfinite(t).all():
        raise ValueError(f"Nonfinite values: {path}")

    return x, t


def make_windows(t):
    """Find windows with approximately equal physical duration.

    Each window starts at an observed instant and ends
    at the closest subsequent observation to t0+duration.
    Windows always contain at least two observations.
    """
    windows = []

    for start in range(len(t) - 1):
        target = t[start] + WINDOW_DURATION

        if target > t[-1] + 1e-10:
            continue

        end = int(np.searchsorted(t, target))

        choices = [
            k for k in (end - 1, end)
            if start < k < len(t)
        ]

        if not choices:
            continue

        end = min(
            choices,
            key=lambda k: abs(t[k] - target),
        )

        windows.append((start, end))

    if not windows:
        raise ValueError(
            "No valid physical-time windows. "
            "Increase observation density or reduce duration."
        )

    return windows


def window_at_fraction(windows, fraction):
    index = min(
        int(float(fraction) * len(windows)),
        len(windows) - 1,
    )
    return windows[index]


def common_schedule(seed, epochs, batches, batch_size):
    """Identical schedule for every level of a given seed.

    Fractions specify physical-window positions, not
    observation indices.
    """
    rng = np.random.default_rng(500000 + seed)
    schedule = []

    for _ in range(epochs):
        epoch_schedule = []

        for _ in range(batches):
            trajectory_indices = rng.choice(
                16, size=batch_size, replace=False
            ).astype(np.int64)

            position = float(rng.random())

            epoch_schedule.append(
                (trajectory_indices, position)
            )

        schedule.append(epoch_schedule)

    buffer = io.BytesIO()

    for epoch in schedule:
        for indices, position in epoch:
            buffer.write(indices.tobytes())
            buffer.write(np.float64(position).tobytes())

    return schedule, sha256_bytes(buffer.getvalue())


def initial_model(seed, hidden):
    seed_everything(700000 + seed)

    model = VectorField(dim=3, hidden=hidden)

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)

    return model, sha256_bytes(buffer.getvalue())


def configure_normalization(model, x_train, epsilon):
    mean = x_train.reshape(-1, 3).mean(axis=0)
    std = x_train.reshape(-1, 3).std(axis=0)
    std = np.maximum(std, epsilon)

    with torch.no_grad():
        model.state_mean.copy_(
            torch.from_numpy(mean.astype(np.float32))
        )
        model.state_std.copy_(
            torch.from_numpy(std.astype(np.float32))
        )


def normalized_data(model, x):
    mean = model.state_mean.cpu().numpy()
    std = model.state_std.cpu().numpy()

    return torch.tensor(
        (x - mean) / std,
        dtype=torch.float32,
    )


def predict_segment(model, x, t, start, end):
    relative_t = torch.tensor(
        t[start:end + 1] - t[start],
        dtype=torch.float32,
    )

    # odeint output: (time, batch, dimension)
    prediction = odeint(
        model,
        x[:, start, :],
        relative_t,
        method="dopri5",
        rtol=1e-4,
        atol=1e-5,
    ).transpose(0, 1)

    target = x[:, start:end + 1, :]

    return prediction, target


def validation_loss(model, x_val, t_val, windows):
    fractions = np.linspace(
        0.0, 1.0, VALIDATION_WINDOWS,
        endpoint=False,
    ) + 0.5 / VALIDATION_WINDOWS

    losses = []

    model.eval()

    with torch.no_grad():
        for fraction in fractions:
            start, end = window_at_fraction(
                windows, fraction
            )

            predicted, target = predict_segment(
                model, x_val, t_val, start, end
            )

            # Each validation window contributes equally.
            losses.append(
                float(torch.mean((predicted - target) ** 2))
            )

    return float(np.mean(losses))


def train_one(seed, level, cfg, schedule, schedule_hash):
    destination = output_dir(level, seed)
    checkpoint_path = destination / "neural_ode.pt"
    history_path = destination / "training_history.csv"
    metadata_path = destination / "training_summary.json"

    if checkpoint_path.exists():
        print(f"SKIPPED seed={seed} level={level}: checkpoint exists")
        return

    if history_path.exists() or metadata_path.exists():
        raise FileExistsError(
            "Partial outputs exist without checkpoint: "
            f"{destination}"
        )

    train_x, train_t = load_split(level, seed, "train")
    val_x, val_t = load_split(level, seed, "val")

    if len(train_x) != 16 or len(val_x) != 3:
        raise ValueError("Unexpected train/validation trajectory counts")

    train_windows = make_windows(train_t)
    val_windows = make_windows(val_t)

    settings = cfg["training"]

    epochs = int(settings["epochs"])
    batches = int(settings["batches_per_epoch"])
    batch_size = int(settings["batch_trajectories"])
    hidden = int(settings["hidden"])
    learning_rate = float(settings["lr"])

    epsilon = float(
        settings.get("normalization", {}).get(
            "epsilon", 1e-6
        )
    )

    model, initial_hash = initial_model(seed, hidden)

    # Same initial trainable weights across levels.
    # Normalization buffers necessarily differ by dataset.
    configure_normalization(model, train_x, epsilon)

    train_z = normalized_data(model, train_x)
    val_z = normalized_data(model, val_x)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    scheduler_cfg = settings.get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(scheduler_cfg.get("factor", 0.5)),
        patience=int(scheduler_cfg.get("patience", 20)),
    )

    patience = int(
        settings.get("early_stopping", {}).get(
            "patience", 60
        )
    )

    best_val = float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    history = []

    print(
        f"\n=== TRAINING seed={seed} level={level} ==="
    )
    print(
        f"train={train_x.shape} val={val_x.shape} "
        f"train_windows={len(train_windows)} "
        f"val_windows={len(val_windows)}"
    )
    print(
        f"initial_weights_sha256={initial_hash[:16]}"
    )
    print(
        f"minibatch_schedule_sha256={schedule_hash[:16]}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for indices, fraction in schedule[epoch - 1]:
            start, end = window_at_fraction(
                train_windows, fraction
            )

            batch = train_z[indices]

            optimizer.zero_grad(set_to_none=True)

            predicted, target = predict_segment(
                model, batch, train_t, start, end
            )

            loss = torch.mean(
                (predicted - target) ** 2
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Nonfinite training loss: "
                    f"seed={seed}, level={level}, "
                    f"epoch={epoch}"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()
            train_losses.append(float(loss.item()))

        val_loss = validation_loss(
            model, val_z, val_t, val_windows
        )

        if not np.isfinite(val_loss):
            raise RuntimeError(
                f"Nonfinite validation loss: "
                f"seed={seed}, level={level}, "
                f"epoch={epoch}"
            )

        scheduler.step(val_loss)

        train_loss = float(np.mean(train_losses))
        lr = float(optimizer.param_groups[0]["lr"])

        history.append({
            "epoch": epoch,
            "train_mse": train_loss,
            "val_mse": val_loss,
            "lr": lr,
        })

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_epoch = epoch
            stale_epochs = 0

            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        else:
            stale_epochs += 1

        if (
            epoch == 1
            or epoch % LOG_EVERY == 0
            or epoch == epochs
        ):
            print(
                f"epoch={epoch:3d} "
                f"train={train_loss:.6f} "
                f"val={val_loss:.6f} "
                f"best={best_val:.6f} "
                f"lr={lr:.6g}"
            )

        if stale_epochs >= patience:
            print(
                f"Early stopping at epoch {epoch}"
            )
            break

    if best_state is None:
        raise RuntimeError("No valid checkpoint was selected")

    destination.mkdir(
        parents=True, exist_ok=True
    )

    torch.save({
        "state_dict": best_state,
        "dim": 3,
        "hidden": hidden,
        "system": SYSTEM,
        "seed": seed,
        "level": level,
        "version": "scarcity_paired_v1",
        "normalized": True,
        "best_val": best_val,
        "best_epoch": best_epoch,
        "initialization_seed": 700000 + seed,
        "initial_weights_sha256": initial_hash,
        "minibatch_schedule_sha256": schedule_hash,
        "window_duration": WINDOW_DURATION,
        "training_config": {
            "epochs": epochs,
            "batches_per_epoch": batches,
            "batch_trajectories": batch_size,
            "learning_rate": learning_rate,
            "normalization_epsilon": epsilon,
            "validation_windows": VALIDATION_WINDOWS,
        },
    }, checkpoint_path)

    with history_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch", "train_mse", "val_mse", "lr"
            ],
        )
        writer.writeheader()
        writer.writerows(history)

    metadata_path.write_text(
        json.dumps({
            "experiment": 25,
            "system": SYSTEM,
            "seed": seed,
            "level": level,
            "best_epoch": best_epoch,
            "best_val_mse": best_val,
            "completed_epochs": len(history),
            "train_windows": len(train_windows),
            "val_windows": len(val_windows),
            "initial_weights_sha256": initial_hash,
            "minibatch_schedule_sha256": schedule_hash,
            "window_duration": WINDOW_DURATION,
            "test_accessed": False,
        }, indent=2),
        encoding="utf-8",
    )

    print(
        f"BEST seed={seed} level={level} "
        f"epoch={best_epoch} "
        f"val_MSE={best_val:.6f}"
    )
    print(f"Checkpoint: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", type=int, default=101
    )
    parser.add_argument(
        "--all", action="store_true"
    )
    parser.add_argument(
        "--level",
        choices=LEVELS,
        default=None,
    )

    args = parser.parse_args()

    cfg = config()

    if cfg["system"] != SYSTEM:
        raise ValueError(
            "Experiment 25 supports Lorenz only"
        )

    settings = cfg["training"]

    epochs = int(settings["epochs"])
    batches = int(settings["batches_per_epoch"])
    batch_size = int(settings["batch_trajectories"])

    seeds = SEEDS if args.all else [args.seed]

    for seed in seeds:
        if seed not in SEEDS:
            parser.error(f"Invalid seed: {seed}")

        schedule, schedule_hash = common_schedule(
            seed, epochs, batches, batch_size
        )

        levels = (
            [args.level]
            if args.level is not None
            else LEVELS
        )

        for level in levels:
            train_one(
                seed,
                level,
                cfg,
                schedule,
                schedule_hash,
            )

    print(
        "\n=== EXPERIMENT 25 COMPLETE ==="
    )
    print(
        "Only training/validation data were used."
    )
    print(
        "Test data were not accessed."
    )


if __name__ == "__main__":
    main()
