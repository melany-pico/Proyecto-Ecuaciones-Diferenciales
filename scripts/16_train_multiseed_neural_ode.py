
"""Train one Neural ODE per benchmark seed, using train/val only."""

import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from dynadistill.utils.common import ROOT, config, seed_all
from dynadistill.models.neuralode import VectorField


def load_data(system, seed, split):
    path = (
        ROOT / "data" / "benchmark" / system
        / f"seed_{seed}" / f"{split}.npz"
    )
    with np.load(path) as data:
        x = np.asarray(data["x"], dtype=np.float32)
        t = np.asarray(data["t"], dtype=np.float32)

    if x.ndim != 3 or t.ndim != 1:
        raise ValueError(f"Invalid data dimensions: {path}")
    if x.shape[1] != len(t):
        raise ValueError(f"Time dimension mismatch: {path}")
    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite values: {path}")
    return x, t


def make_segments(x, length):
    """All consecutive windows, shape (N, length, dim)."""
    windows = [
        x[:, i:i + length, :]
        for i in range(x.shape[1] - length + 1)
    ]
    return torch.cat(windows, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = config()
    system = cfg["system"]
    tr = cfg["training"]
    seed_all(args.seed)

    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))
    device = torch.device(tr.get("device", "cpu"))

    x_train, t_train = load_data(system, args.seed, "train")
    x_val, t_val = load_data(system, args.seed, "val")

    if not np.allclose(t_train, t_val):
        raise ValueError("Training and validation time grids differ")

    length = int(tr["segment_length"])
    if length < 2 or length > x_train.shape[1]:
        raise ValueError("Invalid segment length")

    # Compute normalization using training observations only.
    flat_train = x_train.reshape(-1, x_train.shape[-1])
    mean = flat_train.mean(axis=0)
    std = flat_train.std(axis=0)
    std = np.maximum(
        std,
        float(tr.get("normalization", {}).get("epsilon", 1e-6))
    )

    norm_enabled = tr.get("normalization", {}).get("enabled", True)
    if not norm_enabled:
        mean = np.zeros_like(mean)
        std = np.ones_like(std)

    train_z = torch.tensor(
        (x_train - mean) / std, dtype=torch.float32
    )
    val_z = torch.tensor(
        (x_val - mean) / std, dtype=torch.float32
    )

    train_segments = make_segments(train_z, length)
    val_segments = make_segments(val_z, length)

    # Relative time offsets, not absolute timestamps.
    dt = float(t_train[1] - t_train[0])
    time = torch.arange(
        length, dtype=torch.float32, device=device
    ) * dt

    model = VectorField(
        dim=x_train.shape[-1],
        hidden=int(tr["hidden"]),
        mean=mean,
        std=std,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(tr["lr"])
    )

    scheduler_cfg = tr.get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(scheduler_cfg.get("factor", 0.5)),
        patience=int(scheduler_cfg.get("patience", 20)),
    )

    epochs = (
        args.epochs if args.epochs is not None
        else int(tr["epochs"])
    )
    batch_size = int(tr["batch_trajectories"])
    batches_per_epoch = int(tr["batches_per_epoch"])
    patience = int(tr.get("early_stopping", {}).get("patience", 60))

    # Fixed validation windows, with no stochastic resampling.
    val_segments = val_segments.to(device)

    out_dir = ROOT / "results" / "benchmark" / system
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / f"neural_ode_seed_{args.seed}.pt"
    history_path = out_dir / f"neural_ode_seed_{args.seed}_history.json"

    best_val = float("inf")
    best_epoch = 0
    history = []

    print("\n=== MULTISEED NEURAL ODE TRAINING ===")
    print(f"System: {system} | Seed: {args.seed}")
    print(f"Device: {device} | Epoch limit: {epochs}")
    print(
        f"Train windows: {len(train_segments)} | "
        f"Validation windows: {len(val_segments)}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for _ in range(batches_per_epoch):
            ids = torch.randint(
                0, len(train_segments), (batch_size,)
            )
            target = train_segments[ids].to(device)

            optimizer.zero_grad(set_to_none=True)
            prediction = model.trajectory(target[:, 0, :], time)

            loss = torch.mean((prediction - target) ** 2)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss at epoch {epoch}"
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            # Evaluate in batches to avoid excessive memory use.
            val_losses = []
            for chunk in val_segments.split(64):
                prediction = model.trajectory(
                    chunk[:, 0, :], time
                )
                val_losses.append(
                    (
                        float(torch.sum((prediction - chunk) ** 2)),
                        int(chunk.numel()),
                    )
                )

            val_loss = (
                sum(s for s, _ in val_losses)
                / sum(n for _, n in val_losses)
            )

        train_loss = float(np.mean(losses))
        scheduler.step(val_loss)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch

            torch.save({
                "state_dict": model.state_dict(),
                "dim": int(x_train.shape[-1]),
                "hidden": int(tr["hidden"]),
                "system": system,
                "version": "multiseed_v1",
                "normalized": bool(norm_enabled),
                "best_val": best_val,
                "best_epoch": best_epoch,
                "seed": args.seed,
                "training_config": tr,
            }, checkpoint_path)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d} | "
                f"train={train_loss:.6f} | "
                f"val={val_loss:.6f} | "
                f"best={best_val:.6f} "
                f"(epoch {best_epoch})"
            )

        if epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    history_path.write_text(
        json.dumps({
            "system": system,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "best_val": best_val,
            "history": history,
        }, indent=2),
        encoding="utf-8",
    )

    print("\n=== TRAINING COMPLETED ===")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val:.6f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()
