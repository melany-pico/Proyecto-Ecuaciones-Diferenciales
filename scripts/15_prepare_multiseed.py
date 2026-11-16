
"""Prepare independent multiseed datasets for a reproducible benchmark.

Creates separate train/validation/test splits for each seed.
Does not modify the existing data/processed files.
"""

import sys
import json
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

import numpy as np

from dynadistill.utils.common import ROOT, config
from dynadistill.data.systems import initial, simulate


SEEDS = [101, 202, 303, 404, 505]
SYSTEMS = ["lorenz"]

TRAIN_COUNT = 16
VAL_COUNT = 3
TEST_COUNT = 5


def generate_dataset(system, seed, cfg):
    simulation = cfg["simulation"]

    total = int(simulation["trajectories"])
    points = int(simulation["points"])
    t_end = float(simulation["t_end"])
    noise_std = float(simulation["noise_std"])

    expected = TRAIN_COUNT + VAL_COUNT + TEST_COUNT

    if total != expected:
        raise ValueError(
            f"Configured {total} trajectories, "
            f"but splits require {expected}"
        )

    if points < 3 or t_end <= 0 or noise_std < 0:
        raise ValueError("Invalid simulation configuration")

    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, t_end, points)

    clean_trajectories = []

    for index in range(total):
        x0 = initial(system, rng)
        trajectory = simulate(system, t, x0)

        if trajectory.shape != (points, len(x0)):
            raise ValueError(
                f"Unexpected shape for trajectory {index}"
            )

        if not np.isfinite(trajectory).all():
            raise ValueError(
                f"Non-finite values in trajectory {index}"
            )

        clean_trajectories.append(trajectory)

    clean = np.stack(clean_trajectories, axis=0)

    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=clean.shape,
    )
    observed = clean + noise

    permutation = rng.permutation(total)

    split_indices = {
        "train": permutation[:TRAIN_COUNT],
        "val": permutation[
            TRAIN_COUNT:TRAIN_COUNT + VAL_COUNT
        ],
        "test": permutation[
            TRAIN_COUNT + VAL_COUNT:
        ],
    }

    directory = (
        ROOT / "data" / "benchmark"
        / system / f"seed_{seed}"
    )
    directory.mkdir(parents=True, exist_ok=True)

    for split, indices in split_indices.items():
        path = directory / f"{split}.npz"

        np.savez_compressed(
            path,
            x=observed[indices],
            x_clean=clean[indices],
            t=t,
            trajectory_ids=indices.astype(np.int64),
        )

    metadata = {
        "system": system,
        "seed": seed,
        "trajectories": total,
        "points": points,
        "t_end": t_end,
        "noise_std": noise_std,
        "split_counts": {
            key: len(value)
            for key, value in split_indices.items()
        },
        "split_trajectory_ids": {
            key: value.tolist()
            for key, value in split_indices.items()
        },
        "data_policy": {
            "train": "Model fitting",
            "val": "Hyperparameter and model selection",
            "test": "Final evaluation only",
            "x_clean": (
                "Stored for controlled evaluation; "
                "must not be used for realistic model fitting"
            ),
        },
    }

    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return directory, observed.shape


def main():
    cfg = config()

    if len(SEEDS) != len(set(SEEDS)):
        raise ValueError("Seeds must be unique")

    print("\n=== MULTISEED BENCHMARK DATA PREPARATION ===")
    print(f"Seeds: {SEEDS}")
    print(f"Systems: {SYSTEMS}")
    print(
        f"Splits: train={TRAIN_COUNT}, "
        f"val={VAL_COUNT}, test={TEST_COUNT}"
    )

    manifest = {
        "seeds": SEEDS,
        "systems": SYSTEMS,
        "splits": {
            "train": TRAIN_COUNT,
            "val": VAL_COUNT,
            "test": TEST_COUNT,
        },
        "runs": [],
    }

    for system in SYSTEMS:
        for seed in SEEDS:
            directory, shape = generate_dataset(
                system, seed, cfg
            )

            manifest["runs"].append({
                "system": system,
                "seed": seed,
                "directory": str(
                    directory.relative_to(ROOT)
                ),
                "observed_shape": list(shape),
            })

            print(
                f"[OK] {system:15s} "
                f"seed={seed} "
                f"shape={shape}"
            )

    output = ROOT / "results" / "benchmark"
    output.mkdir(parents=True, exist_ok=True)

    manifest_path = output / "data_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\n=== COMPLETED ===")
    print(f"Generated runs: {len(manifest['runs'])}")
    print(f"Manifest: {manifest_path}")
    print(
        "Original data/processed datasets "
        "were not modified."
    )


if __name__ == "__main__":
    main()
