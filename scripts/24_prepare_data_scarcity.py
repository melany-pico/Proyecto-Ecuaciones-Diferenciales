
"""Experiment 24: Prepare temporally sparse Lorenz observations.

Uses the original five-seed benchmark data, preserving:
- Train/validation/test trajectory identities
- Original observation noise (sigma = 0.01)
- Clean reference trajectories
- Original measurement times

Temporal subsets are deterministic and nested.

No model training, no test-based model selection.

Run:
    python scripts/24_prepare_data_scarcity.py --seed 101
    python scripts/24_prepare_data_scarcity.py --all
"""

import sys
import json
import argparse
import hashlib
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

import numpy as np

from dynadistill.utils.common import ROOT


SYSTEM = "lorenz"
SEEDS = [101, 202, 303, 404, 505]

LEVELS = {
    "obs_100": 160,
    "obs_050": 80,
    "obs_025": 40,
    "obs_010": 16,
}

EXPECTED_SPLITS = {
    "train": 16,
    "val": 3,
    "test": 5,
}

EXPECTED_NOISE_STD = 0.01


def temporal_order(n_points):
    """Create a deterministic, nested, space-filling order.

    Start with both endpoints, then repeatedly insert
    the midpoint of the largest remaining time gap.

    Prefixes of this order yield nested temporal subsets.
    """
    if n_points < 2:
        raise ValueError("At least two time points required")

    order = [0, n_points - 1]
    selected = {0, n_points - 1}

    while len(order) < n_points:
        sorted_indices = sorted(selected)

        best_gap = -1
        best_pair = None

        for left, right in zip(
            sorted_indices[:-1],
            sorted_indices[1:],
        ):
            gap = right - left

            if gap > best_gap:
                best_gap = gap
                best_pair = (left, right)

        if best_gap <= 1:
            raise RuntimeError(
                "No unselected temporal indices remain"
            )

        left, right = best_pair
        midpoint = (left + right) // 2

        if midpoint in selected:
            raise RuntimeError(
                "Temporal sampling produced a duplicate"
            )

        selected.add(midpoint)
        order.append(midpoint)

    if sorted(order) != list(range(n_points)):
        raise RuntimeError(
            "Temporal ordering is not a permutation"
        )

    return np.asarray(order, dtype=np.int64)


def indices_for_levels(n_points):
    order = temporal_order(n_points)

    result = {}

    for label, count in LEVELS.items():
        if count > n_points:
            raise ValueError(
                f"{label} requests {count} points, "
                f"but only {n_points} are available"
            )

        result[label] = np.sort(
            order[:count]
        )

    # Verify strict nesting.
    labels = sorted(
        LEVELS,
        key=lambda name: LEVELS[name],
    )

    for smaller, larger in zip(
        labels[:-1], labels[1:]
    ):
        if not set(result[smaller]).issubset(
            set(result[larger])
        ):
            raise RuntimeError(
                f"Non-nested subsets: {smaller}, {larger}"
            )

    return result


def source_path(seed, split):
    return (
        ROOT / "data" / "benchmark"
        / SYSTEM / f"seed_{seed}"
        / f"{split}.npz"
    )


def destination_path(seed, level, split):
    return (
        ROOT / "data" / "scarcity"
        / SYSTEM / level / f"seed_{seed}"
        / f"{split}.npz"
    )


def read_source(seed, split):
    path = source_path(seed, split)

    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        required = {
            "x",
            "x_clean",
            "t",
            "trajectory_ids",
        }

        missing = required - set(data.files)

        if missing:
            raise ValueError(
                f"{path} missing arrays: {sorted(missing)}"
            )

        result = {
            name: np.asarray(data[name]).copy()
            for name in required
        }

    x = result["x"]
    clean = result["x_clean"]
    t = result["t"]
    ids = result["trajectory_ids"]

    if x.shape != clean.shape:
        raise ValueError(
            f"Observed/clean shape mismatch: {path}"
        )

    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError(
            f"Expected Lorenz trajectories: {path}"
        )

    if x.shape[0] != EXPECTED_SPLITS[split]:
        raise ValueError(
            f"Unexpected trajectory count: {path}"
        )

    if x.shape[1] != len(t):
        raise ValueError(
            f"Time dimension mismatch: {path}"
        )

    if len(ids) != x.shape[0]:
        raise ValueError(
            f"Trajectory ID mismatch: {path}"
        )

    if len(t) != 160:
        raise ValueError(
            f"Expected 160 original time points: {path}"
        )

    if not (
        np.isfinite(x).all()
        and np.isfinite(clean).all()
        and np.isfinite(t).all()
    ):
        raise ValueError(
            f"Non-finite source data: {path}"
        )

    if not np.all(np.diff(t) > 0):
        raise ValueError(
            f"Non-increasing source times: {path}"
        )

    return result


def digest_array(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(
        array.tobytes()
    ).hexdigest()


def prepare_seed(seed, overwrite=False):
    print(
        f"\n=== DATA SCARCITY PREPARATION ===\n"
        f"system={SYSTEM} seed={seed}"
    )

    sources = {
        split: read_source(seed, split)
        for split in EXPECTED_SPLITS
    }

    reference_t = sources["train"]["t"]

    for split, data in sources.items():
        if not np.array_equal(
            data["t"], reference_t
        ):
            raise ValueError(
                f"Time grids differ in {split}"
            )

    subsets = indices_for_levels(
        len(reference_t)
    )

    # Check that trajectory IDs are disjoint across splits.
    id_sets = {
        split: set(
            str(item)
            for item in data["trajectory_ids"]
        )
        for split, data in sources.items()
    }

    for a, b in [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ]:
        if id_sets[a] & id_sets[b]:
            raise ValueError(
                f"Trajectory ID overlap: {a}, {b}"
            )

    # First check every destination to avoid partial
    # writing due to an existing file.
    destinations = [
        destination_path(seed, level, split)
        for level in LEVELS
        for split in EXPECTED_SPLITS
    ]

    existing = [
        path for path in destinations
        if path.exists()
    ]

    if existing and not overwrite:
        raise FileExistsError(
            "Some scarcity files already exist. "
            "Refusing to overwrite. First file: "
            f"{existing[0]}"
        )

    metadata = {
        "experiment": 24,
        "system": SYSTEM,
        "seed": seed,
        "source": "data/benchmark/lorenz",
        "source_noise_std": EXPECTED_NOISE_STD,
        "sampling": (
            "deterministic nested largest-gap "
            "midpoint temporal subsampling"
        ),
        "interpolation": False,
        "n_original_points": len(reference_t),
        "splits": EXPECTED_SPLITS,
        "levels": {},
    }

    for level, indices in subsets.items():
        time_subset = reference_t[indices]

        spacing = np.diff(time_subset)
        uniform = bool(
            np.allclose(
                spacing,
                spacing[0],
                rtol=1e-5,
                atol=1e-7,
            )
        )

        metadata["levels"][level] = {
            "n_points": len(indices),
            "fraction": (
                len(indices) / len(reference_t)
            ),
            "indices": indices.tolist(),
            "indices_sha256": digest_array(
                indices
            ),
            "time_uniform": uniform,
            "min_dt": float(spacing.min()),
            "max_dt": float(spacing.max()),
        }

        print(
            f"\n{level}: "
            f"points={len(indices):3d} "
            f"fraction={len(indices)/len(reference_t):.3f} "
            f"uniform_time={uniform}"
        )

        for split, source in sources.items():
            path = destination_path(
                seed, level, split
            )
            path.parent.mkdir(
                parents=True, exist_ok=True
            )

            x = source["x"][:, indices, :]
            clean = source["x_clean"][:, indices, :]

            # Check exact preservation of source values.
            if not np.array_equal(
                x, source["x"][:, indices, :]
            ):
                raise RuntimeError(
                    "Observed values changed"
                )

            np.savez_compressed(
                path,
                x=x,
                x_clean=clean,
                t=time_subset,
                trajectory_ids=source[
                    "trajectory_ids"
                ],
                temporal_indices=indices,
                noise_std=np.float64(
                    EXPECTED_NOISE_STD
                ),
                seed=np.int64(seed),
            )

            print(
                f"  {split:5s} "
                f"shape={x.shape} "
                f"saved={path.name}"
            )

    metadata_path = (
        ROOT / "results" / "scarcity"
        / SYSTEM / f"seed_{seed}"
        / "metadata.json"
    )

    metadata_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    if metadata_path.exists() and not overwrite:
        raise FileExistsError(
            f"Metadata already exists: {metadata_path}"
        )

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(
        f"\nCompleted seed={seed}"
    )
    print(
        f"Metadata: {metadata_path}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=101,
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

    if args.all:
        seeds = SEEDS
    else:
        if args.seed not in SEEDS:
            parser.error(
                f"Choose a seed from {SEEDS}"
            )
        seeds = [args.seed]

    for seed in seeds:
        prepare_seed(
            seed,
            overwrite=args.overwrite,
        )

    print(
        "\n=== EXPERIMENT 24 COMPLETE ==="
    )
    print(
        "No model training was performed."
    )
    print(
        "No test-based model selection was performed."
    )


if __name__ == "__main__":
    main()
