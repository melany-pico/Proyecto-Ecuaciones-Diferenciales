
"""Experiment 20: Prepare controlled noise-robustness datasets.

Creates five noise conditions from the same clean trajectories
for each seed. Does not train models or access test outcomes.

Run:
    python scripts/20_noise_robustness.py
"""

import sys
import json
import hashlib
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

import numpy as np

from dynadistill.utils.common import ROOT, config


SEEDS = [101, 202, 303, 404, 505]

NOISE_LEVELS = [
    ("noise_000", 0.00),
    ("noise_001", 0.01),
    ("noise_005", 0.05),
    ("noise_010", 0.10),
    ("noise_020", 0.20),
]

SPLITS = ["train", "val", "test"]

# Deterministic but independent random streams.
SPLIT_IDS = {
    "train": 1,
    "val": 2,
    "test": 3,
}

NOISE_IDS = {
    "noise_000": 0,
    "noise_001": 1,
    "noise_005": 5,
    "noise_010": 10,
    "noise_020": 20,
}


def sha256_array(array):
    """Fingerprint the underlying clean trajectory data."""
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(
        contiguous.tobytes()
    ).hexdigest()


def load_original(system, seed, split):
    path = (
        ROOT / "data" / "benchmark"
        / system / f"seed_{seed}" / f"{split}.npz"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path) as data:
        clean = np.asarray(
            data["x_clean"],
            dtype=np.float64,
        )

        t = np.asarray(
            data["t"],
            dtype=np.float64,
        )

        trajectory_ids = (
            np.asarray(data["trajectory_ids"]).copy()
            if "trajectory_ids" in data.files
            else np.arange(len(clean))
        )

    if clean.ndim != 3:
        raise ValueError(
            f"Expected 3D trajectories: {path}"
        )

    if clean.shape[1] != len(t):
        raise ValueError(
            f"Time dimension mismatch: {path}"
        )

    if not np.isfinite(clean).all():
        raise ValueError(
            f"Non-finite clean trajectories: {path}"
        )

    if not np.isfinite(t).all():
        raise ValueError(
            f"Non-finite time grid: {path}"
        )

    if len(trajectory_ids) != len(clean):
        raise ValueError(
            f"Trajectory IDs mismatch: {path}"
        )

    return clean, t, trajectory_ids


def generate_observations(
    clean, seed, split, noise_label, sigma
):
    if sigma == 0.0:
        return clean.copy()

    random_seed = np.random.SeedSequence([
        seed,
        SPLIT_IDS[split],
        NOISE_IDS[noise_label],
        2026,
    ])

    rng = np.random.default_rng(random_seed)

    return clean + rng.normal(
        loc=0.0,
        scale=sigma,
        size=clean.shape,
    )


def prepare_condition(
    system, seed, noise_label, sigma
):
    destination = (
        ROOT / "data" / "robustness"
        / system / noise_label / f"seed_{seed}"
    )

    destination.mkdir(
        parents=True, exist_ok=True
    )

    records = {}

    for split in SPLITS:
        clean, t, trajectory_ids = load_original(
            system, seed, split
        )

        observed = generate_observations(
            clean,
            seed,
            split,
            noise_label,
            sigma,
        )

        if not np.isfinite(observed).all():
            raise ValueError(
                "Generated non-finite observations"
            )

        output = destination / f"{split}.npz"

        # Refuse to silently overwrite experimental data.
        if output.exists():
            with np.load(output) as existing:
                same = (
                    np.array_equal(
                        existing["x"], observed
                    )
                    and np.array_equal(
                        existing["x_clean"], clean
                    )
                    and np.array_equal(
                        existing["t"], t
                    )
                    and np.array_equal(
                        existing["trajectory_ids"],
                        trajectory_ids,
                    )
                )

            if not same:
                raise RuntimeError(
                    "Existing file differs from "
                    f"deterministic generation: {output}"
                )

            status = "verified"

        else:
            np.savez_compressed(
                output,
                x=observed,
                x_clean=clean,
                t=t,
                trajectory_ids=trajectory_ids,
                noise_std=np.float64(sigma),
                seed=np.int64(seed),
            )

            status = "created"

        records[split] = {
            "file": str(output.relative_to(ROOT)),
            "trajectories": int(len(clean)),
            "points": int(clean.shape[1]),
            "dimension": int(clean.shape[2]),
            "clean_sha256": sha256_array(clean),
            "status": status,
        }

        print(
            f"  {split:5s} "
            f"shape={str(clean.shape):17s} "
            f"{status}"
        )

    metadata = {
        "experiment": 20,
        "system": system,
        "seed": seed,
        "noise_label": noise_label,
        "noise_std": sigma,
        "noise_distribution": (
            "independent additive Gaussian, "
            "N(0, sigma^2), per state observation"
        ),
        "clean_source": (
            "data/benchmark/<system>/seed_<seed>"
        ),
        "splits": records,
        "test_policy": (
            "Test data generated and sealed; "
            "no model fitting or selection on test"
        ),
    }

    metadata_path = destination / "metadata.json"

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return metadata


def main():
    cfg = config()
    system = cfg["system"]

    print(
        "\n=== EXPERIMENT 20: NOISE ROBUSTNESS DATA ==="
    )
    print(f"System: {system}")
    print(f"Seeds: {SEEDS}")
    print(
        "Noise levels: "
        + ", ".join(
            str(sigma)
            for _, sigma in NOISE_LEVELS
        )
    )

    manifest = {
        "experiment": 20,
        "system": system,
        "seeds": SEEDS,
        "noise_levels": [
            {
                "label": label,
                "std": sigma,
            }
            for label, sigma in NOISE_LEVELS
        ],
        "conditions": [],
    }

    for noise_label, sigma in NOISE_LEVELS:
        print(
            f"\n=== {noise_label} "
            f"(sigma={sigma:.3f}) ==="
        )

        for seed in SEEDS:
            print(f"Seed {seed}:")

            metadata = prepare_condition(
                system,
                seed,
                noise_label,
                sigma,
            )

            manifest["conditions"].append({
                "seed": seed,
                "noise_label": noise_label,
                "noise_std": sigma,
                "train_count": metadata[
                    "splits"
                ]["train"]["trajectories"],
                "val_count": metadata[
                    "splits"
                ]["val"]["trajectories"],
                "test_count": metadata[
                    "splits"
                ]["test"]["trajectories"],
            })

    output = (
        ROOT / "results" / "robustness"
        / system
    )

    output.mkdir(
        parents=True, exist_ok=True
    )

    manifest_path = (
        output / "noise_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(
        "\n=== DATA PREPARATION COMPLETE ==="
    )
    print(
        f"Conditions: {len(manifest['conditions'])}"
    )
    print(
        f"Dataset files: "
        f"{len(manifest['conditions']) * len(SPLITS)}"
    )
    print(
        f"Manifest: {manifest_path}"
    )
    print(
        "\nNo models trained. "
        "No test metrics calculated."
    )


if __name__ == "__main__":
    main()
