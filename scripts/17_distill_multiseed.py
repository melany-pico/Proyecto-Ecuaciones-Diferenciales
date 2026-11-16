
"""Multiseed Neural ODE -> sparse polynomial equation distillation.

Selection uses only training states and validation trajectories.
The test split is never loaded.

Run:
    python scripts/17_distill_multiseed.py
    python scripts/17_distill_multiseed.py --seed 101
"""

import sys
import csv
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from scipy.integrate import solve_ivp
from sklearn.preprocessing import PolynomialFeatures

from dynadistill.utils.common import ROOT, config, seed_all
from dynadistill.models.neuralode import VectorField
from dynadistill.discovery.sparse import stlsq


SEEDS = [101, 202, 303, 404, 505]
DEGREES = [2, 3]
THRESHOLDS = [0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50]

HORIZON_POINTS = 9
SEGMENTS_PER_TRAJECTORY = 8
COMPLEXITY_PENALTY = 0.005

# Integrator tolerances fixed for all candidate models.
RTOL = 1e-6
ATOL = 1e-8


def load_split(system, seed, split):
    path = (
        ROOT / "data" / "benchmark"
        / system / f"seed_{seed}" / f"{split}.npz"
    )

    with np.load(path) as data:
        x = np.asarray(data["x"], dtype=np.float64)
        t = np.asarray(data["t"], dtype=np.float64)

    if x.ndim != 3 or t.ndim != 1:
        raise ValueError(f"Invalid dataset dimensions: {path}")

    if x.shape[1] != len(t):
        raise ValueError(f"Time dimension mismatch: {path}")

    if not np.isfinite(x).all() or not np.isfinite(t).all():
        raise ValueError(f"Non-finite data: {path}")

    if not np.all(np.diff(t) > 0):
        raise ValueError(f"Time must be increasing: {path}")

    return x, t


def load_neural_model(system, seed):
    path = (
        ROOT / "results" / "benchmark" / system
        / f"neural_ode_seed_{seed}.pt"
    )

    # Checkpoints were produced locally by script 16.
    try:
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(
            path, map_location="cpu"
        )

    if checkpoint["system"] != system:
        raise ValueError("Checkpoint system mismatch")

    if int(checkpoint["seed"]) != seed:
        raise ValueError("Checkpoint seed mismatch")

    model = VectorField(
        dim=int(checkpoint["dim"]),
        hidden=int(checkpoint["hidden"]),
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model


def neural_derivatives(model, states):
    chunks = []

    with torch.no_grad():
        for chunk in np.array_split(
            states, max(1, int(np.ceil(len(states) / 512)))
        ):
            tensor = torch.as_tensor(
                chunk, dtype=torch.float32
            )

            result = model.physical_field(tensor)
            chunks.append(
                result.cpu().numpy().astype(np.float64)
            )

    result = np.concatenate(chunks, axis=0)

    if not np.isfinite(result).all():
        raise ValueError("Non-finite neural derivatives")

    return result


def validation_segments(x, t):
    """Fixed, evenly spaced segments from every validation trajectory."""
    n_trajectories, n_times, dim = x.shape

    if n_times < HORIZON_POINTS:
        raise ValueError("Not enough validation time points")

    starts = np.unique(
        np.linspace(
            0,
            n_times - HORIZON_POINTS,
            SEGMENTS_PER_TRAJECTORY,
            dtype=int,
        )
    )

    segments = []

    for trajectory in range(n_trajectories):
        for start in starts:
            segments.append(
                x[
                    trajectory,
                    start:start + HORIZON_POINTS,
                    :
                ]
            )

    targets = np.asarray(segments, dtype=np.float64)

    # All trajectories share the same uniform time grid.
    dt = float(np.mean(np.diff(t)))

    if not np.allclose(
        np.diff(t), dt, rtol=1e-5, atol=1e-9
    ):
        raise ValueError("Nonuniform time grid is unsupported")

    relative_t = np.arange(HORIZON_POINTS) * dt

    return targets, relative_t


def trajectory_error(library, coefficients, targets, t):
    """Integrate all validation segments in one batched IVP."""
    n_segments, n_times, dim = targets.shape

    initial_states = targets[:, 0, :].copy()

    def rhs(_time, flattened):
        states = flattened.reshape(n_segments, dim)
        derivatives = library.transform(states) @ coefficients

        if not np.isfinite(derivatives).all():
            raise FloatingPointError(
                "Non-finite derivatives during integration"
            )

        return derivatives.reshape(-1)

    try:
        solution = solve_ivp(
            rhs,
            (float(t[0]), float(t[-1])),
            initial_states.reshape(-1),
            t_eval=t,
            method="RK45",
            rtol=RTOL,
            atol=ATOL,
            max_step=float(t[1] - t[0]),
        )

        if not solution.success:
            return float("inf"), n_segments

        if solution.y.shape[1] != n_times:
            return float("inf"), n_segments

        prediction = (
            solution.y.T.reshape(n_times, n_segments, dim)
            .transpose(1, 0, 2)
        )

        if not np.isfinite(prediction).all():
            return float("inf"), n_segments

        relative_l2 = float(
            np.linalg.norm(prediction - targets)
            / max(np.linalg.norm(targets), 1e-12)
        )

        return relative_l2, 0

    except (
        RuntimeError,
        ValueError,
        OverflowError,
        FloatingPointError,
    ):
        return float("inf"), n_segments


def format_equations(names, coef):
    equations = []

    for output in range(coef.shape[1]):
        terms = []

        for name, value in zip(names, coef[:, output]):
            if abs(value) > 1e-10:
                terms.append(f"({value:+.6f})*{name}")

        expression = " ".join(terms) if terms else "0"
        equations.append(f"dx{output}/dt = {expression}")

    return equations


def run_seed(system, seed, cfg):
    seed_all(seed)

    print(f"\n=== SYMBOLIC DISTILLATION: SEED {seed} ===")

    train, _ = load_split(system, seed, "train")
    val, val_t = load_split(system, seed, "val")

    model = load_neural_model(system, seed)

    all_states = train.reshape(-1, train.shape[-1])

    rng = np.random.default_rng(seed)
    n_samples = min(
        int(cfg["discovery"]["sample_points"]),
        len(all_states),
    )

    indices = rng.choice(
        len(all_states), size=n_samples, replace=False
    )

    x_fit = all_states[indices]
    y_fit = neural_derivatives(model, x_fit)

    targets, relative_t = validation_segments(val, val_t)

    print(
        f"Fit states: {len(x_fit)} | "
        f"Validation segments: {len(targets)}"
    )

    rows = []
    best = None

    for degree in DEGREES:
        for threshold in THRESHOLDS:
            library, coefficients = stlsq(
                x_fit,
                y_fit,
                degree=degree,
                threshold=threshold,
                ridge=float(cfg["discovery"]["ridge"]),
            )

            terms = int(
                np.count_nonzero(
                    np.abs(coefficients) > 1e-10
                )
            )

            error, failed = trajectory_error(
                library, coefficients, targets, relative_t
            )

            score = (
                error + COMPLEXITY_PENALTY * terms
                if np.isfinite(error)
                else float("inf")
            )

            row = {
                "seed": seed,
                "degree": degree,
                "threshold": threshold,
                "n_terms": terms,
                "trajectory_relative_l2": error,
                "failed_segments": failed,
                "score": score,
            }

            rows.append(row)

            print(
                f"degree={degree} "
                f"threshold={threshold:.2f} "
                f"terms={terms:2d} "
                f"traj_L2={error:.6f} "
                f"score={score:.6f}"
            )

            if (
                np.isfinite(score)
                and (best is None or score < best["score"])
            ):
                best = {
                    **row,
                    "coef": coefficients.copy(),
                    "names": library.get_feature_names_out(
                        [f"x{i}" for i in range(x_fit.shape[1])]
                    ),
                }

    if best is None:
        raise RuntimeError(
            f"No valid symbolic candidate for seed {seed}"
        )

    output = (
        ROOT / "results" / "benchmark"
        / system / "symbolic"
    )
    output.mkdir(parents=True, exist_ok=True)

    grid_path = output / f"seed_{seed}_grid.csv"

    with grid_path.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)

    npz_path = output / f"seed_{seed}_best.npz"

    np.savez_compressed(
        npz_path,
        coef=best["coef"],
        names=np.asarray(best["names"], dtype=str),
        degree=best["degree"],
        threshold=best["threshold"],
        seed=seed,
    )

    equations = format_equations(
        best["names"], best["coef"]
    )

    summary = {
        key: value
        for key, value in best.items()
        if key not in ("coef", "names")
    }

    summary["equations"] = equations
    summary["selection"] = (
        "Validation trajectory relative L2 "
        "+ 0.005 per nonzero term"
    )
    summary["test_used"] = False

    summary_path = output / f"seed_{seed}_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"\n=== BEST MODEL: SEED {seed} ===")
    print(
        f"Degree={best['degree']} | "
        f"Threshold={best['threshold']} | "
        f"Terms={best['n_terms']}"
    )
    print(
        f"Validation trajectory L2="
        f"{best['trajectory_relative_l2']:.6f}"
    )

    for equation in equations:
        print(equation)

    print(f"Saved: {npz_path}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=None
    )
    args = parser.parse_args()

    cfg = config()
    system = cfg["system"]

    selected_seeds = (
        [args.seed]
        if args.seed is not None
        else SEEDS
    )

    results = []

    for seed in selected_seeds:
        results.append(run_seed(system, seed, cfg))

    print("\n=== MULTISEED DISTILLATION SUMMARY ===")

    for result in results:
        print(
            f"seed={result['seed']} "
            f"terms={result['n_terms']} "
            f"val_traj_L2="
            f"{result['trajectory_relative_l2']:.6f}"
        )

    print(
        "\nAll models selected without accessing test data."
    )


if __name__ == "__main__":
    main()
