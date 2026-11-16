
"""Experiment 22: Symbolic discovery for paired noise robustness.

Fits neural distillation and direct SINDy on TRAIN data.
Selects candidates using noisy VALIDATION trajectories only.
Does not access TEST data.

Examples:
    python scripts/22_discover_noise_robustness.py --seed 101 --noise noise_000
    python scripts/22_discover_noise_robustness.py --seed 101
    python scripts/22_discover_noise_robustness.py --all
"""

import sys
import json
import csv
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
from sklearn.preprocessing import PolynomialFeatures

from dynadistill.utils.common import ROOT, config
from dynadistill.models.neuralode import VectorField
from dynadistill.discovery.sparse import stlsq


SEEDS = [101, 202, 303, 404, 505]
NOISES = [
    "noise_000",
    "noise_001",
    "noise_005",
    "noise_010",
    "noise_020",
]

DEGREES = [2, 3]
THRESHOLDS = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
ESTIMATORS = [
    "gradient",
    "savgol_w5",
    "savgol_w7",
    "savgol_w9",
    "savgol_w11",
    "savgol_w15",
    "savgol_w21",
]

HORIZON = 9
WINDOWS = 8
COMPLEXITY_WEIGHT = 0.005
MAX_STATE = 10000.0
FAILURE_SCORE = 1e6


def read_split(system, noise, seed, split):
    path = (
        ROOT / "data" / "robustness"
        / system / noise / f"seed_{seed}" / f"{split}.npz"
    )

    with np.load(path) as data:
        x = np.asarray(data["x"], dtype=np.float64)
        t = np.asarray(data["t"], dtype=np.float64)

    if x.ndim != 3 or t.ndim != 1:
        raise ValueError(f"Invalid data shape: {path}")

    if x.shape[1] != len(t):
        raise ValueError(f"Time dimension mismatch: {path}")

    if not np.isfinite(x).all() or not np.isfinite(t).all():
        raise ValueError(f"Non-finite data: {path}")

    dt = np.diff(t)

    if np.any(dt <= 0):
        raise ValueError(f"Non-increasing times: {path}")

    if not np.allclose(dt, dt[0], rtol=1e-5, atol=1e-7):
        raise ValueError(f"Nonuniform time grid: {path}")

    return x, t


def load_neural(system, noise, seed):
    path = (
        ROOT / "results" / "robustness_paired"
        / system / noise / f"seed_{seed}" / "neural_ode.pt"
    )

    # weights_only=False is used for locally generated,
    # trusted checkpoints containing metadata.
    checkpoint = torch.load(
        path, map_location="cpu", weights_only=False
    )

    if checkpoint.get("version") != "paired_v1":
        raise ValueError(f"Unexpected checkpoint protocol: {path}")

    if checkpoint["seed"] != seed:
        raise ValueError("Checkpoint seed mismatch")

    if checkpoint["noise_label"] != noise:
        raise ValueError("Checkpoint noise mismatch")

    model = VectorField(
        dim=int(checkpoint["dim"]),
        hidden=int(checkpoint["hidden"]),
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, checkpoint


def estimate_derivatives(x, t, method):
    dt = float(np.mean(np.diff(t)))

    if method == "gradient":
        return np.gradient(
            x, t, axis=1, edge_order=2
        )

    if method.startswith("savgol_w"):
        window = int(method.split("w")[-1])

        if window > x.shape[1]:
            raise ValueError("Savitzky-Golay window too large")

        return savgol_filter(
            x,
            window_length=window,
            polyorder=3,
            deriv=1,
            delta=dt,
            axis=1,
            mode="interp",
        )

    raise ValueError(f"Unknown estimator: {method}")


def validation_windows(x, t):
    starts = np.unique(
        np.linspace(
            0,
            x.shape[1] - HORIZON,
            WINDOWS,
            dtype=int,
        )
    )

    initial_states = []
    targets = []

    for trajectory in x:
        for start in starts:
            segment = trajectory[start:start + HORIZON]
            initial_states.append(segment[0])
            targets.append(segment)

    return (
        np.asarray(initial_states, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
        t[:HORIZON] - t[0],
    )


def integrate_candidate(library, coef, x0, times):
    """Integrate all validation windows in one solve_ivp call."""

    n_segments, dim = x0.shape
    exponents = library.powers_.astype(np.int64)

    def rhs(_, flattened):
        states = flattened.reshape(n_segments, dim)

        if not np.isfinite(states).all():
            return np.zeros_like(flattened)

        if np.max(np.abs(states)) > MAX_STATE:
            return np.zeros_like(flattened)

        theta = np.prod(
            states[:, None, :] ** exponents[None, :, :],
            axis=2,
        )

        derivatives = theta @ coef

        if not np.isfinite(derivatives).all():
            return np.zeros_like(flattened)

        return derivatives.reshape(-1)

    def escape_event(_, flattened):
        return MAX_STATE - np.max(np.abs(flattened))

    escape_event.terminal = True
    escape_event.direction = -1

    dt = float(times[1] - times[0])

    try:
        with np.errstate(over="ignore", invalid="ignore"):
            sol = solve_ivp(
                rhs,
                (float(times[0]), float(times[-1])),
                x0.reshape(-1),
                t_eval=times,
                method="RK45",
                rtol=1e-6,
                atol=1e-8,
                max_step=dt,
                events=escape_event,
            )

        if not sol.success or len(sol.t) != len(times):
            return None

        prediction = sol.y.T.reshape(
            len(times), n_segments, dim
        ).transpose(1, 0, 2)

        if not np.isfinite(prediction).all():
            return None

        if np.max(np.abs(prediction)) >= MAX_STATE:
            return None

        return prediction

    except (ValueError, FloatingPointError, OverflowError):
        return None


def score_candidate(library, coef, validation):
    x0, target, times = validation

    terms = int(np.count_nonzero(np.abs(coef) > 1e-10))

    prediction = integrate_candidate(
        library, coef, x0, times
    )

    if prediction is None:
        return FAILURE_SCORE, FAILURE_SCORE, terms

    denominator = max(
        float(np.linalg.norm(target)), 1e-12
    )

    error = float(
        np.linalg.norm(prediction - target)
        / denominator
    )

    if not np.isfinite(error):
        return FAILURE_SCORE, FAILURE_SCORE, terms

    score = error + COMPLEXITY_WEIGHT * terms

    return score, error, terms


def fit_distillation(model, train_x, sample_points):
    states = train_x.reshape(-1, train_x.shape[-1])

    rng = np.random.default_rng(12345)

    if len(states) > sample_points:
        indices = rng.choice(
            len(states), size=sample_points, replace=False
        )
        states = states[indices]

    with torch.no_grad():
        tensor = torch.tensor(states, dtype=torch.float32)
        derivatives = (
            model.physical_field(tensor).cpu().numpy()
        )

    return states, derivatives.astype(np.float64)


def fit_direct_sindy(train_x, train_t, estimator):
    derivatives = estimate_derivatives(
        train_x, train_t, estimator
    )

    return (
        train_x.reshape(-1, train_x.shape[-1]),
        derivatives.reshape(-1, train_x.shape[-1]),
    )


def save_best(output, best, method):
    np.savez_compressed(
        output / f"{method}_best.npz",
        coef=best["coef"],
        names=np.asarray(best["names"], dtype=str),
        degree=np.int64(best["degree"]),
        threshold=np.float64(best["threshold"]),
        estimator=np.asarray(best["estimator"]),
        terms=np.int64(best["terms"]),
        val_traj_l2=np.float64(best["val_traj_l2"]),
        score=np.float64(best["score"]),
    )

    summary = {
        key: value
        for key, value in best.items()
        if key not in ("coef", "names")
    }

    summary["equations"] = []

    for j in range(best["coef"].shape[1]):
        pieces = []

        for name, coefficient in zip(
            best["names"], best["coef"][:, j]
        ):
            if abs(coefficient) > 1e-10:
                pieces.append(
                    f"({coefficient:+.6f})*{name}"
                )

        summary["equations"].append(
            f"dx{j}/dt = "
            + (" ".join(pieces) if pieces else "0")
        )

    (output / f"{method}_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(
        f"BEST {method:19s} "
        f"degree={best['degree']} "
        f"threshold={best['threshold']:.2f} "
        f"estimator={best['estimator']:12s} "
        f"terms={best['terms']:2d} "
        f"val_L2={best['val_traj_l2']:.6f}"
    )

    for equation in summary["equations"]:
        print("  " + equation)


def run_method(
    method,
    train_x,
    train_t,
    validation,
    output,
    model,
    sample_points,
):
    grid_path = output / f"{method}_grid.csv"
    best_path = output / f"{method}_best.npz"

    if best_path.exists():
        print(f"SKIPPED {method}: model already selected")
        return

    best = None
    records = []

    estimators = (
        ["neural_field"]
        if method == "neural_distillation"
        else ESTIMATORS
    )

    for estimator in estimators:
        if method == "neural_distillation":
            states, derivatives = fit_distillation(
                model, train_x, sample_points
            )
        else:
            states, derivatives = fit_direct_sindy(
                train_x, train_t, estimator
            )

        for degree in DEGREES:
            for threshold in THRESHOLDS:
                try:
                    library, coef = stlsq(
                        states,
                        derivatives,
                        degree=degree,
                        threshold=threshold,
                        ridge=1e-5,
                    )

                    score, val_error, terms = score_candidate(
                        library, coef, validation
                    )

                except (
                    ValueError,
                    np.linalg.LinAlgError,
                    FloatingPointError,
                ):
                    continue

                record = {
                    "method": method,
                    "estimator": estimator,
                    "degree": degree,
                    "threshold": threshold,
                    "terms": terms,
                    "val_traj_l2": val_error,
                    "score": score,
                }

                records.append(record)

                if score >= FAILURE_SCORE:
                    continue

                if best is None or score < best["score"]:
                    best = {
                        **record,
                        "coef": coef.copy(),
                        "names": list(
                            library.get_feature_names_out()
                        ),
                    }

    with grid_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "method",
                "estimator",
                "degree",
                "threshold",
                "terms",
                "val_traj_l2",
                "score",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    if best is None:
        raise RuntimeError(
            f"No valid candidate for {method}"
        )

    save_best(output, best, method)


def run_condition(system, seed, noise, cfg):
    output = (
        ROOT / "results" / "robustness_paired"
        / system / noise / f"seed_{seed}"
        / "symbolic"
    )
    output.mkdir(parents=True, exist_ok=True)

    print(
        f"\n=== SYMBOLIC DISCOVERY ===\n"
        f"system={system} seed={seed} noise={noise}"
    )

    train_x, train_t = read_split(
        system, noise, seed, "train"
    )
    val_x, val_t = read_split(
        system, noise, seed, "val"
    )

    if not np.allclose(train_t, val_t):
        raise ValueError("Train/validation grids differ")

    model, checkpoint = load_neural(
        system, noise, seed
    )

    validation = validation_windows(val_x, val_t)

    sample_points = int(
        cfg.get("discovery", {}).get(
            "sample_points", 3000
        )
    )

    run_method(
        "neural_distillation",
        train_x,
        train_t,
        validation,
        output,
        model,
        sample_points,
    )

    run_method(
        "direct_sindy",
        train_x,
        train_t,
        validation,
        output,
        model,
        sample_points,
    )

    print(
        f"Completed seed={seed} noise={noise}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", type=int, default=101
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

    args = parser.parse_args()
    cfg = config()
    system = cfg["system"]

    if args.all:
        seeds = SEEDS
        noises = NOISES
    else:
        if args.seed not in SEEDS:
            parser.error(f"Choose a seed from {SEEDS}")

        seeds = [args.seed]
        noises = (
            [args.noise]
            if args.noise is not None
            else NOISES
        )

    for seed in seeds:
        for noise in noises:
            run_condition(system, seed, noise, cfg)

    print("\n=== EXPERIMENT 22 COMPLETE ===")
    print("Selection used train/validation only.")
    print("Test data was not accessed.")


if __name__ == "__main__":
    main()
