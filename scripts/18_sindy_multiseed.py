
"""Experiment 18: Direct SINDy multiseed benchmark.

Fits sparse polynomial models from estimated derivatives of
noisy training trajectories. Selects models using validation
trajectory error, with the same selection criterion and
integration settings as experiment 17.

Test data is never accessed.

Examples:
    python scripts/18_sindy_multiseed.py --seed 101
    python scripts/18_sindy_multiseed.py
"""

import sys
import csv
import json
import argparse
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter

from dynadistill.utils.common import ROOT, config
from dynadistill.discovery.sparse import stlsq


SEEDS = [101, 202, 303, 404, 505]

ESTIMATORS = [
    "gradient",
    "savgol_w5",
    "savgol_w7",
    "savgol_w9",
    "savgol_w11",
    "savgol_w15",
    "savgol_w21",
]

DEGREES = [2, 3]
THRESHOLDS = [
    0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50
]

HORIZON_POINTS = 9
SEGMENTS_PER_TRAJECTORY = 8
COMPLEXITY_PENALTY = 0.005

RTOL = 1e-6
ATOL = 1e-8

# Safety limit for obviously divergent candidate trajectories.
STATE_LIMIT = 1e4


def load_split(system, seed, split):
    path = (
        ROOT / "data" / "benchmark"
        / system / f"seed_{seed}" / f"{split}.npz"
    )

    with np.load(path) as data:
        x = np.asarray(data["x"], dtype=np.float64)
        t = np.asarray(data["t"], dtype=np.float64)

    if x.ndim != 3 or t.ndim != 1:
        raise ValueError(f"Invalid dimensions: {path}")

    if x.shape[1] != len(t):
        raise ValueError(f"Time dimension mismatch: {path}")

    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite states: {path}")

    if not np.isfinite(t).all():
        raise ValueError(f"Non-finite times: {path}")

    dt = np.diff(t)

    if np.any(dt <= 0):
        raise ValueError("Time must be strictly increasing")

    if not np.allclose(
        dt, dt[0], rtol=1e-5, atol=1e-9
    ):
        raise ValueError(
            "This experiment requires a uniform time grid"
        )

    return x, t


def estimate_derivatives(x, t, estimator):
    dt = float(t[1] - t[0])

    if estimator == "gradient":
        result = np.gradient(
            x, dt, axis=1, edge_order=2
        )

    elif estimator.startswith("savgol_w"):
        window = int(estimator.split("_w")[1])

        if window > x.shape[1]:
            raise ValueError(
                f"Window {window} exceeds trajectory length"
            )

        result = savgol_filter(
            x,
            window_length=window,
            polyorder=3,
            deriv=1,
            delta=dt,
            axis=1,
            mode="interp",
        )

    else:
        raise ValueError(
            f"Unknown derivative estimator: {estimator}"
        )

    if not np.isfinite(result).all():
        raise ValueError(
            f"Non-finite derivatives: {estimator}"
        )

    return result


def make_validation_segments(x, t):
    """Replicates the validation windows of experiment 17."""

    n_trajectories, n_times, dim = x.shape

    if n_times < HORIZON_POINTS:
        raise ValueError(
            "Insufficient validation trajectory length"
        )

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

    targets = np.asarray(
        segments, dtype=np.float64
    )

    dt = float(np.mean(np.diff(t)))
    relative_t = np.arange(HORIZON_POINTS) * dt

    return targets, relative_t


def trajectory_error(
    library, coefficients, targets, t
):
    """Validation trajectory error using batched integration.

    Same solve_ivp settings as experiment 17.
    """

    n_segments, n_times, dim = targets.shape

    initial_states = targets[:, 0, :].copy()

    def rhs(_time, flattened):
        states = flattened.reshape(n_segments, dim)

        derivatives = (
            library.transform(states) @ coefficients
        )

        if not np.isfinite(derivatives).all():
            raise FloatingPointError(
                "Non-finite model derivatives"
            )

        return derivatives.reshape(-1)

    def divergence_event(_time, flattened):
        maximum = float(
            np.max(np.abs(flattened))
        )
        return STATE_LIMIT - maximum

    divergence_event.terminal = True
    divergence_event.direction = -1

    try:
        solution = solve_ivp(
            rhs,
            (
                float(t[0]),
                float(t[-1]),
            ),
            initial_states.reshape(-1),
            t_eval=t,
            method="RK45",
            rtol=RTOL,
            atol=ATOL,
            max_step=float(t[1] - t[0]),
            events=divergence_event,
        )

        if not solution.success:
            return float("inf"), n_segments

        if solution.y.shape[1] != n_times:
            return float("inf"), n_segments

        prediction = (
            solution.y.T
            .reshape(n_times, n_segments, dim)
            .transpose(1, 0, 2)
        )

        if not np.isfinite(prediction).all():
            return float("inf"), n_segments

        error = float(
            np.linalg.norm(prediction - targets)
            / max(
                np.linalg.norm(targets),
                1e-12,
            )
        )

        return error, 0

    except (
        RuntimeError,
        ValueError,
        OverflowError,
        FloatingPointError,
    ):
        return float("inf"), n_segments


def format_equations(names, coefficients):
    equations = []

    for output in range(coefficients.shape[1]):
        terms = []

        for name, value in zip(
            names, coefficients[:, output]
        ):
            if abs(value) > 1e-10:
                terms.append(
                    f"({value:+.6f})*{name}"
                )

        expression = (
            " ".join(terms) if terms else "0"
        )

        equations.append(
            f"dx{output}/dt = {expression}"
        )

    return equations


def run_seed(system, seed, cfg):
    print(
        f"\n=== DIRECT SINDy: SEED {seed} ==="
    )

    train, train_t = load_split(
        system, seed, "train"
    )

    val, val_t = load_split(
        system, seed, "val"
    )

    targets, relative_t = (
        make_validation_segments(val, val_t)
    )

    x_fit = train.reshape(
        -1, train.shape[-1]
    )

    rows = []
    best = None

    total_candidates = (
        len(ESTIMATORS)
        * len(DEGREES)
        * len(THRESHOLDS)
    )

    print(
        f"Training states: {len(x_fit)}"
    )
    print(
        f"Validation segments: {len(targets)}"
    )
    print(
        f"Candidates: {total_candidates}"
    )

    for estimator in ESTIMATORS:
        derivatives = estimate_derivatives(
            train, train_t, estimator
        )

        y_fit = derivatives.reshape(
            -1, derivatives.shape[-1]
        )

        estimator_best = None

        for degree in DEGREES:
            for threshold in THRESHOLDS:
                library, coefficients = stlsq(
                    x_fit,
                    y_fit,
                    degree=degree,
                    threshold=threshold,
                    ridge=float(
                        cfg["discovery"]["ridge"]
                    ),
                )

                terms = int(
                    np.count_nonzero(
                        np.abs(coefficients) > 1e-10
                    )
                )

                error, failed = trajectory_error(
                    library,
                    coefficients,
                    targets,
                    relative_t,
                )

                score = (
                    error
                    + COMPLEXITY_PENALTY * terms
                    if np.isfinite(error)
                    else float("inf")
                )

                row = {
                    "seed": seed,
                    "estimator": estimator,
                    "degree": degree,
                    "threshold": threshold,
                    "n_terms": terms,
                    "trajectory_relative_l2": error,
                    "failed_segments": failed,
                    "score": score,
                }

                rows.append(row)

                if np.isfinite(score):
                    candidate = {
                        **row,
                        "coef": coefficients.copy(),
                        "names": (
                            library.get_feature_names_out(
                                [
                                    f"x{i}"
                                    for i in range(
                                        x_fit.shape[1]
                                    )
                                ]
                            )
                        ),
                    }

                    if (
                        estimator_best is None
                        or score < estimator_best["score"]
                    ):
                        estimator_best = candidate

                    if (
                        best is None
                        or score < best["score"]
                    ):
                        best = candidate

        if estimator_best is None:
            print(
                f"{estimator:15s} "
                f"no valid candidate"
            )
        else:
            e = estimator_best
            print(
                f"{estimator:15s} "
                f"degree={e['degree']} "
                f"threshold={e['threshold']:.2f} "
                f"terms={e['n_terms']:2d} "
                f"val_L2="
                f"{e['trajectory_relative_l2']:.6f} "
                f"score={e['score']:.6f}"
            )

    if best is None:
        raise RuntimeError(
            f"No valid SINDy model for seed {seed}"
        )

    output = (
        ROOT / "results" / "benchmark"
        / system / "sindy"
    )

    output.mkdir(
        parents=True, exist_ok=True
    )

    grid_path = (
        output / f"seed_{seed}_grid.csv"
    )

    with grid_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    model_path = (
        output / f"seed_{seed}_best.npz"
    )

    np.savez_compressed(
        model_path,
        coef=best["coef"],
        names=np.asarray(
            best["names"], dtype=str
        ),
        degree=best["degree"],
        threshold=best["threshold"],
        estimator=best["estimator"],
        seed=seed,
    )

    equations = format_equations(
        best["names"],
        best["coef"],
    )

    summary = {
        key: value
        for key, value in best.items()
        if key not in ("coef", "names")
    }

    summary["equations"] = equations
    summary["test_used"] = False
    summary["selection"] = (
        "Validation trajectory relative L2 "
        "+ 0.005 per nonzero term"
    )

    summary_path = (
        output / f"seed_{seed}_summary.json"
    )

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(
        f"\n=== BEST DIRECT SINDy: SEED {seed} ==="
    )
    print(
        f"Estimator: {best['estimator']}"
    )
    print(
        f"Degree: {best['degree']}"
    )
    print(
        f"Threshold: {best['threshold']}"
    )
    print(
        f"Terms: {best['n_terms']}"
    )
    print(
        "Validation trajectory L2: "
        f"{best['trajectory_relative_l2']:.6f}"
    )
    print(
        f"Score: {best['score']:.6f}"
    )

    for equation in equations:
        print(equation)

    print(f"Saved: {model_path}")

    # Comparison with the already frozen experiment 17.
    distilled_path = (
        ROOT / "results" / "benchmark"
        / system / "symbolic"
        / f"seed_{seed}_summary.json"
    )

    if distilled_path.exists():
        distilled = json.loads(
            distilled_path.read_text(
                encoding="utf-8"
            )
        )

        print(
            "\n--- VALIDATION COMPARISON ---"
        )

        print(
            "Direct SINDy: "
            f"{best['trajectory_relative_l2']:.6f}"
        )

        print(
            "Neural distillation: "
            f"{distilled['trajectory_relative_l2']:.6f}"
        )

        print(
            "These are validation results, "
            "not final test results."
        )

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
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
        results.append(
            run_seed(system, seed, cfg)
        )

    print(
        "\n=== DIRECT SINDy MULTISEED SUMMARY ==="
    )

    for result in results:
        print(
            f"seed={result['seed']} "
            f"estimator={result['estimator']} "
            f"degree={result['degree']} "
            f"terms={result['n_terms']} "
            f"val_L2="
            f"{result['trajectory_relative_l2']:.6f}"
        )

    if len(results) > 1:
        errors = np.asarray(
            [
                result["trajectory_relative_l2"]
                for result in results
            ],
            dtype=float,
        )

        print(
            "\nMean validation L2: "
            f"{errors.mean():.6f}"
        )

        print(
            "Sample standard deviation: "
            f"{errors.std(ddof=1):.6f}"
        )

    print(
        "\nTest data was not accessed."
    )


if __name__ == "__main__":
    main()
