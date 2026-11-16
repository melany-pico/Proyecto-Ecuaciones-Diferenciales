
"""Experiment 23: Frozen test evaluation of noise robustness.

Evaluate 25 conditions x 3 frozen models:
- Neural ODE
- Neural distillation
- Direct SINDy

No retraining, no hyperparameter tuning.
Test trajectories are accessed only after the checkpoint preflight.

Run:
    python scripts/23_evaluate_noise_robustness.py --preflight
    python scripts/23_evaluate_noise_robustness.py
"""

import sys
import json
import csv
import argparse
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from scipy.integrate import solve_ivp
from sklearn.preprocessing import PolynomialFeatures

from dynadistill.utils.common import ROOT, config
from dynadistill.models.neuralode import VectorField
from dynadistill.data.systems import field


SEEDS = [101, 202, 303, 404, 505]

NOISES = {
    "noise_000": 0.00,
    "noise_001": 0.01,
    "noise_005": 0.05,
    "noise_010": 0.10,
    "noise_020": 0.20,
}

METHODS = [
    "direct_sindy",
    "neural_distillation",
    "neural_ode",
]

HORIZON = 9
WINDOWS = 8
MAX_STATE = 10000.0

# The seven true nonzero Lorenz coefficients:
# dx0: -10*x0 + 10*x1
# dx1: 28*x0 - x1 - x0*x2
# dx2: x0*x1 - (8/3)*x2
TRUE_SUPPORT = {
    (0, "x0"),
    (0, "x1"),
    (1, "x0"),
    (1, "x1"),
    (1, "x0 x2"),
    (2, "x0 x1"),
    (2, "x2"),
}


def condition_dir(system, noise, seed):
    return (
        ROOT / "results" / "robustness_paired"
        / system / noise / f"seed_{seed}"
    )


def symbolic_path(system, noise, seed, method):
    return (
        condition_dir(system, noise, seed)
        / "symbolic" / f"{method}_best.npz"
    )


def neural_path(system, noise, seed):
    return (
        condition_dir(system, noise, seed)
        / "neural_ode.pt"
    )


def test_path(system, noise, seed):
    return (
        ROOT / "data" / "robustness"
        / system / noise / f"seed_{seed}"
        / "test.npz"
    )


def preflight(system):
    """Check files without reading any test observations."""
    missing = []

    for seed in SEEDS:
        for noise in NOISES:
            paths = [
                neural_path(system, noise, seed),
                symbolic_path(
                    system, noise, seed,
                    "direct_sindy",
                ),
                symbolic_path(
                    system, noise, seed,
                    "neural_distillation",
                ),
                test_path(system, noise, seed),
            ]

            for path in paths:
                if not path.is_file():
                    missing.append(str(path))

    if missing:
        print("\nMISSING FILES:")
        for path in missing:
            print(" ", path)
        raise FileNotFoundError(
            f"{len(missing)} required files are missing"
        )

    print(
        "PREFLIGHT OK: 25 neural checkpoints, "
        "50 symbolic models, 25 test files present."
    )
    print("Test data has not been read.")


def load_test(system, noise, seed):
    path = test_path(system, noise, seed)

    with np.load(path) as data:
        observed = np.asarray(
            data["x"], dtype=np.float64
        )
        clean = np.asarray(
            data["x_clean"], dtype=np.float64
        )
        times = np.asarray(
            data["t"], dtype=np.float64
        )

    if observed.shape != clean.shape:
        raise ValueError(
            f"Observed/clean shape mismatch: {path}"
        )

    if observed.ndim != 3:
        raise ValueError(f"Invalid trajectories: {path}")

    if observed.shape[1] != len(times):
        raise ValueError(f"Invalid time grid: {path}")

    if not (
        np.isfinite(observed).all()
        and np.isfinite(clean).all()
        and np.isfinite(times).all()
    ):
        raise ValueError(f"Non-finite test data: {path}")

    dt = np.diff(times)
    if np.any(dt <= 0):
        raise ValueError(f"Non-increasing times: {path}")

    if not np.allclose(
        dt, dt[0], rtol=1e-5, atol=1e-7
    ):
        raise ValueError(f"Nonuniform time grid: {path}")

    return observed, clean, times


def make_test_windows(observed, clean, times):
    """Use identical noisy initial states for all methods."""

    n_traj, n_times, dim = observed.shape

    if HORIZON > n_times:
        raise ValueError("Horizon exceeds trajectory")

    starts = np.unique(
        np.linspace(
            0,
            n_times - HORIZON,
            WINDOWS,
            dtype=int,
        )
    )

    initial = []
    targets = []

    for trajectory in range(n_traj):
        for start in starts:
            initial.append(
                observed[trajectory, start]
            )
            targets.append(
                clean[
                    trajectory,
                    start:start + HORIZON,
                    :,
                ]
            )

    relative_times = (
        times[:HORIZON] - times[0]
    )

    return (
        np.asarray(initial, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
        relative_times,
    )


def load_neural(system, noise, seed):
    checkpoint = torch.load(
        neural_path(system, noise, seed),
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint.get("version") != "paired_v1":
        raise ValueError(
            "Neural checkpoint protocol mismatch"
        )

    if (
        checkpoint["seed"] != seed
        or checkpoint["noise_label"] != noise
        or checkpoint["system"] != system
    ):
        raise ValueError(
            "Neural checkpoint condition mismatch"
        )

    model = VectorField(
        dim=int(checkpoint["dim"]),
        hidden=int(checkpoint["hidden"]),
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )
    model.eval()

    return model


def load_symbolic(system, noise, seed, method):
    path = symbolic_path(
        system, noise, seed, method
    )

    with np.load(path) as data:
        coef = np.asarray(
            data["coef"], dtype=np.float64
        )
        degree = int(data["degree"])
        names = [
            str(value)
            for value in data["names"]
        ]

    if not np.isfinite(coef).all():
        raise ValueError(
            f"Non-finite symbolic coefficients: {path}"
        )

    library = PolynomialFeatures(
        degree=degree,
        include_bias=True,
    )

    # The fitted library is reconstructed using a
    # dummy sample solely to establish its feature order.
    dim = coef.shape[1]
    library.fit(np.zeros((1, dim)))

    expected_names = list(
        library.get_feature_names_out()
    )

    if names != expected_names:
        raise ValueError(
            f"Polynomial feature order mismatch: {path}"
        )

    if coef.shape != (
        len(expected_names), dim
    ):
        raise ValueError(
            f"Invalid coefficient shape: {path}"
        )

    return library, coef, names


def symbolic_field(states, library, coef):
    shape = states.shape
    flattened = states.reshape(-1, shape[-1])

    theta = library.transform(flattened)
    predicted = theta @ coef

    return predicted.reshape(shape)


def integrate_symbolic(library, coef, initial, times):
    """Batch integration with divergence detection."""

    n_windows, dim = initial.shape
    powers = library.powers_.astype(np.int64)

    def rhs(_, flat):
        states = flat.reshape(n_windows, dim)

        with np.errstate(
            over="ignore",
            invalid="ignore",
        ):
            theta = np.prod(
                states[:, None, :]
                ** powers[None, :, :],
                axis=2,
            )
            derivative = theta @ coef

        if not np.isfinite(derivative).all():
            raise FloatingPointError(
                "Non-finite symbolic derivative"
            )

        return derivative.reshape(-1)

    def divergence_event(_, flat):
        return (
            MAX_STATE
            - np.max(np.abs(flat))
        )

    divergence_event.terminal = True
    divergence_event.direction = -1

    dt = float(times[1] - times[0])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", RuntimeWarning
            )

            solution = solve_ivp(
                rhs,
                (
                    float(times[0]),
                    float(times[-1]),
                ),
                initial.reshape(-1),
                t_eval=times,
                method="RK45",
                rtol=1e-6,
                atol=1e-8,
                max_step=dt,
                events=divergence_event,
            )

        if (
            not solution.success
            or len(solution.t) != len(times)
        ):
            return None

        predicted = (
            solution.y.T
            .reshape(len(times), n_windows, dim)
            .transpose(1, 0, 2)
        )

        if (
            not np.isfinite(predicted).all()
            or np.max(np.abs(predicted)) >= MAX_STATE
        ):
            return None

        return predicted

    except (
        ValueError,
        FloatingPointError,
        OverflowError,
    ):
        return None


def integrate_neural(model, initial, times):
    try:
        x0 = torch.tensor(
            initial, dtype=torch.float32
        )

        t = torch.tensor(
            times, dtype=torch.float32
        )

        with torch.no_grad():
            prediction = (
                model.physical_trajectory(x0, t)
                .cpu().numpy()
            )

        if (
            not np.isfinite(prediction).all()
            or np.max(np.abs(prediction)) >= MAX_STATE
        ):
            return None

        return prediction.astype(np.float64)

    except (
        ValueError,
        RuntimeError,
        FloatingPointError,
    ):
        return None


def relative_l2(prediction, target):
    denominator = max(
        float(np.linalg.norm(target)), 1e-12
    )
    return float(
        np.linalg.norm(prediction - target)
        / denominator
    )


def support_metrics(names, coef):
    estimated = set()

    for output_index in range(coef.shape[1]):
        for feature_index, name in enumerate(names):
            if abs(
                coef[feature_index, output_index]
            ) > 1e-10:
                estimated.add(
                    (output_index, name)
                )

    tp = len(estimated & TRUE_SUPPORT)
    fp = len(estimated - TRUE_SUPPORT)
    fn = len(TRUE_SUPPORT - estimated)

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )
    recall = tp / (tp + fn)

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "terms": len(estimated),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_condition(system, noise, sigma, seed):
    observed, clean, times = load_test(
        system, noise, seed
    )

    initial, target, relative_times = (
        make_test_windows(
            observed, clean, times
        )
    )

    # Evaluate field accuracy only on clean test
    # states, against the known Lorenz equations.
    clean_states = clean.reshape(
        -1, clean.shape[-1]
    )

    true_field = field(
        system, clean_states
    )

    neural = load_neural(
        system, noise, seed
    )

    symbolic_models = {}

    for method in [
        "direct_sindy",
        "neural_distillation",
    ]:
        symbolic_models[method] = (
            load_symbolic(
                system, noise, seed, method
            )
        )

    rows = []

    for method in METHODS:
        if method == "neural_ode":
            prediction = integrate_neural(
                neural, initial, relative_times
            )

            with torch.no_grad():
                predicted_field = (
                    neural.physical_field(
                        torch.tensor(
                            clean_states,
                            dtype=torch.float32,
                        )
                    )
                    .cpu().numpy()
                    .astype(np.float64)
                )

            structure = {
                "terms": None,
                "tp": None,
                "fp": None,
                "fn": None,
                "precision": None,
                "recall": None,
                "f1": None,
            }

        else:
            library, coef, names = (
                symbolic_models[method]
            )

            prediction = integrate_symbolic(
                library,
                coef,
                initial,
                relative_times,
            )

            predicted_field = symbolic_field(
                clean_states, library, coef
            )

            structure = support_metrics(
                names, coef
            )

        failed = (
            prediction is None
            or not np.isfinite(
                predicted_field
            ).all()
        )

        trajectory_error = (
            relative_l2(prediction, target)
            if prediction is not None
            else float("nan")
        )

        field_error = (
            relative_l2(
                predicted_field, true_field
            )
            if np.isfinite(
                predicted_field
            ).all()
            else float("nan")
        )

        row = {
            "seed": seed,
            "noise_label": noise,
            "noise_std": sigma,
            "method": method,
            "trajectory_l2": trajectory_error,
            "field_l2": field_error,
            "failed": int(failed),
            "n_test_trajectories": int(
                len(clean)
            ),
            "n_windows": int(len(initial)),
            "horizon_points": HORIZON,
            **structure,
        }

        rows.append(row)

        print(
            f"{method:20s} "
            f"traj_L2={trajectory_error:.6f} "
            f"field_L2={field_error:.6f} "
            f"failed={int(failed)} "
            f"terms={structure['terms']}"
        )

    return rows


def mean_std(values):
    values = np.asarray(
        values, dtype=np.float64
    )

    if not np.isfinite(values).all():
        return float("nan"), float("nan")

    return (
        float(values.mean()),
        float(
            values.std(ddof=1)
            if len(values) > 1
            else 0.0
        ),
    )


def summarize(rows):
    summary = []

    for noise, sigma in NOISES.items():
        for method in METHODS:
            subset = [
                row for row in rows
                if row["noise_label"] == noise
                and row["method"] == method
            ]

            if len(subset) != len(SEEDS):
                raise ValueError(
                    "Incomplete test results: "
                    f"{noise}, {method}"
                )

            traj_mean, traj_std = mean_std(
                [r["trajectory_l2"] for r in subset]
            )

            field_mean, field_std = mean_std(
                [r["field_l2"] for r in subset]
            )

            f1_values = [
                r["f1"] for r in subset
                if r["f1"] is not None
            ]

            f1_mean, f1_std = (
                mean_std(f1_values)
                if f1_values
                else (None, None)
            )

            summary.append({
                "noise_label": noise,
                "noise_std": sigma,
                "method": method,
                "n_seeds": len(subset),
                "trajectory_l2_mean": traj_mean,
                "trajectory_l2_std": traj_std,
                "field_l2_mean": field_mean,
                "field_l2_std": field_std,
                "f1_mean": f1_mean,
                "f1_std": f1_std,
                "failed_conditions": sum(
                    r["failed"] for r in subset
                ),
            })

    return summary


def save_csv(path, rows):
    if not rows:
        raise ValueError("No results to save")

    with path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check all inputs without reading test data",
    )

    args = parser.parse_args()
    cfg = config()
    system = cfg["system"]

    if system != "lorenz":
        raise ValueError(
            "This evaluation currently supports "
            "Lorenz only"
        )

    print(
        "\n=== EXPERIMENT 23: "
        "FROZEN NOISE ROBUSTNESS TEST ==="
    )

    preflight(system)

    if args.preflight:
        print(
            "\nPreflight completed. "
            "No test evaluation performed."
        )
        return

    output = (
        ROOT / "results"
        / "robustness_paired"
        / system / "evaluation"
    )
    output.mkdir(
        parents=True, exist_ok=True
    )

    results_path = output / "test_results.csv"
    summary_csv_path = output / "test_summary.csv"
    summary_json_path = output / "test_summary.json"

    # Prevent accidental repeat testing and overwrites.
    for path in [
        results_path,
        summary_csv_path,
        summary_json_path,
    ]:
        if path.exists():
            raise FileExistsError(
                "Frozen test output already exists. "
                "Refusing to overwrite: "
                f"{path}"
            )

    all_rows = []

    for noise, sigma in NOISES.items():
        for seed in SEEDS:
            print(
                f"\n=== TEST: noise={noise} "
                f"sigma={sigma:.3f} "
                f"seed={seed} ==="
            )

            condition_rows = evaluate_condition(
                system, noise, sigma, seed
            )

            all_rows.extend(condition_rows)

    summary = summarize(all_rows)

    # Reject incomplete or non-finite evaluations.
    # No output files are written in that case.
    for row in all_rows:
        if (
            row["failed"]
            or not np.isfinite(
                row["trajectory_l2"]
            )
            or not np.isfinite(
                row["field_l2"]
            )
        ):
            raise RuntimeError(
                "At least one model failed test "
                "integration or field evaluation. "
                "No final results saved."
            )

    save_csv(results_path, all_rows)
    save_csv(summary_csv_path, summary)

    summary_json_path.write_text(
        json.dumps(
            {
                "experiment": 23,
                "system": system,
                "protocol": "paired_v1",
                "horizon_points": HORIZON,
                "initial_conditions": (
                    "noisy observed states"
                ),
                "trajectory_reference": (
                    "clean test trajectories"
                ),
                "field_reference": (
                    "true Lorenz field at clean "
                    "test states"
                ),
                "results": summary,
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== FINAL TEST SUMMARY ==="
    )

    for item in summary:
        print(
            f"sigma={item['noise_std']:.2f} "
            f"{item['method']:20s} "
            f"traj={item['trajectory_l2_mean']:.6f}"
            f" +/- {item['trajectory_l2_std']:.6f} "
            f"field={item['field_l2_mean']:.6f}"
            f" +/- {item['field_l2_std']:.6f} "
            f"F1={item['f1_mean']}"
        )

    print(f"\nResults: {results_path}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Summary JSON: {summary_json_path}")
    print(
        "\n=== EXPERIMENT 23 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
