
"""Experiment 19: Frozen multiseed test evaluation.

Evaluates direct SINDy, Neural ODE and neural distillation.
No model fitting, tuning or checkpoint updates.

Run:
    python scripts/19_evaluate_multiseed.py
"""

import sys
import csv
import json
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
HORIZON = 9
SEGMENTS_PER_TRAJECTORY = 8

RTOL = 1e-6
ATOL = 1e-8
STATE_LIMIT = 1e4


def load_test(system, seed):
    path = (
        ROOT / "data" / "benchmark"
        / system / f"seed_{seed}" / "test.npz"
    )

    with np.load(path) as data:
        observed = np.asarray(
            data["x"], dtype=np.float64
        )
        clean = np.asarray(
            data["x_clean"], dtype=np.float64
        )
        t = np.asarray(
            data["t"], dtype=np.float64
        )

    return observed, clean, t


def make_segments(x, t):
    n_traj, n_times, dim = x.shape

    starts = np.unique(
        np.linspace(
            0,
            n_times - HORIZON,
            SEGMENTS_PER_TRAJECTORY,
            dtype=int,
        )
    )

    segments = np.asarray([
        x[i, start:start + HORIZON]
        for i in range(n_traj)
        for start in starts
    ])

    dt = float(np.mean(np.diff(t)))
    relative_t = np.arange(HORIZON) * dt

    return segments, relative_t


def load_symbolic(system, seed, method):
    path = (
        ROOT / "results" / "benchmark"
        / system / method / f"seed_{seed}_best.npz"
    )

    with np.load(path) as data:
        coefficients = np.asarray(
            data["coef"], dtype=np.float64
        )
        names = np.asarray(data["names"], dtype=str)
        degree = int(data["degree"])

    library = PolynomialFeatures(
        degree=degree, include_bias=True
    )

    dim = coefficients.shape[1]

    library.fit(np.zeros((1, dim)))

    expected = library.get_feature_names_out(
        [f"x{i}" for i in range(dim)]
    )

    if not np.array_equal(names, expected):
        raise ValueError(
            f"Polynomial feature mismatch: {path}"
        )

    return library, coefficients, names


def load_neural(system, seed):
    path = (
        ROOT / "results" / "benchmark"
        / system / f"neural_ode_seed_{seed}.pt"
    )

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
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


def relative_l2(prediction, target):
    return float(
        np.linalg.norm(prediction - target)
        / max(np.linalg.norm(target), 1e-12)
    )


def polynomial_trajectory_error(
    library, coefficients, targets, t
):
    n_segments, n_times, dim = targets.shape

    def rhs(_time, flat):
        states = flat.reshape(n_segments, dim)
        result = library.transform(states) @ coefficients

        if not np.isfinite(result).all():
            raise FloatingPointError(
                "Non-finite polynomial derivatives"
            )

        return result.reshape(-1)

    def divergence_event(_time, flat):
        return STATE_LIMIT - float(
            np.max(np.abs(flat))
        )

    divergence_event.terminal = True
    divergence_event.direction = -1

    try:
        solution = solve_ivp(
            rhs,
            (float(t[0]), float(t[-1])),
            targets[:, 0, :].reshape(-1),
            t_eval=t,
            method="RK45",
            rtol=RTOL,
            atol=ATOL,
            max_step=float(t[1] - t[0]),
            events=divergence_event,
        )

        if (
            not solution.success
            or solution.y.shape[1] != n_times
        ):
            return float("inf"), n_segments

        prediction = (
            solution.y.T
            .reshape(n_times, n_segments, dim)
            .transpose(1, 0, 2)
        )

        if not np.isfinite(prediction).all():
            return float("inf"), n_segments

        return relative_l2(prediction, targets), 0

    except (
        RuntimeError,
        ValueError,
        OverflowError,
        FloatingPointError,
    ):
        return float("inf"), n_segments


def neural_trajectory_error(model, targets, t):
    try:
        x0 = torch.tensor(
            targets[:, 0, :], dtype=torch.float32
        )
        time = torch.tensor(t, dtype=torch.float32)

        with torch.no_grad():
            prediction = model.physical_trajectory(
                x0, time
            ).cpu().numpy()

        if not np.isfinite(prediction).all():
            return float("inf"), len(targets)

        return relative_l2(prediction, targets), 0

    except (RuntimeError, ValueError, OverflowError):
        return float("inf"), len(targets)


def symbolic_field_error(
    library, coefficients, states, true_derivatives
):
    prediction = library.transform(states) @ coefficients
    return relative_l2(prediction, true_derivatives)


def neural_field_error(model, states, true_derivatives):
    predictions = []

    with torch.no_grad():
        for chunk in np.array_split(
            states,
            max(1, int(np.ceil(len(states) / 512))),
        ):
            tensor = torch.tensor(
                chunk, dtype=torch.float32
            )
            prediction = model.physical_field(tensor)
            predictions.append(
                prediction.cpu().numpy()
            )

    return relative_l2(
        np.concatenate(predictions, axis=0),
        true_derivatives,
    )


def lorenz_support(names, coefficients):
    """Support recovery relative to the 7 Lorenz terms."""

    expected = {
        (0, "x0"),
        (0, "x1"),
        (1, "x0"),
        (1, "x1"),
        (1, "x0 x2"),
        (2, "x0 x1"),
        (2, "x2"),
    }

    predicted = {
        (output, name)
        for output in range(coefficients.shape[1])
        for name, value in zip(
            names, coefficients[:, output]
        )
        if abs(value) > 1e-10
    }

    tp = len(expected & predicted)
    fp = len(predicted - expected)
    fn = len(expected - predicted)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return tp, fp, fn, f1


def evaluate_seed(system, seed):
    print(f"\n=== TEST EVALUATION: SEED {seed} ===")

    observed, clean, t = load_test(system, seed)

    # Initial conditions come from noisy observations.
    # Targets are clean trajectories for controlled evaluation.
    noisy_segments, relative_t = make_segments(
        observed, t
    )
    clean_segments, _ = make_segments(
        clean, t
    )

    targets = clean_segments.copy()
    targets[:, 0, :] = noisy_segments[:, 0, :]

    # Compare predictions against clean ground truth,
    # but initialize all methods with the same noisy x0.
    reference = clean_segments

    clean_states = clean.reshape(
        -1, clean.shape[-1]
    )
    true_derivatives = field(system, clean_states)

    neural = load_neural(system, seed)

    results = []

    for method in ["sindy", "symbolic"]:
        library, coef, names = load_symbolic(
            system, seed, method
        )

        # Integration starts at observed x0, while the
        # scoring reference remains the clean trajectory.
        # We calculate the error using a shared evaluator
        # with clean targets, replacing initial states below.
        integration_targets = targets.copy()

        # The helper also scores against its input targets.
        # Correct for that by using a separate integration
        # function defined below.
        trajectory_l2, failed = evaluate_polynomial_clean(
            library,
            coef,
            integration_targets[:, 0, :],
            reference,
            relative_t,
        )

        field_l2 = symbolic_field_error(
            library,
            coef,
            clean_states,
            true_derivatives,
        )

        tp = fp = fn = f1 = None

        if system == "lorenz":
            tp, fp, fn, f1 = lorenz_support(
                names, coef
            )

        results.append({
            "seed": seed,
            "method": (
                "direct_sindy"
                if method == "sindy"
                else "neural_distillation"
            ),
            "trajectory_relative_l2": trajectory_l2,
            "field_relative_l2": field_l2,
            "failed_segments": failed,
            "n_terms": int(np.count_nonzero(
                np.abs(coef) > 1e-10
            )),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support_f1": f1,
        })

    neural_trajectory_l2, neural_failed = (
        evaluate_neural_clean(
            neural,
            targets[:, 0, :],
            reference,
            relative_t,
        )
    )

    results.append({
        "seed": seed,
        "method": "neural_ode",
        "trajectory_relative_l2": neural_trajectory_l2,
        "field_relative_l2": neural_field_error(
            neural, clean_states, true_derivatives
        ),
        "failed_segments": neural_failed,
        "n_terms": None,
        "tp": None,
        "fp": None,
        "fn": None,
        "support_f1": None,
    })

    for result in results:
        print(
            f"{result['method']:20s} "
            f"traj_L2={result['trajectory_relative_l2']:.6f} "
            f"field_L2={result['field_relative_l2']:.6f} "
            f"failed={result['failed_segments']}"
        )

    return results


def evaluate_polynomial_clean(
    library, coefficients, x0, reference, t
):
    n_segments, n_times, dim = reference.shape

    def rhs(_time, flat):
        states = flat.reshape(n_segments, dim)
        values = library.transform(states) @ coefficients
        if not np.isfinite(values).all():
            raise FloatingPointError("Non-finite values")
        return values.reshape(-1)

    def divergence_event(_time, flat):
        return STATE_LIMIT - float(np.max(np.abs(flat)))

    divergence_event.terminal = True
    divergence_event.direction = -1

    try:
        sol = solve_ivp(
            rhs,
            (float(t[0]), float(t[-1])),
            x0.reshape(-1),
            t_eval=t,
            rtol=RTOL,
            atol=ATOL,
            max_step=float(t[1] - t[0]),
            events=divergence_event,
        )

        if not sol.success or sol.y.shape[1] != n_times:
            return float("inf"), n_segments

        prediction = (
            sol.y.T.reshape(n_times, n_segments, dim)
            .transpose(1, 0, 2)
        )

        if not np.isfinite(prediction).all():
            return float("inf"), n_segments

        return relative_l2(prediction, reference), 0

    except (
        RuntimeError,
        ValueError,
        OverflowError,
        FloatingPointError,
    ):
        return float("inf"), n_segments


def evaluate_neural_clean(model, x0, reference, t):
    try:
        with torch.no_grad():
            prediction = model.physical_trajectory(
                torch.tensor(x0, dtype=torch.float32),
                torch.tensor(t, dtype=torch.float32),
            ).cpu().numpy()

        if not np.isfinite(prediction).all():
            return float("inf"), len(x0)

        return relative_l2(prediction, reference), 0

    except (RuntimeError, ValueError, OverflowError):
        return float("inf"), len(x0)


def main():
    cfg = config()
    system = cfg["system"]

    output = (
        ROOT / "results" / "benchmark"
        / system / "evaluation"
    )
    output.mkdir(parents=True, exist_ok=True)

    print("\n=== FROZEN MULTISEED TEST BENCHMARK ===")
    print(f"System: {system}")
    print(f"Seeds: {SEEDS}")
    print("Models are not retrained or retuned.")

    all_results = []

    for seed in SEEDS:
        all_results.extend(
            evaluate_seed(system, seed)
        )

    csv_path = output / "test_results.csv"

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(all_results[0])
        )
        writer.writeheader()
        writer.writerows(all_results)

    summary = {}

    print("\n=== FINAL TEST SUMMARY ===")

    for method in [
        "direct_sindy",
        "neural_distillation",
        "neural_ode",
    ]:
        subset = [
            row for row in all_results
            if row["method"] == method
        ]

        traj = np.asarray([
            row["trajectory_relative_l2"]
            for row in subset
        ])
        vector = np.asarray([
            row["field_relative_l2"]
            for row in subset
        ])

        summary[method] = {
            "trajectory_mean": float(np.mean(traj)),
            "trajectory_std": float(
                np.std(traj, ddof=1)
            ),
            "field_mean": float(np.mean(vector)),
            "field_std": float(
                np.std(vector, ddof=1)
            ),
            "total_failed_segments": int(
                sum(row["failed_segments"] for row in subset)
            ),
        }

        print(
            f"{method:20s} "
            f"traj={traj.mean():.6f} "
            f"+/- {traj.std(ddof=1):.6f} | "
            f"field={vector.mean():.6f} "
            f"+/- {vector.std(ddof=1):.6f}"
        )

    summary_path = output / "test_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"\nCSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print("\nTest evaluation completed.")


if __name__ == "__main__":
    main()
