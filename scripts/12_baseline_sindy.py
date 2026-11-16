
"""Direct SINDy baseline with noisy-trajectory derivative estimation.

Train/validation:
    Estimate derivatives from observed trajectories using Savitzky-Golay.
    Fit sparse polynomial models on train.
    Select degree/threshold using validation only.

Test:
    Evaluate the selected model against the true physical vector field.
    Compare with the previously frozen neural-distillation candidate.
"""

import sys
import csv
import json
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import PolynomialFeatures

from dynadistill.utils.common import ROOT, config, seed_all
from dynadistill.data.systems import field
from dynadistill.discovery.sparse import stlsq


DEGREES = [2, 3]
THRESHOLDS = [0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50]
COMPLEXITY_PENALTY = 0.005

# Fixed in advance: do not tune these values on the test set.
SAVGOL_WINDOW = 11
SAVGOL_ORDER = 3


def load_trajectories(system, split):
    path = ROOT / "data" / "processed" / f"{system}_{split}.npz"

    with np.load(path) as data:
        if "x" not in data:
            raise KeyError(f"Missing 'x' array in {path}")
        x = np.asarray(data["x"], dtype=np.float64)

    if x.ndim != 3:
        raise ValueError(
            f"Expected (trajectories, time, dimensions), got {x.shape}"
        )

    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite observations in {path}")

    return x


def estimate_derivatives(x, dt):
    n_time = x.shape[1]

    window = min(SAVGOL_WINDOW, n_time)
    if window % 2 == 0:
        window -= 1

    if window <= SAVGOL_ORDER:
        raise ValueError(
            "Not enough time points for Savitzky-Golay differentiation"
        )

    return savgol_filter(
        x,
        window_length=window,
        polyorder=SAVGOL_ORDER,
        deriv=1,
        delta=dt,
        axis=1,
        mode="interp",
    )


def flatten(x):
    return x.reshape(-1, x.shape[-1])


def metrics(predicted, reference):
    residual = predicted - reference

    return {
        "mse": float(np.mean(residual ** 2)),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "relative_l2": float(
            np.linalg.norm(residual)
            / max(np.linalg.norm(reference), 1e-12)
        ),
    }


def equations_from_coefficients(names, coefficients):
    equations = []

    for j in range(coefficients.shape[1]):
        terms = [
            f"({value:+.6f})*{name}"
            for name, value in zip(names, coefficients[:, j])
            if abs(value) > 1e-10
        ]

        equations.append(
            f"dx{j}/dt = " + " ".join(terms)
            if terms
            else f"dx{j}/dt = 0"
        )

    return equations


def evaluate_symbolic_npz(path, x_test, y_true):
    """Evaluate an already-selected symbolic model without refitting."""
    with np.load(path, allow_pickle=True) as data:
        coefficients = np.asarray(data["coef"], dtype=np.float64)
        names = np.asarray(data["names"]).astype(str)

        if "degree" in data:
            degree = int(data["degree"])
        else:
            raise ValueError(
                "Saved symbolic model must contain its polynomial degree"
            )

    library = PolynomialFeatures(
        degree=degree,
        include_bias=True,
    )
    library.fit(x_test[:1])

    expected_names = library.get_feature_names_out(
        [f"x{i}" for i in range(x_test.shape[1])]
    )

    if not np.array_equal(names, expected_names):
        raise ValueError(
            "Saved symbolic feature names do not match the library"
        )

    if coefficients.shape != (
        len(expected_names), x_test.shape[1]
    ):
        raise ValueError("Invalid saved symbolic coefficient shape")

    predictions = library.transform(x_test) @ coefficients
    return metrics(predictions, y_true)


def main():
    cfg = config()
    seed_all(cfg["seed"])

    system = cfg["system"]

    if system != "lorenz":
        print(
            f"Note: evaluating the configured system '{system}'."
        )

    simulation = cfg["simulation"]
    dt = float(simulation["t_end"]) / (
        int(simulation["points"]) - 1
    )

    if dt <= 0:
        raise ValueError("Invalid simulation time step")

    print("\n=== DIRECT SINDy BASELINE ===")
    print(f"System: {system}")
    print(f"Time step: {dt:.8f}")
    print(
        f"Derivative estimator: Savitzky-Golay "
        f"(window={SAVGOL_WINDOW}, order={SAVGOL_ORDER})"
    )

    # Only training and validation data are used for model selection.
    train_trajectories = load_trajectories(system, "train")
    val_trajectories = load_trajectories(system, "val")

    train_derivatives = estimate_derivatives(
        train_trajectories, dt
    )
    val_derivatives = estimate_derivatives(
        val_trajectories, dt
    )

    x_train_all = flatten(train_trajectories)
    y_train_all = flatten(train_derivatives)

    x_val = flatten(val_trajectories)
    y_val = flatten(val_derivatives)

    rng = np.random.default_rng(cfg["seed"])
    sample_count = min(
        len(x_train_all),
        int(cfg["discovery"]["sample_points"]),
    )

    indices = rng.choice(
        len(x_train_all),
        size=sample_count,
        replace=False,
    )

    x_train = x_train_all[indices]
    y_train = y_train_all[indices]

    print(
        f"Train samples: {len(x_train)} | "
        f"Validation samples: {len(x_val)}"
    )

    rows = []
    best = None

    print("\n=== VALIDATION SEARCH ===")

    for degree in DEGREES:
        for threshold in THRESHOLDS:
            library, coefficients = stlsq(
                x_train,
                y_train,
                degree=degree,
                threshold=threshold,
                ridge=float(cfg["discovery"]["ridge"]),
            )

            train_prediction = (
                library.transform(x_train) @ coefficients
            )
            val_prediction = (
                library.transform(x_val) @ coefficients
            )

            train_error = metrics(
                train_prediction, y_train
            )["relative_l2"]

            val_error = metrics(
                val_prediction, y_val
            )["relative_l2"]

            n_terms = int(
                np.count_nonzero(
                    np.abs(coefficients) > 1e-10
                )
            )

            score = (
                val_error
                + COMPLEXITY_PENALTY * n_terms
            )

            row = {
                "degree": degree,
                "threshold": threshold,
                "n_terms": n_terms,
                "train_relative_l2": train_error,
                "val_relative_l2": val_error,
                "selection_score": score,
            }
            rows.append(row)

            print(
                f"degree={degree} "
                f"threshold={threshold:.2f} "
                f"terms={n_terms:2d} "
                f"train_L2={train_error:.6f} "
                f"val_L2={val_error:.6f} "
                f"score={score:.6f}"
            )

            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "degree": degree,
                    "threshold": threshold,
                    "n_terms": n_terms,
                    "train_error": train_error,
                    "val_error": val_error,
                    "coefficients": coefficients.copy(),
                    "library": library,
                }

    output_dir = ROOT / "results" / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)

    with (
        output_dir / f"{system}_sindy_tuning.csv"
    ).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    names = best["library"].get_feature_names_out(
        [f"x{i}" for i in range(x_train.shape[1])]
    )

    equations = equations_from_coefficients(
        names, best["coefficients"]
    )

    print("\n=== BEST DIRECT SINDy MODEL ===")
    print(
        f"Degree: {best['degree']} | "
        f"Threshold: {best['threshold']} | "
        f"Terms: {best['n_terms']}"
    )
    print(
        f"Validation relative L2: "
        f"{best['val_error']:.6f}"
    )

    for equation in equations:
        print(equation)

    # The test set is loaded only after the best model is frozen.
    test_trajectories = load_trajectories(system, "test")
    x_test = flatten(test_trajectories)

    # Ground-truth derivatives are for final evaluation only.
    y_true = np.asarray(
        field(system, x_test),
        dtype=np.float64,
    )

    direct_prediction = (
        best["library"].transform(x_test)
        @ best["coefficients"]
    )

    direct_metrics = metrics(
        direct_prediction, y_true
    )

    distilled_path = (
        ROOT / "results" / "equations"
        / f"{system}_v05_symbolic_tuned.npz"
    )

    distilled_metrics = evaluate_symbolic_npz(
        distilled_path, x_test, y_true
    )

    print("\n=== INDEPENDENT TEST: VECTOR FIELD ===")
    print(
        f"Direct SINDy: RMSE={direct_metrics['rmse']:.6f}, "
        f"relative_L2={direct_metrics['relative_l2']:.6f}"
    )
    print(
        f"Neural distillation: "
        f"RMSE={distilled_metrics['rmse']:.6f}, "
        f"relative_L2={distilled_metrics['relative_l2']:.6f}"
    )

    np.savez(
        output_dir / f"{system}_sindy_direct.npz",
        coef=best["coefficients"],
        names=np.asarray(names, dtype=str),
        degree=best["degree"],
        threshold=best["threshold"],
    )

    summary = {
        "system": system,
        "seed": int(cfg["seed"]),
        "derivative_estimator": {
            "method": "savgol",
            "window": SAVGOL_WINDOW,
            "polynomial_order": SAVGOL_ORDER,
            "dt": dt,
        },
        "selection": {
            "degree": best["degree"],
            "threshold": best["threshold"],
            "n_terms": best["n_terms"],
            "validation_relative_l2": best["val_error"],
            "complexity_penalty": COMPLEXITY_PENALTY,
            "score": best["score"],
        },
        "test": {
            "direct_sindy": direct_metrics,
            "neural_distillation": distilled_metrics,
        },
        "equations": equations,
        "note": (
            "Hyperparameters selected against estimated derivatives "
            "on validation; physical ground truth used only for test."
        ),
    }

    json_path = (
        output_dir / f"{system}_sindy_baseline_summary.json"
    )
    json_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved: {output_dir}")
    print(f"Summary: {json_path}")


if __name__ == "__main__":
    main()
