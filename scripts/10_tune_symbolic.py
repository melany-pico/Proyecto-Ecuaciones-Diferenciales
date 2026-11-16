
"""Tune sparse symbolic models using validation data only."""

import sys
import csv
import json
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

import numpy as np
import torch

from dynadistill.utils.common import ROOT, config, seed_all
from dynadistill.models.neuralode import VectorField
from dynadistill.discovery.sparse import stlsq


DEGREES = [2, 3]
THRESHOLDS = [0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50]
COMPLEXITY_PENALTY = 0.005


def load_states(system, split):
    path = (
        ROOT / "data" / "processed"
        / f"{system}_{split}.npz"
    )

    with np.load(path) as data:
        x = data["x"].astype(np.float64)

    return x.reshape(-1, x.shape[-1])


def neural_derivatives(model, states, batch_size=1024):
    outputs = []

    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            batch = torch.as_tensor(
                states[start:start + batch_size],
                dtype=torch.float32,
            )
            result = model.physical_field(batch)
            outputs.append(result.cpu().numpy())

    return np.concatenate(outputs).astype(np.float64)


def relative_error(prediction, target):
    return float(
        np.linalg.norm(prediction - target)
        / max(np.linalg.norm(target), 1e-12)
    )


def format_equations(library, coefficients):
    names = library.get_feature_names_out(
        [f"x{i}" for i in range(coefficients.shape[1])]
    )

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

    return names, equations


def main():
    cfg = config()
    seed_all(cfg["seed"])

    system = cfg["system"]
    dc = cfg["discovery"]

    # No test data are loaded anywhere in this script.
    train_states = load_states(system, "train")
    val_states = load_states(system, "val")

    rng = np.random.default_rng(cfg["seed"])

    n_samples = min(
        len(train_states),
        int(dc["sample_points"]),
    )

    indices = rng.choice(
        len(train_states),
        size=n_samples,
        replace=False,
    )

    train_states = train_states[indices]

    checkpoint_path = (
        ROOT / "results" / "checkpoints"
        / f"{system}_v05.pt"
    )

    ck = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model = VectorField(
        dim=ck["dim"],
        hidden=ck["hidden"],
    )
    model.load_state_dict(ck["state_dict"])
    model.eval()

    if (
        train_states.shape[1] != ck["dim"]
        or val_states.shape[1] != ck["dim"]
    ):
        raise ValueError("Data/checkpoint dimension mismatch")

    train_derivatives = neural_derivatives(
        model, train_states
    )
    val_derivatives = neural_derivatives(
        model, val_states
    )

    results = []
    best = None

    for degree in DEGREES:
        for threshold in THRESHOLDS:
            library, coefficients = stlsq(
                train_states,
                train_derivatives,
                degree=degree,
                threshold=threshold,
                ridge=float(dc["ridge"]),
            )

            train_pred = (
                library.transform(train_states) @ coefficients
            )
            val_pred = (
                library.transform(val_states) @ coefficients
            )

            train_error = relative_error(
                train_pred, train_derivatives
            )
            val_error = relative_error(
                val_pred, val_derivatives
            )

            n_terms = int(
                np.count_nonzero(
                    np.abs(coefficients) > 1e-10
                )
            )

            # Fixed, predeclared selection criterion.
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

            results.append(row)

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
                    "library": library,
                    "coefficients": coefficients.copy(),
                    "train_error": train_error,
                    "val_error": val_error,
                    "n_terms": n_terms,
                }

    output_dir = ROOT / "results" / "equations"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = (
        output_dir / f"{system}_v05_tuning.csv"
    )

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    names, equations = format_equations(
        best["library"],
        best["coefficients"],
    )

    np.savez(
        output_dir / f"{system}_v05_symbolic_tuned.npz",
        coef=best["coefficients"],
        names=names,
        degree=best["degree"],
        threshold=best["threshold"],
    )

    summary = {
        "system": system,
        "checkpoint": str(checkpoint_path),
        "selection": "validation_neural_field",
        "complexity_penalty": COMPLEXITY_PENALTY,
        "degree": best["degree"],
        "threshold": best["threshold"],
        "n_terms": best["n_terms"],
        "train_relative_l2": best["train_error"],
        "val_relative_l2": best["val_error"],
        "selection_score": best["score"],
        "equations": equations,
    }

    json_path = (
        output_dir / f"{system}_v05_symbolic_tuned.json"
    )
    json_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== BEST VALIDATION MODEL ===")
    print(
        f"Degree: {best['degree']} | "
        f"Threshold: {best['threshold']} | "
        f"Terms: {best['n_terms']}"
    )
    print(
        f"Validation relative L2: "
        f"{best['val_error']:.6f}"
    )
    print(
        f"Selection score: {best['score']:.6f}"
    )

    print("\n=== RECOVERED EQUATIONS ===")
    for equation in equations:
        print(equation)

    print(f"\nSaved tuning table: {csv_path}")
    print(f"Saved best model: {json_path}")


if __name__ == "__main__":
    main()
