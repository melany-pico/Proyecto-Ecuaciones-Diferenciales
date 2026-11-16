
"""Distill a trained Neural ODE v0.5 into sparse polynomial equations."""

import sys
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


def main():
    cfg = config()
    seed_all(cfg["seed"])

    system = cfg["system"]
    dc = cfg["discovery"]

    # Load training data only.
    data_path = ROOT / "data" / "processed" / f"{system}_train.npz"
    data = np.load(data_path)

    trajectories = data["x"]
    dim = trajectories.shape[-1]
    states = trajectories.reshape(-1, dim)

    rng = np.random.default_rng(cfg["seed"])
    n_samples = min(len(states), int(dc["sample_points"]))

    indices = rng.choice(
        len(states),
        size=n_samples,
        replace=False,
    )
    states = states[indices].astype(np.float64)

    # Load the best Neural ODE v0.5 checkpoint.
    checkpoint_path = (
        ROOT / "results" / "checkpoints" / f"{system}_v05.pt"
    )

    ck = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if ck["dim"] != dim:
        raise ValueError(
            f"Checkpoint dimension {ck['dim']} != data dimension {dim}"
        )

    model = VectorField(
        dim=ck["dim"],
        hidden=ck["hidden"],
    )

    model.load_state_dict(ck["state_dict"])
    model.eval()

    # Convert normalized dynamics back to physical derivatives.
    with torch.no_grad():
        x_tensor = torch.tensor(states, dtype=torch.float32)
        derivatives = model.physical_field(x_tensor).cpu().numpy()

    derivatives = derivatives.astype(np.float64)

    # Sparse polynomial regression.
    library, coefficients = stlsq(
        states,
        derivatives,
        degree=dc["polynomial_degree"],
        threshold=dc["threshold"],
        ridge=dc["ridge"],
    )

    names = library.get_feature_names_out(
        [f"x{i}" for i in range(dim)]
    )

    # Predictions of the distilled symbolic field.
    theta = library.transform(states)
    symbolic_derivatives = theta @ coefficients

    mse = float(
        np.mean((symbolic_derivatives - derivatives) ** 2)
    )

    relative_l2 = float(
        np.linalg.norm(symbolic_derivatives - derivatives)
        / max(np.linalg.norm(derivatives), 1e-12)
    )

    # Save results without overwriting previous versions.
    output_dir = ROOT / "results" / "equations"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_dir / f"{system}_v05_symbolic.npz",
        coef=coefficients,
        names=names,
        mse_vs_neural=mse,
        relative_l2_vs_neural=relative_l2,
    )

    equations = []

    print("\n=== NEURAL ODE v0.5 -> SYMBOLIC EQUATIONS ===")

    for j in range(dim):
        terms = [
            f"({value:+.6f})*{name}"
            for name, value in zip(names, coefficients[:, j])
            if abs(value) > 1e-10
        ]

        equation = (
            f"dx{j}/dt = " + " ".join(terms)
            if terms
            else f"dx{j}/dt = 0"
        )

        equations.append(equation)
        print(equation)

    metadata = {
        "system": system,
        "checkpoint": str(checkpoint_path),
        "n_samples": int(n_samples),
        "polynomial_degree": int(dc["polynomial_degree"]),
        "threshold": float(dc["threshold"]),
        "ridge": float(dc["ridge"]),
        "mse_vs_neural": mse,
        "relative_l2_vs_neural": relative_l2,
        "equations": equations,
    }

    json_path = output_dir / f"{system}_v05_symbolic.json"
    json_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"\nMSE vs Neural ODE: {mse:.8f}")
    print(f"Relative L2 vs Neural ODE: {relative_l2:.8f}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
