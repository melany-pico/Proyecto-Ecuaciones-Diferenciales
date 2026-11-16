
#!/usr/bin/env python3
"""Experiment 41: Duffing neural-field diagnostic.

Post-hoc analysis of frozen confirmatory models.
Uses only train/validation states, never test states.
No retraining, hyperparameter tuning or model selection.
"""

from pathlib import Path
import csv
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dynadistill.models.neuralode import VectorField

DATA = ROOT / "data" / "scarcity" / "duffing"
MODELS = ROOT / "results" / "scarcity" / "duffing"
OUT = (
    ROOT / "results" / "scarcity"
    / "confirmatory_34" / "field_diagnostic_41"
)

LEVELS = ("obs_100", "obs_050", "obs_025", "obs_010")
SEEDS = (
    606, 707, 808, 909, 1010,
    1111, 1212, 1313, 1414, 1515,
    1616, 1717, 1818, 1919, 2020
)
SPLITS = ("train", "val")


def true_field(x):
    x0 = x[:, 0]
    x1 = x[:, 1]
    return np.column_stack((
        x1,
        x0 - 0.2 * x1 - x0**3
    ))


def load_states(level, seed, split):
    path = (
        DATA / level / f"seed_{seed}"
        / f"{split}.npz"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        x = np.asarray(
            data["x_clean"], dtype=float
        )

    if x.ndim != 3 or x.shape[-1] != 2:
        raise ValueError(
            f"Invalid state array: {path}, {x.shape}"
        )

    if not np.all(np.isfinite(x)):
        raise ValueError(f"Nonfinite states: {path}")

    return x.reshape(-1, 2)


def load_neural_model(level, seed):
    path = (
        MODELS / level / f"seed_{seed}"
        / "neural_ode.pt"
    )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    if checkpoint["system"] != "duffing":
        raise ValueError(f"Unexpected system: {path}")

    if checkpoint["level"] != level:
        raise ValueError(f"Unexpected level: {path}")

    if int(checkpoint["seed"]) != seed:
        raise ValueError(f"Unexpected seed: {path}")

    model = VectorField(
        dim=int(checkpoint["dim"]),
        hidden=int(checkpoint["hidden"])
    )

    model.load_state_dict(
        checkpoint["state_dict"],
        strict=True
    )
    model.eval()

    return model


def neural_field(model, x):
    with torch.no_grad():
        tensor = torch.as_tensor(
            x, dtype=torch.float32
        )
        result = model.physical_field(tensor)

    return result.cpu().numpy().astype(float)


def polynomial_features(x, names):
    columns = []

    for name in names:
        if name == "1":
            columns.append(
                np.ones(len(x), dtype=float)
            )
            continue

        result = np.ones(len(x), dtype=float)

        for factor in name.split():
            if "^" in factor:
                variable, power = factor.split("^")
                power = int(power)
            else:
                variable = factor
                power = 1

            if not variable.startswith("x"):
                raise ValueError(
                    f"Unknown polynomial term: {factor}"
                )

            index = int(variable[1:])

            if index < 0 or index >= x.shape[1]:
                raise ValueError(
                    f"Invalid variable: {factor}"
                )

            result *= x[:, index] ** power

        columns.append(result)

    return np.column_stack(columns)


def load_symbolic(level, seed, method):
    path = (
        MODELS / level / f"seed_{seed}"
        / "symbolic" / f"{method}_best.npz"
    )

    with np.load(path, allow_pickle=False) as data:
        coef = np.asarray(
            data["coef"], dtype=float
        )
        names = [str(v) for v in data["names"]]

    if coef.shape != (len(names), 2):
        raise ValueError(
            f"Invalid symbolic model: {path}"
        )

    return names, coef


def symbolic_field(x, names, coef):
    return polynomial_features(x, names) @ coef


def relative_l2(predicted, reference):
    denominator = np.linalg.norm(reference)

    if denominator <= 0:
        raise ValueError("Zero reference norm")

    return float(
        np.linalg.norm(
            predicted - reference
        ) / denominator
    )


def write_csv(path, rows):
    if not rows:
        raise ValueError("No rows to write")

    with path.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []

    for level in LEVELS:
        for seed in SEEDS:
            model = load_neural_model(
                level, seed
            )

            nd_names, nd_coef = load_symbolic(
                level, seed, "neural_distillation"
            )
            sd_names, sd_coef = load_symbolic(
                level, seed, "direct_sindy"
            )

            for split in SPLITS:
                x = load_states(
                    level, seed, split
                )

                f_true = true_field(x)
                f_neural = neural_field(model, x)
                f_nd = symbolic_field(
                    x, nd_names, nd_coef
                )
                f_sindy = symbolic_field(
                    x, sd_names, sd_coef
                )

                rows.append({
                    "system": "duffing",
                    "level": level,
                    "seed": seed,
                    "split": split,
                    "n_states": len(x),
                    "neural_vs_true": relative_l2(
                        f_neural, f_true
                    ),
                    "nd_vs_true": relative_l2(
                        f_nd, f_true
                    ),
                    "sindy_vs_true": relative_l2(
                        f_sindy, f_true
                    ),
                    "nd_vs_neural": relative_l2(
                        f_nd, f_neural
                    ),
                    "nd_minus_neural_error": (
                        relative_l2(f_nd, f_true)
                        - relative_l2(f_neural, f_true)
                    ),
                })

    if len(rows) != 120:
        raise ValueError(
            f"Expected 120 rows, got {len(rows)}"
        )

    write_csv(
        OUT / "per_seed_split.csv",
        rows
    )

    metric_names = [
        "neural_vs_true",
        "nd_vs_true",
        "sindy_vs_true",
        "nd_vs_neural",
        "nd_minus_neural_error",
    ]

    summary = []

    for level in LEVELS:
        for split in SPLITS:
            group = [
                row for row in rows
                if row["level"] == level
                and row["split"] == split
            ]

            result = {
                "system": "duffing",
                "level": level,
                "split": split,
                "n_seeds": len(group),
            }

            for metric in metric_names:
                values = np.array(
                    [row[metric] for row in group],
                    dtype=float
                )

                result[f"{metric}_mean"] = float(
                    np.mean(values)
                )
                result[f"{metric}_median"] = float(
                    np.median(values)
                )
                result[f"{metric}_sd"] = float(
                    np.std(values, ddof=1)
                )

            summary.append(result)

    write_csv(
        OUT / "summary.csv",
        summary
    )

    metadata = {
        "experiment": 41,
        "system": "duffing",
        "levels": list(LEVELS),
        "seeds": list(SEEDS),
        "splits": list(SPLITS),
        "checkpoints": len(LEVELS) * len(SEEDS),
        "rows": len(rows),
        "states": "Original x_clean train/val states",
        "neural_evaluation": "physical_field",
        "note": (
            "Post-hoc diagnostic using frozen models. "
            "No test data, retraining, tuning, or "
            "new model selection."
        ),
    }

    (OUT / "metadata.json").write_text(
        json.dumps(
            metadata,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "=== EXPERIMENT 41: "
        "DUFFING FIELD DIAGNOSTIC ==="
    )
    print(
        f"Validated {len(LEVELS) * len(SEEDS)} "
        f"checkpoints and {len(rows)} "
        "seed/split evaluations."
    )

    for row in summary:
        print(
            f"{row['level']} "
            f"{row['split']:5s} "
            f"Neural/True="
            f"{row['neural_vs_true_mean']:.4f} "
            f"ND/True="
            f"{row['nd_vs_true_mean']:.4f} "
            f"SINDy/True="
            f"{row['sindy_vs_true_mean']:.4f} "
            f"ND/Neural="
            f"{row['nd_vs_neural_mean']:.4f} "
            f"ND-Neural="
            f"{row['nd_minus_neural_error_mean']:+.4f}"
        )

    print("SAVED", OUT)


if __name__ == "__main__":
    main()
