
#!/usr/bin/env python3
"""Experiment 40: symbolic coefficient recovery audit.

Uses only the 15 confirmatory seeds.
No training, model selection, or new test evaluation.
"""

from pathlib import Path
import csv
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "scarcity"
OUT = BASE / "confirmatory_34" / "symbolic_audit_40"

SYSTEMS = ("vanderpol", "duffing", "lotka_volterra")
LEVELS = ("obs_100", "obs_050", "obs_025", "obs_010")
SEEDS = (
    606, 707, 808, 909, 1010,
    1111, 1212, 1313, 1414, 1515,
    1616, 1717, 1818, 1919, 2020
)
METHODS = ("neural_distillation", "direct_sindy")

# True equations from src/dynadistill/data/systems.py
# Keys are (polynomial term, derivative equation index).
TRUTH = {
    "vanderpol": {
        ("x1", 0): 1.0,
        ("x0", 1): -1.0,
        ("x1", 1): 1.0,
        ("x0^2 x1", 1): -1.0,
    },
    "duffing": {
        ("x1", 0): 1.0,
        ("x0", 1): 1.0,
        ("x1", 1): -0.2,
        ("x0^3", 1): -1.0,
    },
    "lotka_volterra": {
        ("x0", 0): 1.5,
        ("x0 x1", 0): -1.0,
        ("x1", 1): -3.0,
        ("x0 x1", 1): 1.0,
    },
}

DIMS = {system: 2 for system in SYSTEMS}


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path}")

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def calculate_metrics(recovered, truth):
    diff = recovered - truth

    true_mask = truth != 0
    pred_mask = recovered != 0

    tp = int(np.sum(true_mask & pred_mask))
    fp = int(np.sum(~true_mask & pred_mask))
    fn = int(np.sum(true_mask & ~pred_mask))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "structural_precision": precision,
        "structural_recall": recall,
        "structural_f1": f1,
        "exact_support": int(
            np.array_equal(true_mask, pred_mask)
        ),
        "true_terms": int(np.sum(true_mask)),
        "recovered_terms": int(np.sum(pred_mask)),
        "false_positive_terms": fp,
        "false_negative_terms": fn,
        "coefficient_relative_l2": float(
            np.linalg.norm(diff) / np.linalg.norm(truth)
        ),
        "coefficient_mae_true_terms": float(
            np.mean(np.abs(diff[true_mask]))
        ),
        "spurious_coefficient_l1": float(
            np.sum(np.abs(recovered[~true_mask]))
        ),
        "coefficient_max_abs_error": float(
            np.max(np.abs(diff))
        ),
    }


def load_model(system, level, seed, method):
    path = (
        BASE / system / level / f"seed_{seed}"
        / "symbolic" / f"{method}_best.npz"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        coef = np.asarray(data["coef"], dtype=float).copy()
        names = [str(x) for x in data["names"]]
        declared_terms = int(data["terms"])

    if coef.shape != (len(names), DIMS[system]):
        raise ValueError(
            f"Invalid coefficient shape: {path}"
        )

    if len(set(names)) != len(names):
        raise ValueError(
            f"Duplicate polynomial terms: {path}"
        )

    if not np.all(np.isfinite(coef)):
        raise ValueError(
            f"Nonfinite coefficients: {path}"
        )

    if int(np.count_nonzero(coef)) != declared_terms:
        raise ValueError(
            f"Term count mismatch: {path}"
        )

    return path, coef, names


def align_with_truth(system, coef, names):
    """Build common polynomial basis.

    True terms absent from the model library are assigned
    a recovered coefficient of zero.
    """

    all_names = list(names)

    for term, equation in TRUTH[system]:
        if term not in all_names:
            all_names.append(term)

    index = {
        name: i for i, name in enumerate(all_names)
    }

    recovered = np.zeros(
        (len(all_names), DIMS[system]),
        dtype=float
    )
    truth = np.zeros_like(recovered)

    for i, name in enumerate(names):
        recovered[index[name], :] = coef[i, :]

    for (term, equation), value in TRUTH[system].items():
        truth[index[term], equation] = value

    return all_names, recovered, truth


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    model_rows = []
    detail_rows = []

    for system in SYSTEMS:
        for level in LEVELS:
            for seed in SEEDS:
                for method in METHODS:
                    path, coef, names = load_model(
                        system, level, seed, method
                    )

                    all_names, recovered, truth = (
                        align_with_truth(
                            system, coef, names
                        )
                    )

                    result = calculate_metrics(
                        recovered, truth
                    )

                    model_rows.append({
                        "system": system,
                        "level": level,
                        "seed": seed,
                        "method": method,
                        **result,
                    })

                    for i, name in enumerate(all_names):
                        for equation in range(DIMS[system]):
                            actual = float(
                                recovered[i, equation]
                            )
                            expected = float(
                                truth[i, equation]
                            )

                            detail_rows.append({
                                "system": system,
                                "level": level,
                                "seed": seed,
                                "method": method,
                                "term": name,
                                "equation": equation,
                                "true_coefficient": expected,
                                "recovered_coefficient": actual,
                                "absolute_error": abs(
                                    actual - expected
                                ),
                                "is_true_term": int(
                                    expected != 0
                                ),
                                "is_recovered_term": int(
                                    actual != 0
                                ),
                            })

    if len(model_rows) != 360:
        raise ValueError(
            f"Expected 360 models, found {len(model_rows)}"
        )

    write_csv(
        OUT / "per_model.csv",
        model_rows
    )
    write_csv(
        OUT / "coefficient_details.csv",
        detail_rows
    )

    lookup = {
        (
            r["system"],
            r["level"],
            r["seed"],
            r["method"]
        ): r
        for r in model_rows
    }

    metric_names = [
        "structural_f1",
        "exact_support",
        "coefficient_relative_l2",
        "coefficient_mae_true_terms",
        "spurious_coefficient_l1",
        "recovered_terms",
    ]

    summary = []
    paired = []

    for system in SYSTEMS:
        for level in LEVELS:
            for method in METHODS:
                group = [
                    lookup[
                        (system, level, seed, method)
                    ]
                    for seed in SEEDS
                ]

                row = {
                    "system": system,
                    "level": level,
                    "method": method,
                    "n_seeds": len(group),
                }

                for metric in metric_names:
                    vals = np.asarray(
                        [r[metric] for r in group],
                        dtype=float
                    )

                    row[f"{metric}_mean"] = float(
                        np.mean(vals)
                    )
                    row[f"{metric}_median"] = float(
                        np.median(vals)
                    )
                    row[f"{metric}_sd"] = float(
                        np.std(vals, ddof=1)
                    )

                summary.append(row)

            for seed in SEEDS:
                nd = lookup[
                    (
                        system, level, seed,
                        "neural_distillation"
                    )
                ]
                sindy = lookup[
                    (
                        system, level, seed,
                        "direct_sindy"
                    )
                ]

                row = {
                    "system": system,
                    "level": level,
                    "seed": seed,
                }

                for metric in metric_names:
                    row[f"nd_{metric}"] = nd[metric]
                    row[f"sindy_{metric}"] = sindy[metric]
                    row[f"delta_{metric}"] = (
                        nd[metric] - sindy[metric]
                    )

                paired.append(row)

    write_csv(
        OUT / "summary.csv",
        summary
    )
    write_csv(
        OUT / "paired.csv",
        paired
    )

    metadata = {
        "experiment": 40,
        "scope": "15 confirmatory seeds",
        "systems": list(SYSTEMS),
        "levels": list(LEVELS),
        "methods": list(METHODS),
        "models": len(model_rows),
        "paired_comparisons": len(paired),
        "coefficient_records": len(detail_rows),
        "note": (
            "Descriptive post-hoc coefficient audit. "
            "Missing true polynomial terms are treated "
            "as recovered coefficients equal to zero. "
            "No retraining or model selection."
        ),
    }

    (OUT / "metadata.json").write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(
        "=== EXPERIMENT 40: "
        "SYMBOLIC RECOVERY AUDIT ==="
    )
    print(
        f"Validated {len(model_rows)} symbolic models; "
        f"{len(paired)} paired comparisons."
    )

    for system in SYSTEMS:
        for level in LEVELS:
            nd = next(
                r for r in summary
                if r["system"] == system
                and r["level"] == level
                and r["method"] == "neural_distillation"
            )
            sd = next(
                r for r in summary
                if r["system"] == system
                and r["level"] == level
                and r["method"] == "direct_sindy"
            )

            print(
                f"{system:14s} {level} "
                f"F1 ND/SINDy="
                f"{nd['structural_f1_mean']:.3f}/"
                f"{sd['structural_f1_mean']:.3f} "
                f"coef_L2 ND/SINDy="
                f"{nd['coefficient_relative_l2_mean']:.3f}/"
                f"{sd['coefficient_relative_l2_mean']:.3f} "
                f"exact_support ND/SINDy="
                f"{nd['exact_support_mean']:.2f}/"
                f"{sd['exact_support_mean']:.2f}"
            )

    print("SAVED", OUT)


if __name__ == "__main__":
    main()
