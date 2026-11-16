
"""Ablacion de estimadores de derivadas para SINDy directo.

Usa entrenamiento y validacion; no carga el conjunto de prueba.
El caso oracle_true_field es solo una referencia diagnostica.
"""

import sys
import csv
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from scipy.signal import savgol_filter

from dynadistill.utils.common import ROOT, config, seed_all
from dynadistill.data.systems import field
from dynadistill.discovery.sparse import stlsq


DEGREES = (2, 3)
THRESHOLDS = (0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50)
PENALTY = 0.005

ESTIMATORS = [("gradient", None)] + [
    ("savgol", w) for w in (5, 7, 9, 11, 15, 21)
]


def load_split(system, split):
    path = ROOT / "data" / "processed" / f"{system}_{split}.npz"

    with np.load(path) as data:
        x = np.asarray(data["x"], dtype=np.float64)
        t = (
            np.asarray(data["t"], dtype=np.float64)
            if "t" in data.files
            else None
        )

    if x.ndim != 3 or not np.isfinite(x).all():
        raise ValueError(f"Invalid trajectory array: {path}")

    return x, t


def estimate(x, dt, method, window):
    if method == "gradient":
        return np.gradient(x, dt, axis=1, edge_order=2)

    if window > x.shape[1]:
        raise ValueError("Savitzky-Golay window exceeds trajectory length")

    return savgol_filter(
        x,
        window_length=window,
        polyorder=3,
        deriv=1,
        delta=dt,
        axis=1,
        mode="interp",
    )


def rel_l2(pred, target):
    return float(
        np.linalg.norm(pred - target)
        / max(np.linalg.norm(target), 1e-12)
    )


def main():
    cfg = config()
    seed_all(cfg["seed"])
    system = cfg["system"]

    train, t_train = load_split(system, "train")
    val, t_val = load_split(system, "val")

    if train.shape[1:] != val.shape[1:]:
        raise ValueError("Train and validation dimensions differ")

    if t_train is not None:
        t_train = t_train.reshape(-1)
        dt = float(np.mean(np.diff(t_train)))

        if not np.allclose(
            np.diff(t_train), dt, rtol=1e-5, atol=1e-9
        ):
            raise ValueError("Nonuniform time grid is not supported")
    else:
        dt = (
            float(cfg["simulation"]["t_end"])
            / (train.shape[1] - 1)
        )

    if t_val is not None and t_train is not None:
        if not np.allclose(t_val.reshape(-1), t_train):
            raise ValueError(
                "Validation time grid differs from training"
            )

    if dt <= 0:
        raise ValueError("Invalid time step")

    x_train_all = train.reshape(-1, train.shape[-1])
    x_val = val.reshape(-1, val.shape[-1])

    rng = np.random.default_rng(cfg["seed"])
    n = min(
        len(x_train_all),
        int(cfg["discovery"]["sample_points"]),
    )
    indices = rng.choice(
        len(x_train_all), size=n, replace=False
    )
    x_train = x_train_all[indices]

    cases = [
        (name, window, False)
        for name, window in ESTIMATORS
    ]
    cases.append(("oracle_true_field", None, True))

    rows = []
    winners = []

    output = ROOT / "results" / "baselines"
    output.mkdir(parents=True, exist_ok=True)

    print("\n=== DERIVATIVE ESTIMATOR ABLATION ===")

    for method, window, oracle in cases:
        if oracle:
            y_train_all = field(system, x_train_all)
            y_val = field(system, x_val)
        else:
            y_train_all = estimate(
                train, dt, method, window
            ).reshape(-1, train.shape[-1])

            y_val = estimate(
                val, dt, method, window
            ).reshape(-1, val.shape[-1])

        y_train = y_train_all[indices]

        case_best = None
        label = (
            method if window is None
            else f"{method}_w{window}"
        )

        for degree in DEGREES:
            for threshold in THRESHOLDS:
                library, coef = stlsq(
                    x_train,
                    y_train,
                    degree=degree,
                    threshold=threshold,
                    ridge=float(cfg["discovery"]["ridge"]),
                )

                pred_val = library.transform(x_val) @ coef
                val_err = rel_l2(pred_val, y_val)

                terms = int(
                    np.count_nonzero(np.abs(coef) > 1e-10)
                )
                score = val_err + PENALTY * terms

                row = {
                    "estimator": label,
                    "oracle": oracle,
                    "degree": degree,
                    "threshold": threshold,
                    "n_terms": terms,
                    "val_relative_l2": val_err,
                    "score": score,
                }
                rows.append(row)

                if (
                    case_best is None
                    or score < case_best["score"]
                ):
                    case_best = dict(row)
                    case_best["coef"] = coef.copy()
                    case_best["names"] = (
                        library.get_feature_names_out(
                            [
                                f"x{i}"
                                for i in range(x_train.shape[1])
                            ]
                        )
                    )

        print(
            f'{label:20s} '
            f'degree={case_best["degree"]} '
            f'threshold={case_best["threshold"]:.2f} '
            f'terms={case_best["n_terms"]:2d} '
            f'val_L2={case_best["val_relative_l2"]:.6f} '
            f'score={case_best["score"]:.6f}'
            + (" [ORACLE]" if oracle else "")
        )

        np.savez(
            output / f"{system}_sindy_ablation_{label}.npz",
            coef=case_best["coef"],
            names=np.asarray(
                case_best["names"], dtype=str
            ),
            degree=case_best["degree"],
            threshold=case_best["threshold"],
        )

        winners.append({
            k: v
            for k, v in case_best.items()
            if k not in ("coef", "names")
        })

    best_real = min(
        (r for r in winners if not r["oracle"]),
        key=lambda r: r["score"],
    )

    print(
        "\n=== BEST REALISTIC BASELINE "
        "(VALIDATION ONLY) ==="
    )
    print(json.dumps(best_real, indent=2))

    print(
        "\nOracle results are diagnostic, "
        "not eligible for realistic baseline selection."
    )

    csv_path = (
        output / f"{system}_sindy_derivative_ablation.csv"
    )
    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_path = (
        output
        / f"{system}_sindy_derivative_ablation_summary.json"
    )
    summary = {
        "system": system,
        "seed": int(cfg["seed"]),
        "dt": dt,
        "complexity_penalty": PENALTY,
        "best_realistic": best_real,
        "winners": winners,
        "note": (
            "Validation only. Oracle uses "
            "ground-truth derivatives."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"\nCSV: {csv_path}")
    print(f"JSON: {summary_path}")


if __name__ == "__main__":
    main()
