
#!/usr/bin/env python3
"""
EXPERIMENT 42
Publication-ready figures and tables.

Uses frozen results from experiments 37-41.

No model training.
No hyperparameter tuning.
No new test-set evaluation.
No modification of previous results.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CONF = (
    ROOT
    / "results"
    / "scarcity"
    / "confirmatory_34"
)

OUT = CONF / "manuscript_42"

SYSTEMS = [
    "vanderpol",
    "duffing",
    "lotka_volterra",
]

LEVELS = [
    "obs_100",
    "obs_050",
    "obs_025",
    "obs_010",
]

SEEDS = [
    606, 707, 808, 909, 1010,
    1111, 1212, 1313, 1414, 1515,
    1616, 1717, 1818, 1919, 2020,
]

METHODS = [
    "neural_ode",
    "neural_distillation",
    "direct_sindy",
]

SYMBOLIC_METHODS = [
    "neural_distillation",
    "direct_sindy",
]

SYSTEM_LABELS = {
    "vanderpol": "Van der Pol",
    "duffing": "Duffing",
    "lotka_volterra": "Lotka-Volterra",
}

METHOD_LABELS = {
    "neural_ode": "Neural ODE",
    "neural_distillation": "Neural Distillation",
    "direct_sindy": "Direct SINDy",
}

COLORS = {
    "neural_ode": "#64748b",
    "neural_distillation": "#2563eb",
    "direct_sindy": "#d97706",
}

MARKERS = {
    "neural_ode": "o",
    "neural_distillation": "s",
    "direct_sindy": "^",
}

X_VALUES = np.array([100, 50, 25, 10])


# ============================================================
# UTILITIES
# ============================================================

def save_figure(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)

    png = OUT / f"{name}.png"
    pdf = OUT / f"{name}.pdf"

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"  SAVED {png.name}")
    print(f"  SAVED {pdf.name}")


def configure_plot_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_axis(ax):
    ax.grid(
        axis="y",
        alpha=0.20,
        linewidth=0.7,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xticks(X_VALUES)

    ax.set_xticklabels([
        "100%", "50%", "25%", "10%"
    ])

    ax.invert_xaxis()


def locate_csv(
    required_columns,
    preferred_parts=(),
    exclude_parts=(),
):
    """
    Find a CSV containing the required columns.

    Preferred path fragments help select the intended
    experiment without hardcoding unverified filenames.
    """

    candidates = []

    for path in CONF.rglob("*.csv"):
        if OUT in path.parents:
            continue

        path_text = str(path).lower()

        if any(
            part.lower() in path_text
            for part in exclude_parts
        ):
            continue

        try:
            columns = set(
                pd.read_csv(path, nrows=0).columns
            )
        except Exception:
            continue

        if not set(required_columns).issubset(columns):
            continue

        score = sum(
            part.lower() in path_text
            for part in preferred_parts
        )

        candidates.append((score, path))

    candidates.sort(
        key=lambda item: (
            -item[0],
            str(item[1])
        )
    )

    return candidates


def finite_required(df, columns, name):
    for column in columns:
        if column not in df.columns:
            raise ValueError(
                f"{name}: missing column {column}"
            )

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        invalid = ~np.isfinite(
            values.to_numpy(dtype=float)
        )

        if invalid.any():
            print(
                f"\nINVALID {name}.{column}: "
                f"{invalid.sum()} rows"
            )

            keys = [
                c for c in (
                    "system", "level", "seed",
                    "method", "split"
                )
                if c in df.columns
            ]

            if keys:
                print(
                    df.loc[invalid, keys]
                    .head(15)
                    .to_string(index=False)
                )

            raise ValueError(
                f"Nonfinite required metric: "
                f"{name}.{column}"
            )


def normalize_method(value):
    value = str(value).strip().lower()

    aliases = {
        "neural_ode": "neural_ode",
        "neural ode": "neural_ode",
        "neuralode": "neural_ode",
        "neural_distillation": "neural_distillation",
        "neural distillation": "neural_distillation",
        "nd": "neural_distillation",
        "direct_sindy": "direct_sindy",
        "direct sindy": "direct_sindy",
        "sindy": "direct_sindy",
    }

    return aliases.get(value, value)


def normalize_levels(df):
    df = df.copy()

    df["level"] = (
        df["level"].astype(str).str.strip()
    )

    df["method"] = (
        df["method"].map(normalize_method)
    )

    df["system"] = (
        df["system"].astype(str).str.strip()
    )

    df["seed"] = pd.to_numeric(
        df["seed"], errors="raise"
    ).astype(int)

    return df


def validate_keys(df, methods, label):
    expected = {
        (system, level, seed, method)
        for system in SYSTEMS
        for level in LEVELS
        for seed in SEEDS
        for method in methods
    }

    actual = [
        (
            row.system,
            row.level,
            int(row.seed),
            row.method,
        )
        for row in df.itertuples(index=False)
    ]

    if len(actual) != len(set(actual)):
        raise ValueError(
            f"{label}: duplicate model keys"
        )

    missing = expected - set(actual)
    extra = set(actual) - expected

    if missing or extra:
        print(
            f"{label}: missing={len(missing)}, "
            f"extra={len(extra)}"
        )

        if missing:
            print(
                "Example missing keys:",
                sorted(missing)[:5]
            )

        if extra:
            print(
                "Example extra keys:",
                sorted(extra)[:5]
            )

        raise ValueError(
            f"{label}: frozen design mismatch"
        )


# ============================================================
# LOAD PREDICTIVE RESULTS
# ============================================================


def load_prediction():
    print("\nLoading confirmatory prediction results...")

    frames = []

    for system in SYSTEMS:
        path = (
            CONF
            / "evaluation"
            / system
            / "test_metrics_all.csv"
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing confirmatory CSV: {path}"
            )

        df = pd.read_csv(path)

        required = [
            "system",
            "level",
            "seed",
            "method",
            "traj_rel_l2",
        ]

        missing = set(required) - set(df.columns)

        if missing:
            raise ValueError(
                f"{path}: missing columns "
                f"{sorted(missing)}"
            )

        df = normalize_levels(df)

        if not (df["system"] == system).all():
            raise ValueError(
                f"Unexpected system in {path}"
            )

        if len(df) != 180:
            raise ValueError(
                f"{system}: expected 180 models, "
                f"found {len(df)}"
            )

        print(
            f"  {system}: {len(df)} models"
        )

        frames.append(df)

    prediction = pd.concat(
        frames,
        ignore_index=True,
    )

    validate_keys(
        prediction,
        METHODS,
        "prediction",
    )

    finite_required(
        prediction,
        ["traj_rel_l2"],
        "prediction",
    )

    print(
        "  TOTAL:",
        len(prediction),
        "confirmatory models"
    )

    return prediction


# ============================================================
# LOAD SYMBOLIC AUDIT
# ============================================================

def load_symbolic():
    print("\nLoading symbolic audit...")

    path = (
        CONF
        / "symbolic_audit_40"
        / "per_model.csv"
    )

    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = [
        "system",
        "level",
        "seed",
        "method",
    ]

    for column in required:
        if column not in df.columns:
            raise ValueError(
                f"Symbolic audit missing: {column}"
            )

    df = normalize_levels(df)

    validate_keys(
        df,
        SYMBOLIC_METHODS,
        "symbolic audit",
    )

    # Identify metric columns without silently
    # inventing or replacing scientific results.
    possible_f1 = [
        "structural_f1",
        "support_f1",
        "f1",
    ]

    possible_coef = [
        "coefficient_relative_l2",
        "coef_relative_l2",
        "coefficient_rel_l2",
        "coef_rel_l2",
        "coef_l2",
    ]

    possible_exact = [
        "exact_support",
        "exact_support_recovery",
        "support_exact",
    ]

    def find_column(options, label):
        matches = [
            c for c in options
            if c in df.columns
        ]

        if not matches:
            raise ValueError(
                f"Missing {label} column. "
                f"Available: {list(df.columns)}"
            )

        return matches[0]

    f1_column = find_column(
        possible_f1, "F1"
    )

    coef_column = find_column(
        possible_coef, "coefficient error"
    )

    exact_column = find_column(
        possible_exact, "exact support"
    )

    df = df.rename(columns={
        f1_column: "plot_f1",
        coef_column: "plot_coef_l2",
        exact_column: "plot_exact_support",
    })

    finite_required(
        df,
        [
            "plot_f1",
            "plot_coef_l2",
            "plot_exact_support",
        ],
        "symbolic",
    )

    print(
        "  Source:",
        path.relative_to(ROOT)
    )
    print("  Models:", len(df))

    return df


# ============================================================
# LOAD DUFFING DIAGNOSTIC
# ============================================================

def load_duffing():
    print("\nLoading Duffing field diagnostic...")

    path = (
        CONF
        / "field_diagnostic_41"
        / "per_seed_split.csv"
    )

    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = [
        "level",
        "seed",
        "split",
        "neural_vs_true",
        "nd_vs_true",
        "sindy_vs_true",
        "nd_vs_neural",
    ]

    finite_required(
        df,
        [
            "neural_vs_true",
            "nd_vs_true",
            "sindy_vs_true",
            "nd_vs_neural",
        ],
        "duffing",
    )

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"Duffing missing columns: {missing}"
        )

    expected = {
        (level, seed, split)
        for level in LEVELS
        for seed in SEEDS
        for split in ("train", "val")
    }

    actual = [
        (
            str(row.level),
            int(row.seed),
            str(row.split),
        )
        for row in df.itertuples(index=False)
    ]

    if (
        len(actual) != 120
        or len(set(actual)) != 120
        or set(actual) != expected
    ):
        raise ValueError(
            "Duffing diagnostic design mismatch"
        )

    print(
        "  Source:",
        path.relative_to(ROOT)
    )
    print("  Evaluations:", len(df))

    return df


# ============================================================
# FIGURE 1
# ============================================================

def plot_prediction(df):
    print("\nGenerating Figure 1...")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4.4),
    )

    for ax, system in zip(
        axes, SYSTEMS
    ):
        for method in METHODS:
            means = []
            sems = []

            for level in LEVELS:
                values = df.loc[
                    (df["system"] == system)
                    & (df["level"] == level)
                    & (df["method"] == method),
                    "traj_rel_l2",
                ].to_numpy(dtype=float)

                if len(values) != 15:
                    raise ValueError(
                        "Unexpected sample count "
                        f"for {system}/{level}/{method}"
                    )

                means.append(
                    float(np.mean(values))
                )

                sems.append(
                    float(
                        np.std(values, ddof=1)
                        / np.sqrt(len(values))
                    )
                )

            ax.errorbar(
                X_VALUES,
                means,
                yerr=sems,
                label=METHOD_LABELS[method],
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=1.8,
                markersize=5,
                capsize=3,
            )

        ax.set_title(
            SYSTEM_LABELS[system]
        )

        ax.set_xlabel(
            "Observed time points"
        )

        ax.set_ylabel(
            "Trajectory relative L2"
        )

        style_axis(ax)

    handles, labels = (
        axes[0].get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
        frameon=False,
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.93)
    )

    save_figure(
        fig,
        "figure_1_prediction",
    )


# ============================================================
# FIGURE 2
# ============================================================

def plot_symbolic(df):
    print("\nGenerating Figure 2...")

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14, 7.5),
    )

    metrics = [
        ("plot_f1", "Structural F1"),
        (
            "plot_coef_l2",
            "Coefficient relative L2"
        ),
    ]

    for col, system in enumerate(
        SYSTEMS
    ):
        for row, (
            metric, ylabel
        ) in enumerate(metrics):

            ax = axes[row, col]

            for method in SYMBOLIC_METHODS:
                means = []

                for level in LEVELS:
                    values = df.loc[
                        (df["system"] == system)
                        & (df["level"] == level)
                        & (df["method"] == method),
                        metric,
                    ].to_numpy(dtype=float)

                    if len(values) != 15:
                        raise ValueError(
                            "Unexpected symbolic "
                            "sample count"
                        )

                    means.append(
                        float(np.mean(values))
                    )

                ax.plot(
                    X_VALUES,
                    means,
                    color=COLORS[method],
                    marker=MARKERS[method],
                    linewidth=1.8,
                    label=METHOD_LABELS[method],
                )

            ax.set_ylabel(ylabel)
            ax.set_xlabel(
                "Observed time points"
            )

            if row == 0:
                ax.set_title(
                    SYSTEM_LABELS[system]
                )
                ax.set_ylim(0, 1.05)

            style_axis(ax)

    handles, labels = (
        axes[0, 0].get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.95)
    )

    save_figure(
        fig,
        "figure_2_symbolic_recovery",
    )


# ============================================================
# FIGURE 3
# ============================================================

def plot_duffing(df):
    print("\nGenerating Figure 3...")

    validation = df[
        df["split"] == "val"
    ].copy()

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    series = [
        (
            "neural_vs_true",
            "neural_ode",
        ),
        (
            "nd_vs_true",
            "neural_distillation",
        ),
        (
            "sindy_vs_true",
            "direct_sindy",
        ),
    ]

    for column, method in series:
        means = []
        sems = []

        for level in LEVELS:
            values = validation.loc[
                validation["level"] == level,
                column,
            ].to_numpy(dtype=float)

            if len(values) != 15:
                raise ValueError(
                    f"Invalid Duffing sample: {level}"
                )

            means.append(
                float(np.mean(values))
            )

            sems.append(
                float(
                    np.std(values, ddof=1)
                    / np.sqrt(len(values))
                )
            )

        ax.errorbar(
            X_VALUES,
            means,
            yerr=sems,
            label=METHOD_LABELS[method],
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=2,
            markersize=6,
            capsize=3,
        )

    ax.set_title(
        "Duffing: vector-field fidelity"
    )

    ax.set_xlabel(
        "Observed time points"
    )

    ax.set_ylabel(
        "Field relative L2 vs. true dynamics"
    )

    style_axis(ax)
    ax.legend(frameon=False)

    fig.tight_layout()

    save_figure(
        fig,
        "figure_3_duffing_diagnostic",
    )


# ============================================================
# TABLES
# ============================================================

def generate_tables(
    prediction,
    symbolic,
    duffing,
):
    print("\nGenerating manuscript tables...")

    prediction_summary = (
        prediction.groupby(
            ["system", "level", "method"],
            as_index=False,
        )
        .agg(
            n=("traj_rel_l2", "size"),
            traj_mean=(
                "traj_rel_l2", "mean"
            ),
            traj_sd=(
                "traj_rel_l2", "std"
            ),
        )
    )

    if "field_rel_l2" in prediction.columns:
        field_summary = (
            prediction.groupby(
                ["system", "level", "method"],
                as_index=False,
            )
            .agg(
                field_mean=(
                    "field_rel_l2", "mean"
                ),
                field_sd=(
                    "field_rel_l2", "std"
                ),
            )
        )

        prediction_summary = (
            prediction_summary.merge(
                field_summary,
                on=[
                    "system",
                    "level",
                    "method",
                ],
                validate="one_to_one",
            )
        )

    prediction_summary.to_csv(
        OUT / "table_1_prediction.csv",
        index=False,
    )

    symbolic_summary = (
        symbolic.groupby(
            ["system", "level", "method"],
            as_index=False,
        )
        .agg(
            n=("plot_f1", "size"),
            f1_mean=(
                "plot_f1", "mean"
            ),
            f1_sd=(
                "plot_f1", "std"
            ),
            coef_l2_mean=(
                "plot_coef_l2", "mean"
            ),
            coef_l2_sd=(
                "plot_coef_l2", "std"
            ),
            exact_support_rate=(
                "plot_exact_support", "mean"
            ),
        )
    )

    symbolic_summary.to_csv(
        OUT / "table_2_symbolic_recovery.csv",
        index=False,
    )

    duffing_summary = (
        duffing.groupby(
            ["level", "split"],
            as_index=False,
        )
        .agg(
            n=("seed", "size"),
            neural_mean=(
                "neural_vs_true", "mean"
            ),
            nd_mean=(
                "nd_vs_true", "mean"
            ),
            sindy_mean=(
                "sindy_vs_true", "mean"
            ),
            nd_neural_mean=(
                "nd_vs_neural", "mean"
            ),
        )
    )

    duffing_summary.to_csv(
        OUT / "table_3_duffing_diagnostic.csv",
        index=False,
    )

    print("  SAVED table_1_prediction.csv")
    print("  SAVED table_2_symbolic_recovery.csv")
    print("  SAVED table_3_duffing_diagnostic.csv")


# ============================================================
# MAIN
# ============================================================

def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    configure_plot_style()

    print(
        "=== EXPERIMENT 42: "
        "MANUSCRIPT FIGURES ==="
    )

    prediction = load_prediction()
    symbolic = load_symbolic()
    duffing = load_duffing()

    plot_prediction(prediction)
    plot_symbolic(symbolic)
    plot_duffing(duffing)

    generate_tables(
        prediction,
        symbolic,
        duffing,
    )

    metadata = {
        "experiment": 42,
        "systems": SYSTEMS,
        "levels": LEVELS,
        "seeds": SEEDS,
        "prediction_models": len(prediction),
        "symbolic_models": len(symbolic),
        "duffing_diagnostic_rows": len(duffing),
        "figures": 3,
        "tables": 3,
        "error_bars": (
            "Standard error across 15 seeds; "
            "descriptive only."
        ),
        "analysis_type": (
            "Post-hoc manuscript visualization"
        ),
        "no_retraining": True,
        "no_new_test_evaluation": True,
        "notes": (
            "Confirmatory inference from experiment "
            "38 remains separate from the "
            "descriptive figures generated here."
        ),
    }

    (
        OUT / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== EXPERIMENT 42 COMPLETED ===")
    print(
        "Prediction models:",
        len(prediction)
    )
    print(
        "Symbolic models:",
        len(symbolic)
    )
    print(
        "Duffing diagnostic rows:",
        len(duffing)
    )
    print("Figures: 3 PNG + 3 PDF")
    print("Tables: 3 CSV")
    print("Saved to:", OUT)


if __name__ == "__main__":
    main()
