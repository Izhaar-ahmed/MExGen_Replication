"""
Generate all result figures from metrics.json.
Produces 5 figures in results/figures/.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

METRICS_DIR = Path(__file__).parent.parent / "results" / "metrics"
FIGURES_DIR = Path(__file__).parent.parent / "results" / "figures"

# Style
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

NICE_NAMES = {
    "log_prob": "Log Prob",
    "bert": "BERT",
    "bart": "BART",
    "summ": "SUMM",
    "clime": "C-LIME",
    "lshap": "L-SHAP",
    "loo": "LOO",
    "pshap": "P-SHAP",
    "selfexplain": "Self-Expl.",
}

COLORS = {
    "clime": "#1f77b4",
    "lshap": "#ff7f0e",
    "loo": "#2ca02c",
    "pshap": "#d62728",
    "selfexplain": "#9467bd",
}


def load_metrics():
    path = METRICS_DIR / "metrics.json"
    with open(path) as f:
        return json.load(f)


def plot_spearman_heatmap(metrics):
    """Figure 3: Spearman correlation heatmap (XSUM C-LIME)."""
    data = metrics["spearman"]
    names = [NICE_NAMES.get(n, n) for n in data["scalarizer_names"]]
    matrix = np.array(data["matrix"])

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    sns.heatmap(
        matrix, annot=True, fmt=".2f",
        xticklabels=names, yticklabels=names,
        vmin=0.3, vmax=1.0,
        cmap="RdYlBu_r", ax=ax,
        square=True,
    )
    ax.set_title("Spearman Rank Correlation\n(C-LIME, DistilBART/XSUM)")
    plt.tight_layout()

    out = FIGURES_DIR / "figure3_spearman.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_perturbation_curves_by_scalarizer(metrics):
    """Figure 4: Perturbation curves for C-LIME on XSUM, one panel per eval scalarizer."""
    curves = metrics.get("curves", {})
    scalarizers = ["log_prob", "bert", "summ"]

    fig, axes = plt.subplots(1, len(scalarizers), figsize=(5 * len(scalarizers), 4), sharey=True)
    if len(scalarizers) == 1:
        axes = [axes]

    for idx, scal in enumerate(scalarizers):
        ax = axes[idx]
        key = f"clime_{scal}"
        if key in curves:
            cd = curves[key]
            grid = np.array(cd["grid"])
            mean = np.array(cd["mean"])
            stderr = np.array(cd["stderr"])

            ax.plot(grid * 100, mean, color=COLORS["clime"], linewidth=2, label="C-LIME")
            ax.fill_between(grid * 100, mean - stderr, mean + stderr, alpha=0.2, color=COLORS["clime"])

        ax.set_xlabel("% Tokens Removed")
        if idx == 0:
            ax.set_ylabel("Avg. Scalarizer Decrease")
        ax.set_title(f"Eval: {NICE_NAMES.get(scal, scal)}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Perturbation Curves — C-LIME (DistilBART/XSUM)", y=1.02, fontsize=14)
    plt.tight_layout()

    out = FIGURES_DIR / "figure4_pertcurves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_explainer_comparison(metrics):
    """Figure 5: Perturbation curves comparing all explainers (eval: Log Prob)."""
    curves = metrics.get("curves", {})

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    for exp in ["clime", "lshap", "loo", "pshap", "selfexplain"]:
        key = f"{exp}_log_prob"
        if key in curves:
            cd = curves[key]
            grid = np.array(cd["grid"])
            mean = np.array(cd["mean"])
            stderr = np.array(cd["stderr"])

            ax.plot(grid * 100, mean, color=COLORS.get(exp, "gray"), linewidth=2,
                    label=NICE_NAMES.get(exp, exp))
            ax.fill_between(grid * 100, mean - stderr, mean + stderr, alpha=0.15,
                            color=COLORS.get(exp, "gray"))

    ax.set_xlabel("% Tokens Removed")
    ax.set_ylabel("Avg. Log Prob Decrease")
    ax.set_title("Explainer Comparison (DistilBART/XSUM, Eval: Log Prob)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = FIGURES_DIR / "figure5_explainer_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_aupc_comparison(metrics):
    """Table 1: AUPC bar chart for all methods (eval: Log Prob)."""
    aupc_data = metrics.get("aupc", {})

    methods = ["clime", "lshap", "loo", "pshap"]
    values = []
    errors = []
    labels = []

    for m in methods:
        key = f"{m}_log_prob"
        if key in aupc_data:
            values.append(aupc_data[key]["mean"])
            errors.append(aupc_data[key]["stderr"])
            labels.append(NICE_NAMES.get(m, m))

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, yerr=errors, capsize=5,
                  color=[COLORS.get(m, "gray") for m in methods[:len(labels)]],
                  edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("AUPC (×100)")
    ax.set_title("AUPC Comparison — XSUM/DistilBART (Eval: Log Prob)")
    ax.grid(True, axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    out = FIGURES_DIR / "table1_aupc_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_selfexplain_comparison(metrics):
    """Table 2: AUPC with self-explanation included."""
    aupc_data = metrics.get("aupc", {})

    methods = ["clime", "lshap", "loo", "pshap", "selfexplain"]
    values = []
    errors = []
    labels = []

    for m in methods:
        key = f"{m}_log_prob"
        if key in aupc_data:
            values.append(aupc_data[key]["mean"])
            errors.append(aupc_data[key]["stderr"])
            labels.append(NICE_NAMES.get(m, m))

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, yerr=errors, capsize=5,
                  color=[COLORS.get(m, "gray") for m in methods[:len(labels)]],
                  edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("AUPC (×100)")
    ax.set_title("MExGen vs Self-Explanation — XSUM/DistilBART (Eval: Log Prob)")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    out = FIGURES_DIR / "table2_selfexplain.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics()

    print("Generating figures...")
    plot_spearman_heatmap(metrics)
    plot_perturbation_curves_by_scalarizer(metrics)
    plot_explainer_comparison(metrics)
    plot_aupc_comparison(metrics)
    plot_selfexplain_comparison(metrics)
    print("\nAll figures generated!")


if __name__ == "__main__":
    main()
