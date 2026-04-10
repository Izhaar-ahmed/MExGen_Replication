"""
Hybrid Attribution Experiment — LOO→L-SHAP vs MExGen baselines.

Runs all attribution methods (C-LIME, L-SHAP, LOO, Hybrid) on the same
XSUM samples with the log_prob scalarizer, then generates comparison plots.

Uses the same data and evaluation setup as the original MExGen replication.
Existing self-explanation metrics are loaded from results/metrics/metrics.json.
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_xsum
from src.model_wrapper import load_distilbart, select_device
from src.segmentation import segment_text
from src.attribution import explain_loo, explain_clime, explain_lshap
from src.hybrid_attribution import (
    explain_hybrid_loo_lshap,
    explain_hybrid_dynamic,
    estimate_calls_loo,
    estimate_calls_clime,
    estimate_calls_lshap,
    estimate_calls_hybrid,
)
from src.perturbation_eval import drop_top_k_and_score, aupc, average_curves

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_SAMPLES = 5  # Same as the original replication
HYBRID_TOP_FRACTION = 0.7

RESULTS_DIR = Path("results/raw")
METRICS_DIR = Path("results/metrics")
FIGURES_DIR = Path("results/figures")

# Plot styling — matches existing plot_results.py
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

NICE_NAMES = {
    "clime": "C-LIME",
    "lshap": "L-SHAP",
    "loo": "LOO",
    "hybrid": "Hybrid\n(Fixed)",
    "hybrid_dyn": "Hybrid\n(Dynamic)",
    "selfexplain": "Self-Expl.",
}

COLORS = {
    "clime": "#1f77b4",
    "lshap": "#ff7f0e",
    "loo": "#2ca02c",
    "hybrid": "#e377c2",
    "hybrid_dyn": "#d62728",
    "selfexplain": "#9467bd",
}


# ---------------------------------------------------------------------------
# Model call counter wrapper
# ---------------------------------------------------------------------------
class CallCounter:
    """Wraps a scalarizer to count calls."""
    def __init__(self, scalarizer_fn):
        self._fn = scalarizer_fn
        self.count = 0

    def __call__(self, perturbed_input, original_output):
        self.count += 1
        return self._fn(perturbed_input, original_output)

    def reset(self):
        self.count = 0


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_experiment():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("HYBRID ATTRIBUTION EXPERIMENT — LOO→L-SHAP vs MExGen Baselines")
    print(f"Samples: {N_SAMPLES} | Hybrid top_fraction: {HYBRID_TOP_FRACTION}")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Load data & model
    # ------------------------------------------------------------------
    print("\n[1/5] Loading data and model...")
    samples = load_xsum(n_samples=N_SAMPLES)
    distilbart = load_distilbart()
    print(f"  Loaded {len(samples)} XSUM samples, model on {distilbart.device}")

    # ------------------------------------------------------------------
    # 2. Generate outputs & segment
    # ------------------------------------------------------------------
    print("\n[2/5] Generating outputs & segmenting...")
    all_inputs, all_outputs, all_units = [], [], []
    for s in tqdm(samples, desc="  Generating"):
        inp = s["input"]
        out = distilbart.generate(inp)
        units = segment_text(inp, mode="sentence")
        all_inputs.append(inp)
        all_outputs.append(out)
        all_units.append(units)
        tqdm.write(f"    → {len(units)} sentences")

    # ------------------------------------------------------------------
    # 3. Run all attribution methods
    # ------------------------------------------------------------------
    print("\n[3/5] Running attribution methods...")

    # We'll use log_prob as scalarizer (consistent with original metrics)
    def make_scalarizer():
        return lambda p_in, o_out: distilbart.log_prob(p_in, o_out)

    # Store per-sample results
    methods = ["loo", "clime", "lshap", "hybrid", "hybrid_dyn"]
    results = {m: {"scores": [], "curves": [], "calls": []} for m in methods}

    for i in range(N_SAMPLES):
        units = all_units[i]
        orig_out = all_outputs[i]
        orig_inp = all_inputs[i]

        print(f"\n  Sample {i+1}/{N_SAMPLES} ({len(units)} sentences)")

        if len(units) < 2:
            for m in methods:
                results[m]["scores"].append(np.zeros(len(units)))
                results[m]["curves"].append([(0.0, 0.0)])
                results[m]["calls"].append(0)
            continue

        # --- LOO ---
        counter = CallCounter(make_scalarizer())
        sc_loo = explain_loo(units, None, counter, orig_out, orig_inp)
        results["loo"]["scores"].append(sc_loo)
        results["loo"]["calls"].append(counter.count)
        # Use a fresh counter for the curve
        counter_curve = CallCounter(make_scalarizer())
        results["loo"]["curves"].append(
            drop_top_k_and_score(units, sc_loo, counter_curve, orig_out, orig_inp)
        )
        print(f"    LOO:    {counter.count} calls")

        # --- C-LIME ---
        counter = CallCounter(make_scalarizer())
        sc_clime = explain_clime(units, None, counter, orig_out, orig_inp, n_samples_ratio=3)
        results["clime"]["scores"].append(sc_clime)
        results["clime"]["calls"].append(counter.count)
        counter_curve = CallCounter(make_scalarizer())
        results["clime"]["curves"].append(
            drop_top_k_and_score(units, sc_clime, counter_curve, orig_out, orig_inp)
        )
        print(f"    C-LIME: {counter.count} calls")

        # --- L-SHAP ---
        counter = CallCounter(make_scalarizer())
        sc_lshap = explain_lshap(units, None, counter, orig_out, orig_inp)
        results["lshap"]["scores"].append(sc_lshap)
        results["lshap"]["calls"].append(counter.count)
        counter_curve = CallCounter(make_scalarizer())
        results["lshap"]["curves"].append(
            drop_top_k_and_score(units, sc_lshap, counter_curve, orig_out, orig_inp)
        )
        print(f"    L-SHAP: {counter.count} calls")

        # --- HYBRID FIXED (LOO → L-SHAP, top_fraction=0.7) ---
        counter = CallCounter(make_scalarizer())
        sc_hybrid, hybrid_info = explain_hybrid_loo_lshap(
            units, None, counter, orig_out, orig_inp,
            top_fraction=HYBRID_TOP_FRACTION,
        )
        results["hybrid"]["scores"].append(sc_hybrid)
        results["hybrid"]["calls"].append(hybrid_info["model_calls"])
        counter_curve = CallCounter(make_scalarizer())
        results["hybrid"]["curves"].append(
            drop_top_k_and_score(units, sc_hybrid, counter_curve, orig_out, orig_inp)
        )
        print(f"    Hybrid(Fixed):   {hybrid_info['model_calls']} calls "
              f"(k={hybrid_info['k']})")

        # --- HYBRID DYNAMIC (LOO → L-SHAP, adaptive threshold) ---
        counter = CallCounter(make_scalarizer())
        sc_hybrid_dyn, dyn_info = explain_hybrid_dynamic(
            units, None, counter, orig_out, orig_inp,
            threshold_alpha=0.5,
        )
        results["hybrid_dyn"]["scores"].append(sc_hybrid_dyn)
        results["hybrid_dyn"]["calls"].append(dyn_info["model_calls"])
        counter_curve = CallCounter(make_scalarizer())
        results["hybrid_dyn"]["curves"].append(
            drop_top_k_and_score(units, sc_hybrid_dyn, counter_curve, orig_out, orig_inp)
        )
        print(f"    Hybrid(Dynamic): {dyn_info['model_calls']} calls "
              f"(k={dyn_info['k']}, threshold={dyn_info['threshold_value']:.3f})")

    # ------------------------------------------------------------------
    # 4. Compute metrics
    # ------------------------------------------------------------------
    print("\n[4/5] Computing metrics...")

    # AUPC
    aupc_results = {}
    for m in methods:
        aupcs = [aupc(c) for c in results[m]["curves"]]
        aupc_results[m] = {
            "mean": float(np.mean(aupcs)),
            "std": float(np.std(aupcs)),
            "stderr": float(np.std(aupcs) / np.sqrt(len(aupcs))),
            "values": [float(a) for a in aupcs],
        }
        print(f"  {NICE_NAMES[m]:>18s}: AUPC = {aupc_results[m]['mean']:.2f} "
              f"± {aupc_results[m]['stderr']:.2f}")

    # Load self-explanation AUPC from existing metrics
    existing_metrics_path = METRICS_DIR / "metrics.json"
    selfexplain_aupc = None
    if existing_metrics_path.exists():
        with open(existing_metrics_path) as f:
            old_metrics = json.load(f)
        se = old_metrics.get("aupc", {}).get("selfexplain", None)
        if se:
            selfexplain_aupc = se
            print(f"  {'Self-Expl.':>18s}: AUPC = {se['mean']:.2f} "
                  f"± {se['stderr']:.2f} (from previous run)")

    # Model calls
    call_results = {}
    for m in methods:
        call_results[m] = {
            "mean": float(np.mean(results[m]["calls"])),
            "total": int(np.sum(results[m]["calls"])),
        }
    print("\n  Model Calls (mean per sample):")
    for m in methods:
        print(f"    {NICE_NAMES[m]:>18s}: {call_results[m]['mean']:.0f}")

    # Spearman correlation
    print("\n  Spearman Rank Correlations:")
    spearman_matrix = np.ones((len(methods), len(methods)))
    for a_idx, a in enumerate(methods):
        for b_idx, b in enumerate(methods):
            if a_idx >= b_idx:
                continue
            corrs = []
            for k_idx in range(N_SAMPLES):
                s1 = results[a]["scores"][k_idx]
                s2 = results[b]["scores"][k_idx]
                if len(s1) >= 2 and len(s2) >= 2 and len(s1) == len(s2):
                    r, _ = stats.spearmanr(s1, s2)
                    if not np.isnan(r):
                        corrs.append(r)
            if corrs:
                avg = np.mean(corrs)
                spearman_matrix[a_idx, b_idx] = avg
                spearman_matrix[b_idx, a_idx] = avg
                print(f"    {NICE_NAMES[a]:>18s} vs {NICE_NAMES[b]:<18s}: {avg:.3f}")

    # ------------------------------------------------------------------
    # 5. Generate comparison plots
    # ------------------------------------------------------------------
    print("\n[5/5] Generating comparison plots...")

    # ---- Plot A: AUPC Comparison Bar Chart ----
    plot_methods = ["clime", "lshap", "loo", "hybrid", "hybrid_dyn"]
    if selfexplain_aupc:
        plot_methods.append("selfexplain")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(plot_methods))
    vals, errs, labels, cols = [], [], [], []
    for m in plot_methods:
        if m == "selfexplain" and selfexplain_aupc:
            vals.append(selfexplain_aupc["mean"])
            errs.append(selfexplain_aupc["stderr"])
        else:
            vals.append(aupc_results[m]["mean"])
            errs.append(aupc_results[m]["stderr"])
        labels.append(NICE_NAMES[m])
        cols.append(COLORS[m])

    bars = ax.bar(x, vals, yerr=errs, capsize=6, color=cols,
                  edgecolor="black", linewidth=0.5, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("AUPC (×100)")
    ax.set_title("AUPC Comparison — Including Hybrid (LOO→L-SHAP)\n"
                 "XSUM / DistilBART / Log Prob Scalarizer")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "hybrid_aupc_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  → hybrid_aupc_comparison.png")

    # ---- Plot B: Perturbation Curves (all methods) ----
    grid = np.linspace(0, 0.20, 21)
    fig, ax = plt.subplots(figsize=(9, 6))

    for m in ["clime", "lshap", "loo", "hybrid", "hybrid_dyn"]:
        curves = results[m]["curves"]
        g, mean, stderr = average_curves(curves, grid)
        label = NICE_NAMES[m].replace("\n", " ")
        ax.plot(g * 100, mean, color=COLORS[m], linewidth=2.5, label=label)
        ax.fill_between(g * 100, mean - stderr, mean + stderr,
                        alpha=0.15, color=COLORS[m])

    ax.set_xlabel("% Tokens Removed")
    ax.set_ylabel("Avg. Log Prob Decrease")
    ax.set_title("Perturbation Curves — Hybrid vs MExGen Methods\n"
                 "XSUM / DistilBART / Log Prob Scalarizer")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "hybrid_perturbation_curves.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  → hybrid_perturbation_curves.png")

    # ---- Plot C: Model Calls Comparison ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: actual calls
    ax = axes[0]
    call_methods = ["loo", "clime", "lshap", "hybrid", "hybrid_dyn"]
    x = np.arange(len(call_methods))
    call_vals = [call_results[m]["mean"] for m in call_methods]
    call_cols = [COLORS[m] for m in call_methods]
    call_labels = [NICE_NAMES[m] for m in call_methods]

    bars = ax.bar(x, call_vals, color=call_cols, edgecolor="black",
                  linewidth=0.5, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(call_labels)
    ax.set_ylabel("Model Calls (avg per sample)")
    ax.set_title("Actual Model Calls")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, call_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Right: efficiency ratio (AUPC per model call)
    ax = axes[1]
    efficiency = []
    for m in call_methods:
        a = aupc_results[m]["mean"]
        c = call_results[m]["mean"]
        efficiency.append(a / c * 100 if c > 0 else 0)

    bars = ax.bar(x, efficiency, color=call_cols, edgecolor="black",
                  linewidth=0.5, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(call_labels)
    ax.set_ylabel("AUPC per 100 Model Calls")
    ax.set_title("Efficiency: Faithfulness per Compute")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, efficiency):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    fig.suptitle("Computational Efficiency — Hybrid vs Baselines", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "hybrid_efficiency.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  → hybrid_efficiency.png")

    # ---- Plot D: Spearman Heatmap (including Hybrid) ----
    fig, ax = plt.subplots(figsize=(7, 6))
    method_labels = [NICE_NAMES[m].replace("\n", " ") for m in methods]
    sns.heatmap(
        spearman_matrix, annot=True, fmt=".2f",
        xticklabels=method_labels, yticklabels=method_labels,
        vmin=0.3, vmax=1.0, cmap="RdYlBu_r", ax=ax, square=True,
    )
    ax.set_title("Spearman Rank Correlation (Including Hybrid)\n"
                 "XSUM / DistilBART / Log Prob")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "hybrid_spearman.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  → hybrid_spearman.png")

    # ---- Plot E: Summary Dashboard ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: AUPC
    ax = axes[0]
    bar_methods = ["clime", "lshap", "loo", "hybrid", "hybrid_dyn"]
    x = np.arange(len(bar_methods))
    v = [aupc_results[m]["mean"] for m in bar_methods]
    e = [aupc_results[m]["stderr"] for m in bar_methods]
    c = [COLORS[m] for m in bar_methods]
    l = [NICE_NAMES[m] for m in bar_methods]
    bars = ax.bar(x, v, yerr=e, capsize=5, color=c, edgecolor="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(l)
    ax.set_ylabel("AUPC (×100)")
    ax.set_title("Faithfulness (AUPC)")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, v):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Panel 2: Model Calls
    ax = axes[1]
    v2 = [call_results[m]["mean"] for m in bar_methods]
    bars = ax.bar(x, v2, color=c, edgecolor="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(l)
    ax.set_ylabel("Avg Model Calls")
    ax.set_title("Computational Cost")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, v2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Panel 3: Efficiency
    ax = axes[2]
    v3 = [aupc_results[m]["mean"] / call_results[m]["mean"] * 100
          if call_results[m]["mean"] > 0 else 0 for m in bar_methods]
    bars = ax.bar(x, v3, color=c, edgecolor="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(l)
    ax.set_ylabel("AUPC / 100 Calls")
    ax.set_title("Efficiency (Faithfulness ÷ Cost)")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, v3):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.suptitle("Hybrid LOO→L-SHAP: Summary Dashboard\nXSUM / DistilBART / Log Prob",
                 fontsize=15, y=1.04)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "hybrid_summary_dashboard.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  → hybrid_summary_dashboard.png")

    # ------------------------------------------------------------------
    # Save updated metrics
    # ------------------------------------------------------------------
    updated_metrics = {
        "aupc": {},
        "spearman": {
            "methods": methods,
            "matrix": spearman_matrix.tolist(),
        },
        "n_samples": N_SAMPLES,
        "model_calls": call_results,
    }

    for m in methods:
        updated_metrics["aupc"][m] = {
            "mean": aupc_results[m]["mean"],
            "stderr": aupc_results[m]["stderr"],
        }
    if selfexplain_aupc:
        updated_metrics["aupc"]["selfexplain"] = selfexplain_aupc

    metrics_path = METRICS_DIR / "metrics_with_hybrid.json"
    with open(metrics_path, "w") as f:
        json.dump(updated_metrics, f, indent=2)
    print(f"\n  Metrics saved to {metrics_path}")

    # ------------------------------------------------------------------
    # Print final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("EXPERIMENT COMPLETE — SUMMARY")
    print("=" * 65)

    print(f"\n{'Method':>20s} {'AUPC':>8s} {'Calls':>8s} {'Efficiency':>12s}")
    print("-" * 52)
    for m in methods:
        a = aupc_results[m]["mean"]
        c = call_results[m]["mean"]
        eff = a / c * 100 if c > 0 else 0
        print(f"{NICE_NAMES[m]:>20s} {a:>8.2f} {c:>8.0f} {eff:>12.2f}")
    print("-" * 52)

    # Highlight savings
    lshap_calls = call_results["lshap"]["mean"]
    hybrid_calls = call_results["hybrid"]["mean"]
    if lshap_calls > 0:
        savings = (1 - hybrid_calls / lshap_calls) * 100
        print(f"\n  Hybrid uses {savings:.0f}% fewer model calls than L-SHAP")

    lshap_aupc = aupc_results["lshap"]["mean"]
    hybrid_aupc = aupc_results["hybrid"]["mean"]
    if lshap_aupc > 0:
        diff = (hybrid_aupc - lshap_aupc) / lshap_aupc * 100
        direction = "higher" if diff > 0 else "lower"
        print(f"  Hybrid AUPC is {abs(diff):.1f}% {direction} than L-SHAP")

    print(f"\n  Plots saved to {FIGURES_DIR}/")
    print("=" * 65)


if __name__ == "__main__":
    run_experiment()
