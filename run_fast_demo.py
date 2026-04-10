"""
Expanded Fast Demo — runs a multi-scalarizer experiment pipeline.
Includes Log Prob, BERT, and BART scalarizers.
Generates Spearman heatmaps for C-LIME and L-SHAP.
"""

import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_xsum
from src.model_wrapper import load_distilbart, load_flan_t5_large, select_device
from src.segmentation import segment_text
from src.attribution import explain_loo, explain_clime, explain_lshap
from src.self_explanation import self_explain
from src.perturbation_eval import drop_top_k_and_score, aupc, average_curves
from src.scalarizers import SCALARIZERS

RESULTS_DIR = Path("results/raw")
METRICS_DIR = Path("results/metrics")
FIGURES_DIR = Path("results/figures")

N_SAMPLES = 3  # Reduced to 3 because multi-scalarizer is slower
SCAL_NAMES = ["log_prob", "bert", "bart"]


def make_scalarizer(name, model_wrapper):
    base_fn = SCALARIZERS[name]
    if name == "log_prob":
        return lambda p_in, o_out: base_fn(p_in, o_out, model_wrapper=model_wrapper)
    else:
        # For text scalarizers (bert, bart), we need to generate output from perturbed input first
        def scal_fn(p_in, o_out, _mw=model_wrapper, _bf=base_fn):
            p_out = _mw.generate(p_in)
            return _bf(p_out, o_out)
        return scal_fn


def run_expanded_demo():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MExGen EXPANDED DEMO — XSUM / DistilBART")
    print(f"Samples: {N_SAMPLES} | Scalarizers: {SCAL_NAMES}")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading data...")
    samples = load_xsum(n_samples=N_SAMPLES)
    
    # Load models
    print("\n[2/5] Loading models...")
    distilbart = load_distilbart()
    flan_t5 = load_flan_t5_large()

    # Generate outputs & segment
    print("\n[3/5] Generating outputs...")
    all_inputs, all_outputs, all_units = [], [], []
    for s in tqdm(samples, desc="  Generating"):
        inp = s["input"]
        out = distilbart.generate(inp)
        units = segment_text(inp, mode="sentence")
        all_inputs.append(inp)
        all_outputs.append(out)
        all_units.append(units)

    # Initialize results structure
    # results[method][scalarizer] = {scores: [], curves: []}
    results = {m: {s_n: {"scores": [], "curves": []} for s_n in SCAL_NAMES} 
               for m in ["clime", "lshap", "loo"]}
    results["selfexplain"] = {"scores": [], "curves": []}

    # Pre-bound scalarizers
    bound_scals = {s_n: make_scalarizer(s_n, distilbart) for s_n in SCAL_NAMES}

    # -----------------------------------------------------------------------
    # Phase 4: Run attributions
    # -----------------------------------------------------------------------
    print("\n[4/5] Running attributions...")
    
    for i in range(N_SAMPLES):
        print(f"\n  Checking Sample {i+1}/{N_SAMPLES}...")
        units = all_units[i]
        orig_out = all_outputs[i]
        orig_inp = all_inputs[i]
        
        if len(units) < 2:
            for m in ["clime", "lshap", "loo"]:
                for s_n in SCAL_NAMES:
                    results[m][s_n]["scores"].append(np.zeros(len(units)))
                    results[m][s_n]["curves"].append([(0.0, 0.0)])
            results["selfexplain"]["scores"].append(np.zeros(len(units)))
            results["selfexplain"]["curves"].append([(0.0, 0.0)])
            continue

        # MExGen methods × Scalarizers
        for s_n in SCAL_NAMES:
            scal_fn = bound_scals[s_n]
            
            # C-LIME
            sc_clime = explain_clime(units, None, scal_fn, orig_out, orig_inp, n_samples_ratio=3)
            results["clime"][s_n]["scores"].append(sc_clime)
            results["clime"][s_n]["curves"].append(drop_top_k_and_score(units, sc_clime, scal_fn, orig_out, orig_inp))
            
            # L-SHAP
            sc_lshap = explain_lshap(units, None, scal_fn, orig_out, orig_inp)
            results["lshap"][s_n]["scores"].append(sc_lshap)
            results["lshap"][s_n]["curves"].append(drop_top_k_and_score(units, sc_lshap, scal_fn, orig_out, orig_inp))
            
            # LOO
            sc_loo = explain_loo(units, None, scal_fn, orig_out, orig_inp)
            results["loo"][s_n]["scores"].append(sc_loo)
            results["loo"][s_n]["curves"].append(drop_top_k_and_score(units, sc_loo, scal_fn, orig_out, orig_inp))

        # Self-explanation (once per sample, evaluated with log_prob)
        sc_se, _ = self_explain(units, orig_out, flan_t5, task="summarization")
        results["selfexplain"]["scores"].append(sc_se)
        results["selfexplain"]["curves"].append(drop_top_k_and_score(units, sc_se, bound_scals["log_prob"], orig_out, orig_inp))

    # -----------------------------------------------------------------------
    # Phase 5: Metrics & Plotting
    # -----------------------------------------------------------------------
    print("\n[5/5] Generating plots...")
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    
    NICE_M = {"clime": "C-LIME", "lshap": "L-SHAP", "loo": "LOO", "selfexplain": "Self-Expl."}
    NICE_S = {"log_prob": "Log Prob", "bert": "BERT", "bart": "BART"}
    COLORS = {"clime": "#1f77b4", "lshap": "#ff7f0e", "loo": "#2ca02c", "selfexplain": "#9467bd"}

    # --- Figure 3A: Spearman C-LIME ---
    def plot_spearman(method, filename, title):
        corr_mat = np.ones((len(SCAL_NAMES), len(SCAL_NAMES)))
        for a in range(len(SCAL_NAMES)):
            for b in range(a+1, len(SCAL_NAMES)):
                corrs = []
                for k in range(N_SAMPLES):
                    s1 = results[method][SCAL_NAMES[a]]["scores"][k]
                    s2 = results[method][SCAL_NAMES[b]]["scores"][k]
                    if len(s1) >= 2:
                        r, _ = stats.spearmanr(s1, s2)
                        if not np.isnan(r): corrs.append(r)
                if corrs: corr_mat[a, b] = corr_mat[b, a] = np.mean(corrs)
        
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(corr_mat, annot=True, fmt=".2f",
                    xticklabels=[NICE_S[s] for s in SCAL_NAMES],
                    yticklabels=[NICE_S[s] for s in SCAL_NAMES],
                    vmin=0.0, vmax=1.0, cmap="YlGnBu", ax=ax, square=True)
        ax.set_title(title)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / filename)
        plt.close(fig)
        print(f"  → {filename}")

    plot_spearman("clime", "figure3a_spearman_clime.png", "Spearman Correlation (C-LIME)\nAcross Scalarizers")
    plot_spearman("lshap", "figure3b_spearman_lshap.png", "Spearman Correlation (L-SHAP)\nAcross Scalarizers")

    # --- Figure 4: Multipanel Perturbation Curves ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    grid = np.linspace(0, 0.20, 21)
    for idx, s_n in enumerate(SCAL_NAMES):
        ax = axes[idx]
        for m in ["clime", "lshap", "loo"]:
            g, mean, se = average_curves(results[m][s_n]["curves"], grid)
            ax.plot(g * 100, mean, color=COLORS[m], lw=2, label=NICE_M[m])
            ax.fill_between(g * 100, mean - se, mean + se, alpha=0.15, color=COLORS[m])
        ax.set_title(f"Eval Scalarizer: {NICE_S[s_n]}")
        ax.set_xlabel("% Tokens Removed")
        if idx == 0: ax.set_ylabel("Avg. Scalarizer Decrease")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figure4_expanded_curves.png")
    plt.close(fig)
    print("  → figure4_expanded_curves.png")

    # --- Table 2 Update (MExGen vs Self-Explain) ---
    # Using log_prob for comparison
    fig, ax = plt.subplots(figsize=(7, 5))
    all_m = ["clime", "lshap", "loo", "selfexplain"]
    labels = [NICE_M[m] for m in all_m]
    vals = []
    errs = []
    for m in all_m:
        curves = results[m]["log_prob"]["curves"] if m != "selfexplain" else results["selfexplain"]["curves"]
        aupcs = [aupc(c) for c in curves]
        vals.append(np.mean(aupcs))
        errs.append(np.std(aupcs)/np.sqrt(len(aupcs)))
    
    bars = ax.bar(labels, vals, yerr=errs, capsize=5, color=[COLORS[m] for m in all_m], edgecolor="black", lw=0.5)
    ax.set_ylabel("AUPC (×100)")
    ax.set_title("MExGen vs Self-Explanation (Log Prob)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "table2_expanded_selfexplain.png")
    plt.close(fig)
    print("  → table2_expanded_selfexplain.png")

    print("\nDONE! All plots saved to results/figures/")


if __name__ == "__main__":
    run_expanded_demo()
