"""
Compute Spearman correlation matrices and AUPC values from raw experiment results.
Saves metrics to results/metrics/metrics.json.
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats

from src.perturbation_eval import aupc, average_curves


RESULTS_DIR = Path(__file__).parent.parent / "results"
RAW_DIR = RESULTS_DIR / "raw"
METRICS_DIR = RESULTS_DIR / "metrics"

SCALARIZER_NAMES = ["log_prob", "bert", "bart", "summ"]
EXPLAINER_NAMES = ["clime", "lshap", "loo"]


def load_npz(filename: str):
    """Load .npz file from results/raw/."""
    path = RAW_DIR / filename
    if not path.exists():
        print(f"  Warning: {path} not found")
        return None
    return np.load(path, allow_pickle=True)


def compute_spearman_matrix():
    """
    Compute Spearman rank correlation matrix between scalarizers for C-LIME on XSUM.
    """
    print("Computing Spearman correlation matrix...")

    # Load C-LIME scores for all 4 scalarizers
    scores_by_scal = {}
    for scal in SCALARIZER_NAMES:
        data = load_npz(f"xsum_distilbart_clime_{scal}.npz")
        if data is not None:
            scores_by_scal[scal] = data["scores"]

    n_scal = len(scores_by_scal)
    scal_names = list(scores_by_scal.keys())

    # Compute pairwise Spearman correlations
    n_samples = len(list(scores_by_scal.values())[0])
    corr_matrix = np.ones((n_scal, n_scal))

    for i in range(n_scal):
        for j in range(i + 1, n_scal):
            correlations = []
            for k in range(n_samples):
                s1 = scores_by_scal[scal_names[i]][k]
                s2 = scores_by_scal[scal_names[j]][k]
                if len(s1) >= 2 and len(s2) >= 2 and len(s1) == len(s2):
                    rho, _ = stats.spearmanr(s1, s2)
                    if not np.isnan(rho):
                        correlations.append(rho)
            if correlations:
                avg_corr = np.mean(correlations)
                corr_matrix[i, j] = avg_corr
                corr_matrix[j, i] = avg_corr

    return scal_names, corr_matrix.tolist()


def compute_aupc_values():
    """Compute AUPC values for all methods on XSUM."""
    print("Computing AUPC values...")

    aupc_values = {}

    # MExGen methods with all scalarizers
    for exp in EXPLAINER_NAMES:
        for scal in SCALARIZER_NAMES:
            data = load_npz(f"xsum_distilbart_{exp}_{scal}.npz")
            if data is not None:
                curves = data["curves"]
                aupcs = []
                for curve in curves:
                    curve_list = curve.tolist() if hasattr(curve, 'tolist') else list(curve)
                    aupcs.append(aupc(curve_list))
                key = f"{exp}_{scal}"
                aupc_values[key] = {
                    "mean": float(np.mean(aupcs)),
                    "std": float(np.std(aupcs)),
                    "stderr": float(np.std(aupcs) / np.sqrt(len(aupcs))),
                }
                print(f"  {key}: AUPC = {aupc_values[key]['mean']:.2f} ± {aupc_values[key]['stderr']:.2f}")

    # P-SHAP
    data = load_npz("xsum_distilbart_pshap_log_prob.npz")
    if data is not None:
        curves = data["curves"]
        aupcs = [aupc(c.tolist() if hasattr(c, 'tolist') else list(c)) for c in curves]
        aupc_values["pshap_log_prob"] = {
            "mean": float(np.mean(aupcs)),
            "std": float(np.std(aupcs)),
            "stderr": float(np.std(aupcs) / np.sqrt(len(aupcs))),
        }
        print(f"  pshap_log_prob: AUPC = {aupc_values['pshap_log_prob']['mean']:.2f}")

    # Self-Explanation
    data = load_npz("xsum_distilbart_selfexplain_log_prob.npz")
    if data is not None:
        curves = data["curves"]
        aupcs = [aupc(c.tolist() if hasattr(c, 'tolist') else list(c)) for c in curves]
        aupc_values["selfexplain_log_prob"] = {
            "mean": float(np.mean(aupcs)),
            "std": float(np.std(aupcs)),
            "stderr": float(np.std(aupcs) / np.sqrt(len(aupcs))),
        }
        print(f"  selfexplain_log_prob: AUPC = {aupc_values['selfexplain_log_prob']['mean']:.2f}")

    return aupc_values


def compute_perturbation_curves():
    """Compute averaged perturbation curves for plotting."""
    print("Computing averaged perturbation curves...")

    grid = np.linspace(0, 0.20, 21)
    curve_data = {}

    # C-LIME with each scalarizer (for Figure 4)
    for scal in SCALARIZER_NAMES:
        data = load_npz(f"xsum_distilbart_clime_{scal}.npz")
        if data is not None:
            curves = [c.tolist() if hasattr(c, 'tolist') else list(c) for c in data["curves"]]
            g, mean, stderr = average_curves(curves, grid)
            curve_data[f"clime_{scal}"] = {
                "grid": g.tolist(),
                "mean": mean.tolist(),
                "stderr": stderr.tolist(),
            }

    # All explainers with log_prob (for Figure 5)
    for exp in EXPLAINER_NAMES + ["pshap", "selfexplain"]:
        key = f"xsum_distilbart_{exp}_log_prob.npz"
        data = load_npz(key)
        if data is not None:
            curves = [c.tolist() if hasattr(c, 'tolist') else list(c) for c in data["curves"]]
            g, mean, stderr = average_curves(curves, grid)
            curve_data[f"{exp}_log_prob"] = {
                "grid": g.tolist(),
                "mean": mean.tolist(),
                "stderr": stderr.tolist(),
            }

    return curve_data


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {}

    # Spearman matrix
    scal_names, corr_matrix = compute_spearman_matrix()
    metrics["spearman"] = {
        "scalarizer_names": scal_names,
        "matrix": corr_matrix,
    }

    # AUPC values
    metrics["aupc"] = compute_aupc_values()

    # Perturbation curves
    metrics["curves"] = compute_perturbation_curves()

    # Save
    out_path = METRICS_DIR / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {out_path}")


if __name__ == "__main__":
    main()
