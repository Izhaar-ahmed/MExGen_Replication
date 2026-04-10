"""
Attribution methods: C-LIME, L-SHAP, and LOO using the ICX360 library.
Also includes a standalone LOO implementation for flexibility.
"""

import numpy as np
from typing import Callable


def explain_loo(
    units: list[str],
    model_fn: Callable,
    scalarizer_fn: Callable,
    original_output: str,
    original_input: str,
) -> np.ndarray:
    """
    Leave-One-Out attribution.
    score_s = S(x_original) - S(x_without_unit_s)
    """
    d = len(units)
    scores = np.zeros(d)

    # Baseline score with full input
    baseline = scalarizer_fn(original_input, original_output)

    for i in range(d):
        # Remove unit i, concatenate remaining
        perturbed_units = units[:i] + units[i + 1:]
        perturbed_input = " ".join(perturbed_units)
        if not perturbed_input.strip():
            perturbed_input = " "  # avoid empty input

        perturbed_score = scalarizer_fn(perturbed_input, original_output)
        scores[i] = baseline - perturbed_score

    return scores


def explain_clime(
    units: list[str],
    model_fn: Callable,
    scalarizer_fn: Callable,
    original_output: str,
    original_input: str,
    n_samples_ratio: int = 10,
    max_simultaneous_drops: int = 2,
    alpha: float = 0.01,
) -> np.ndarray:
    """
    Constrained LIME attribution.
    Fits a local linear model using n=10*d perturbations.
    Limits simultaneous unit drops to K=2.
    """
    from sklearn.linear_model import Ridge

    d = len(units)
    n_samples = n_samples_ratio * d
    sigma = 0.75 * np.sqrt(d)

    # Generate perturbation masks (binary vectors with at most K zeros)
    masks = np.ones((n_samples, d), dtype=int)
    for i in range(n_samples):
        n_drops = np.random.randint(1, max_simultaneous_drops + 1)
        drop_indices = np.random.choice(d, size=min(n_drops, d), replace=False)
        masks[i, drop_indices] = 0

    # Compute scalarizer scores for each perturbation
    y = np.zeros(n_samples)
    weights = np.zeros(n_samples)
    z_original = np.ones(d)

    for i in range(n_samples):
        # Build perturbed input
        perturbed_units = [units[j] for j in range(d) if masks[i, j] == 1]
        perturbed_input = " ".join(perturbed_units) if perturbed_units else " "

        y[i] = scalarizer_fn(perturbed_input, original_output)

        # Exponential kernel weight
        dist = np.sqrt(np.sum((masks[i] - z_original) ** 2))
        weights[i] = np.exp(-(dist ** 2) / (sigma ** 2))

    # Fit Ridge regression
    reg = Ridge(alpha=alpha)
    reg.fit(masks, y, sample_weight=weights)

    return reg.coef_


def explain_lshap(
    units: list[str],
    model_fn: Callable,
    scalarizer_fn: Callable,
    original_output: str,
    original_input: str,
    max_K: int = 2,
) -> np.ndarray:
    """
    Local Shapley (L-SHAP) attribution.
    Estimates Shapley values in a local neighborhood with at most K=2 drops.
    """
    d = len(units)
    scores = np.zeros(d)

    # Cache scores for efficiency
    score_cache = {}

    def get_score(mask_tuple):
        if mask_tuple not in score_cache:
            perturbed_units = [units[j] for j in range(d) if mask_tuple[j] == 1]
            perturbed_input = " ".join(perturbed_units) if perturbed_units else " "
            score_cache[mask_tuple] = scalarizer_fn(perturbed_input, original_output)
        return score_cache[mask_tuple]

    # Full input score
    full_mask = tuple([1] * d)
    full_score = get_score(full_mask)

    for s in range(d):
        marginal_contributions = []

        # Consider all subsets within distance max_K of unit s
        # For K=2, we consider: dropping s alone, and dropping s with one other unit
        # Marginal contribution = score with s present - score without s

        # 1. Marginal contribution when dropping only s
        mask_without_s = list(full_mask)
        mask_without_s[s] = 0
        score_without = get_score(tuple(mask_without_s))
        marginal_contributions.append(full_score - score_without)

        # 2. For each other unit j, marginal contribution of s given j is also dropped
        for j in range(d):
            if j == s:
                continue
            # Score with both j and s dropped
            mask_both = list(full_mask)
            mask_both[s] = 0
            mask_both[j] = 0
            score_both_dropped = get_score(tuple(mask_both))

            # Score with only j dropped
            mask_j = list(full_mask)
            mask_j[j] = 0
            score_j_dropped = get_score(tuple(mask_j))

            marginal_contributions.append(score_j_dropped - score_both_dropped)

        scores[s] = np.mean(marginal_contributions)

    return scores


# Registry for convenience
EXPLAINERS = {
    "loo": explain_loo,
    "clime": explain_clime,
    "lshap": explain_lshap,
}


if __name__ == "__main__":
    # Quick validation with synthetic data
    units = ["The cat sat on the mat.", "It was a sunny day.", "Birds were singing."]
    print(f"Units: {units}")
    print(f"Number of units: {len(units)}")

    # Dummy scalarizer for testing
    def dummy_scalarizer(perturbed_input, original_output):
        return len(perturbed_input) / 100.0  # longer input = higher score

    print("\n=== LOO ===")
    loo_scores = explain_loo(units, None, dummy_scalarizer, "summary", " ".join(units))
    print(f"  Scores: {loo_scores}")

    print("\n=== C-LIME ===")
    clime_scores = explain_clime(units, None, dummy_scalarizer, "summary", " ".join(units))
    print(f"  Scores: {clime_scores}")

    print("\n=== L-SHAP ===")
    lshap_scores = explain_lshap(units, None, dummy_scalarizer, "summary", " ".join(units))
    print(f"  Scores: {lshap_scores}")
