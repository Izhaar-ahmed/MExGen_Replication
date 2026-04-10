"""
Hybrid Two-Stage Attribution: LOO Screening → L-SHAP Refinement.

Improvement over MExGen's independent methods:
  Stage 1: LOO (linear cost: d+1 calls) identifies candidate important sentences.
  Stage 2: L-SHAP (restricted to candidates only) refines importance with fewer calls.

References:
  - Jethani et al., "FastSHAP: Real-Time Shapley Value Estimation" (ICLR 2022)
  - Covert et al., "Stochastic Amortization" (NeurIPS 2024)
"""

import numpy as np
from typing import Callable, Optional


def explain_hybrid_loo_lshap(
    units: list[str],
    model_fn: Callable,
    scalarizer_fn: Callable,
    original_output: str,
    original_input: str,
    top_fraction: float = 0.7,
    max_K: int = 2,
    blend_alpha: float = 0.3,
) -> tuple[np.ndarray, dict]:
    """
    Two-stage hybrid attribution: LOO screening + restricted L-SHAP.

    Stage 1: Compute LOO scores for all units (d+1 model calls).
    Stage 2: Compute L-SHAP only for top candidates from LOO.

    Scoring strategy:
      - Candidates: blend of LOO and L-SHAP → α·LOO + (1-α)·LSHAP
      - Non-candidates: raw LOO scores (no artificial scaling)

    Args:
        units: List of input text segments.
        model_fn: Not used directly (kept for API consistency).
        scalarizer_fn: fn(perturbed_input, original_output) -> float.
        original_output: Original model output for the full input.
        original_input: The full input text.
        top_fraction: Fraction of units to promote to L-SHAP stage (default 0.7).
        max_K: Maximum number of simultaneously dropped units in L-SHAP.
        blend_alpha: Weight for LOO in the blended score (default 0.3).
                     Final = alpha * LOO_normalized + (1-alpha) * LSHAP_score.

    Returns:
        (scores, info_dict)
        scores: np.ndarray of importance scores for each unit.
        info_dict: Contains 'model_calls', 'loo_scores', 'candidate_indices'.
    """
    d = len(units)
    model_calls = 0

    # --- Caching wrapper ---
    score_cache = {}

    def get_score(mask_tuple: tuple) -> float:
        nonlocal model_calls
        if mask_tuple not in score_cache:
            perturbed_units = [units[j] for j in range(d) if mask_tuple[j] == 1]
            perturbed_input = " ".join(perturbed_units) if perturbed_units else " "
            score_cache[mask_tuple] = scalarizer_fn(perturbed_input, original_output)
            model_calls += 1
        return score_cache[mask_tuple]

    # =======================================================================
    # Stage 1: LOO — linear cost, d+1 unique model calls
    # =======================================================================
    full_mask = tuple([1] * d)
    baseline = get_score(full_mask)

    loo_scores = np.zeros(d)
    for i in range(d):
        mask_without_i = list(full_mask)
        mask_without_i[i] = 0
        score_without = get_score(tuple(mask_without_i))
        loo_scores[i] = baseline - score_without

    # Select top-k candidates (minimum 2 to capture interactions)
    k = max(2, int(d * top_fraction))
    k = min(k, d)  # can't exceed total units
    ranked = np.argsort(np.abs(loo_scores))[::-1]
    candidate_set = set(ranked[:k].tolist())

    # =======================================================================
    # Stage 2: Restricted L-SHAP — only for candidate units
    # Computes marginal contributions considering only candidate coalitions.
    # Non-candidates always remain in the context (mask=1).
    # =======================================================================
    lshap_scores = np.zeros(d)

    for s in candidate_set:
        marginal_contributions = []

        # 2a. Marginal contribution when dropping only s
        # (already cached from LOO stage — no extra call)
        mask_without_s = list(full_mask)
        mask_without_s[s] = 0
        score_without_s = get_score(tuple(mask_without_s))
        marginal_contributions.append(baseline - score_without_s)

        # 2b. For each other candidate j, marginal contribution of s
        #     given that j is also dropped. This captures interaction effects.
        for j in candidate_set:
            if j == s:
                continue

            # Score with both s and j dropped
            mask_both = list(full_mask)
            mask_both[s] = 0
            mask_both[j] = 0
            score_both_dropped = get_score(tuple(mask_both))

            # Score with only j dropped (may already be cached from LOO)
            mask_j = list(full_mask)
            mask_j[j] = 0
            score_j_dropped = get_score(tuple(mask_j))

            marginal_contributions.append(score_j_dropped - score_both_dropped)

        lshap_scores[s] = np.mean(marginal_contributions)

    # =======================================================================
    # Stage 3: Blended scoring
    #   - Candidates: α·LOO + (1-α)·LSHAP  (captures both individual & interaction)
    #   - Non-candidates: raw LOO scores    (no artificial suppression)
    # =======================================================================
    hybrid_scores = np.copy(loo_scores)  # start with LOO for everyone

    for s in candidate_set:
        hybrid_scores[s] = (
            blend_alpha * loo_scores[s]
            + (1 - blend_alpha) * lshap_scores[s]
        )

    info = {
        "model_calls": model_calls,
        "loo_scores": loo_scores,
        "lshap_scores": lshap_scores,
        "candidate_indices": sorted(candidate_set),
        "k": k,
    }

    return hybrid_scores, info


# ---------------------------------------------------------------------------
# Model call count estimators (for theoretical comparison)
# ---------------------------------------------------------------------------
def estimate_calls_loo(d: int) -> int:
    """LOO: 1 baseline + d leave-one-out = d+1."""
    return d + 1


def estimate_calls_lshap(d: int) -> int:
    """Full L-SHAP with max_K=2: d single-drops + C(d,2) pair-drops + 1 baseline."""
    return 1 + d + d * (d - 1) // 2


def estimate_calls_clime(d: int, n_samples_ratio: int = 10) -> int:
    """C-LIME: n_samples_ratio * d perturbations."""
    return n_samples_ratio * d


def estimate_calls_hybrid(d: int, top_fraction: float = 0.5) -> int:
    """Hybrid: d+1 (LOO) + C(k,2) new pairs (some cached from LOO)."""
    k = max(1, int(d * top_fraction))
    loo_calls = d + 1
    # Single-drops for candidates are cached from LOO
    # New calls = pair-drops for candidate pairs
    new_pair_calls = k * (k - 1) // 2
    return loo_calls + new_pair_calls


if __name__ == "__main__":
    # Quick validation with synthetic data
    units = [
        "The cat sat on the mat.",
        "It was a sunny day.",
        "Birds were singing.",
        "The dog ran in the park.",
        "Clouds gathered slowly.",
    ]

    def dummy_scalarizer(perturbed_input, original_output):
        # Simulate: "cat" and "mat" are the important parts
        score = len(perturbed_input) / 100.0
        if "cat" in perturbed_input:
            score += 0.5
        if "mat" in perturbed_input:
            score += 0.3
        return score

    scores, info = explain_hybrid_loo_lshap(
        units, None, dummy_scalarizer, "summary", " ".join(units),
        top_fraction=0.4,
    )

    print("=== Hybrid LOO→L-SHAP Test ===")
    for i, (u, s) in enumerate(zip(units, scores)):
        marker = " ← CANDIDATE" if i in info["candidate_indices"] else ""
        print(f"  [{i}] score={s:.4f} | {u}{marker}")
    print(f"\nModel calls: {info['model_calls']}")
    print(f"Candidates: {info['candidate_indices']}")
    print(f"LOO scores: {info['loo_scores']}")

    # Theoretical comparison
    d = len(units)
    print(f"\n--- Theoretical Call Counts (d={d}) ---")
    print(f"  LOO:    {estimate_calls_loo(d)}")
    print(f"  C-LIME: {estimate_calls_clime(d)}")
    print(f"  L-SHAP: {estimate_calls_lshap(d)}")
    print(f"  Hybrid: {estimate_calls_hybrid(d, 0.4)}")
