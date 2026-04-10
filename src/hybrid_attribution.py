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
    Uses a FIXED top_fraction to select candidates.

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

    Returns:
        (scores, info_dict)
        scores: np.ndarray of importance scores for each unit.
        info_dict: Contains 'model_calls', 'loo_scores', 'candidate_indices'.
    """
    d = len(units)

    # Stage 1: LOO
    loo_scores, baseline, score_cache, model_calls = _run_loo(
        units, scalarizer_fn, original_output
    )

    # Candidate selection: fixed top-k
    k = max(2, int(d * top_fraction))
    k = min(k, d)
    ranked = np.argsort(np.abs(loo_scores))[::-1]
    candidate_set = set(ranked[:k].tolist())

    # Stage 2 + 3: Restricted L-SHAP + blending
    hybrid_scores, lshap_scores, model_calls = _run_restricted_lshap_and_blend(
        units, scalarizer_fn, original_output,
        loo_scores, baseline, candidate_set,
        score_cache, model_calls, blend_alpha,
    )

    info = {
        "model_calls": model_calls,
        "loo_scores": loo_scores,
        "lshap_scores": lshap_scores,
        "candidate_indices": sorted(candidate_set),
        "k": len(candidate_set),
        "selection_mode": "fixed",
        "top_fraction": top_fraction,
    }

    return hybrid_scores, info


def explain_hybrid_dynamic(
    units: list[str],
    model_fn: Callable,
    scalarizer_fn: Callable,
    original_output: str,
    original_input: str,
    threshold_alpha: float = 0.5,
    max_K: int = 2,
    blend_alpha: float = 0.3,
) -> tuple[np.ndarray, dict]:
    """
    Two-stage hybrid attribution with DYNAMIC candidate selection.

    Instead of a fixed top-k fraction, uses the statistical distribution of
    LOO scores to adaptively select candidates:
        candidate if |LOO_score| > mean(|LOO_scores|) + α · std(|LOO_scores|)

    Args:
        units: List of input text segments.
        model_fn: Not used directly (kept for API consistency).
        scalarizer_fn: fn(perturbed_input, original_output) -> float.
        original_output: Original model output for the full input.
        original_input: The full input text.
        threshold_alpha: Sensitivity parameter for the threshold (default 0.5).
        max_K: Maximum number of simultaneously dropped units in L-SHAP.
        blend_alpha: Weight for LOO in the blended score (default 0.3).

    Returns:
        (scores, info_dict)
        scores: np.ndarray of importance scores for each unit.
        info_dict: Contains selection stats including 'threshold', 'k', 'selection_mode'.
    """
    d = len(units)

    # Stage 1: LOO
    loo_scores, baseline, score_cache, model_calls = _run_loo(
        units, scalarizer_fn, original_output
    )

    # -------------------------------------------------------------------
    # Dynamic candidate selection using statistical threshold
    # -------------------------------------------------------------------
    abs_scores = np.abs(loo_scores)
    score_mean = np.mean(abs_scores)
    score_std = np.std(abs_scores)
    threshold = score_mean + threshold_alpha * score_std

    # Select all sentences above the threshold
    candidate_set = set(i for i in range(d) if abs_scores[i] > threshold)

    # Ensure at least 2 candidates (need ≥2 for interaction analysis)
    if len(candidate_set) < 2:
        ranked = np.argsort(abs_scores)[::-1]
        candidate_set = set(ranked[:min(2, d)].tolist())

    # For very short documents (≤4 sentences), use all as candidates
    if d <= 4:
        candidate_set = set(range(d))

    # Stage 2 + 3: Restricted L-SHAP + blending
    hybrid_scores, lshap_scores, model_calls = _run_restricted_lshap_and_blend(
        units, scalarizer_fn, original_output,
        loo_scores, baseline, candidate_set,
        score_cache, model_calls, blend_alpha,
    )

    info = {
        "model_calls": model_calls,
        "loo_scores": loo_scores,
        "lshap_scores": lshap_scores,
        "candidate_indices": sorted(candidate_set),
        "k": len(candidate_set),
        "selection_mode": "dynamic",
        "threshold_alpha": threshold_alpha,
        "threshold_value": float(threshold),
        "score_mean": float(score_mean),
        "score_std": float(score_std),
    }

    return hybrid_scores, info


# ---------------------------------------------------------------------------
# Shared internal helpers
# ---------------------------------------------------------------------------
def _run_loo(
    units: list[str],
    scalarizer_fn: Callable,
    original_output: str,
) -> tuple[np.ndarray, float, dict, int]:
    """Run LOO screening. Returns (loo_scores, baseline, score_cache, model_calls)."""
    d = len(units)
    model_calls = 0
    score_cache = {}

    def get_score(mask_tuple):
        nonlocal model_calls
        if mask_tuple not in score_cache:
            perturbed_units = [units[j] for j in range(d) if mask_tuple[j] == 1]
            perturbed_input = " ".join(perturbed_units) if perturbed_units else " "
            score_cache[mask_tuple] = scalarizer_fn(perturbed_input, original_output)
            model_calls += 1
        return score_cache[mask_tuple]

    full_mask = tuple([1] * d)
    baseline = get_score(full_mask)

    loo_scores = np.zeros(d)
    for i in range(d):
        mask_without_i = list(full_mask)
        mask_without_i[i] = 0
        score_without = get_score(tuple(mask_without_i))
        loo_scores[i] = baseline - score_without

    return loo_scores, baseline, score_cache, model_calls


def _run_restricted_lshap_and_blend(
    units: list[str],
    scalarizer_fn: Callable,
    original_output: str,
    loo_scores: np.ndarray,
    baseline: float,
    candidate_set: set,
    score_cache: dict,
    model_calls: int,
    blend_alpha: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run restricted L-SHAP on candidates and blend scores.
    Returns (hybrid_scores, lshap_scores, model_calls)."""
    d = len(units)
    full_mask = tuple([1] * d)

    def get_score(mask_tuple):
        nonlocal model_calls
        if mask_tuple not in score_cache:
            perturbed_units = [units[j] for j in range(d) if mask_tuple[j] == 1]
            perturbed_input = " ".join(perturbed_units) if perturbed_units else " "
            score_cache[mask_tuple] = scalarizer_fn(perturbed_input, original_output)
            model_calls += 1
        return score_cache[mask_tuple]

    # Restricted L-SHAP for candidates only
    lshap_scores = np.zeros(d)

    for s in candidate_set:
        marginal_contributions = []

        mask_without_s = list(full_mask)
        mask_without_s[s] = 0
        score_without_s = get_score(tuple(mask_without_s))
        marginal_contributions.append(baseline - score_without_s)

        for j in candidate_set:
            if j == s:
                continue
            mask_both = list(full_mask)
            mask_both[s] = 0
            mask_both[j] = 0
            score_both_dropped = get_score(tuple(mask_both))

            mask_j = list(full_mask)
            mask_j[j] = 0
            score_j_dropped = get_score(tuple(mask_j))

            marginal_contributions.append(score_j_dropped - score_both_dropped)

        lshap_scores[s] = np.mean(marginal_contributions)

    # Blended scoring
    hybrid_scores = np.copy(loo_scores)
    for s in candidate_set:
        hybrid_scores[s] = (
            blend_alpha * loo_scores[s]
            + (1 - blend_alpha) * lshap_scores[s]
        )

    return hybrid_scores, lshap_scores, model_calls


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


def estimate_calls_hybrid(d: int, top_fraction: float = 0.7) -> int:
    """Hybrid (fixed): d+1 (LOO) + C(k,2) new pairs."""
    k = max(2, int(d * top_fraction))
    loo_calls = d + 1
    new_pair_calls = k * (k - 1) // 2
    return loo_calls + new_pair_calls


if __name__ == "__main__":
    # Quick validation with synthetic data
    units = [
        "A fire broke out in a warehouse in London.",
        "Three fire engines were dispatched to the scene.",
        "The warehouse stored electronics and furniture.",
        "No injuries were reported by authorities.",
        "The cause of the fire is under investigation.",
        "Local residents were evacuated from nearby homes.",
        "The fire was contained after several hours.",
        "Insurance claims are expected to be significant.",
    ]

    def dummy_scalarizer(perturbed_input, original_output):
        score = -2.0
        if "fire" in perturbed_input and "warehouse" in perturbed_input:
            score += 0.8  # S1 is very important
        if "investigation" in perturbed_input:
            score += 0.4  # S5 matters
        if "contained" in perturbed_input:
            score += 0.2  # S7 somewhat
        return score

    print("=" * 60)
    print("FIXED MODE (top_fraction=0.7)")
    print("=" * 60)
    scores_fixed, info_fixed = explain_hybrid_loo_lshap(
        units, None, dummy_scalarizer, "summary", " ".join(units),
    )
    for i, (u, s) in enumerate(zip(units, scores_fixed)):
        marker = " ← CANDIDATE" if i in info_fixed["candidate_indices"] else ""
        print(f"  [{i}] score={s:+.4f} | {u[:50]}{marker}")
    print(f"  Calls: {info_fixed['model_calls']}, k={info_fixed['k']}")

    print(f"\n{'=' * 60}")
    print("DYNAMIC MODE (threshold_alpha=0.5)")
    print("=" * 60)
    scores_dyn, info_dyn = explain_hybrid_dynamic(
        units, None, dummy_scalarizer, "summary", " ".join(units),
    )
    for i, (u, s) in enumerate(zip(units, scores_dyn)):
        marker = " ← CANDIDATE" if i in info_dyn["candidate_indices"] else ""
        print(f"  [{i}] score={s:+.4f} | {u[:50]}{marker}")
    print(f"  Calls: {info_dyn['model_calls']}, k={info_dyn['k']}")
    print(f"  Threshold: {info_dyn['threshold_value']:.4f} "
          f"(mean={info_dyn['score_mean']:.4f}, std={info_dyn['score_std']:.4f})")
    print(f"  Savings vs fixed: {info_fixed['model_calls'] - info_dyn['model_calls']} fewer calls")
