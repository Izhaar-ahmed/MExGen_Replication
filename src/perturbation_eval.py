"""
Perturbation evaluation: perturbation curves and AUPC computation.
Drops top-ranked units and measures scalarizer decrease.
"""

import numpy as np
from typing import Callable


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def drop_top_k_and_score(
    units: list[str],
    scores: np.ndarray,
    scalarizer_fn: Callable,
    original_output: str,
    original_input: str,
    max_pct: float = 0.20,
) -> list[tuple[float, float]]:
    """
    Iteratively remove top-ranked units and measure scalarizer decrease.

    Args:
        units: List of input text units
        scores: Attribution scores (higher = more important)
        scalarizer_fn: Function(perturbed_input, original_output) -> float
        original_output: The original model output
        original_input: The original full input text
        max_pct: Stop at this fraction of total tokens removed

    Returns:
        List of (fraction_tokens_removed, scalarizer_decrease) pairs
    """
    # Baseline score
    baseline = scalarizer_fn(original_input, original_output)

    # Total word count
    total_words = sum(count_words(u) for u in units)
    if total_words == 0:
        return [(0.0, 0.0)]

    # Sort units by score (descending — most important first)
    ranked_indices = np.argsort(scores)[::-1]

    curve_points = [(0.0, 0.0)]  # Start at (0%, 0 decrease)
    removed_words = 0
    remaining_indices = set(range(len(units)))

    for idx in ranked_indices:
        unit_words = count_words(units[idx])
        removed_words += unit_words
        remaining_indices.discard(idx)

        frac = removed_words / total_words
        if frac > max_pct:
            break

        # Build perturbed input from remaining units
        remaining_units = [units[i] for i in sorted(remaining_indices)]
        perturbed_input = " ".join(remaining_units) if remaining_units else " "

        perturbed_score = scalarizer_fn(perturbed_input, original_output)
        decrease = baseline - perturbed_score

        curve_points.append((frac, decrease))

    return curve_points


def interpolate_curve(
    curve_points: list[tuple[float, float]],
    grid: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate curve onto a common grid for averaging.

    Args:
        curve_points: List of (frac, decrease) pairs (sorted by frac)
        grid: Common x-axis grid. Default: np.linspace(0, 0.20, 21)

    Returns:
        (grid, interpolated_values)
    """
    if grid is None:
        grid = np.linspace(0, 0.20, 21)

    fracs = np.array([p[0] for p in curve_points])
    decreases = np.array([p[1] for p in curve_points])

    # Interpolate
    interp_values = np.interp(grid, fracs, decreases)
    return grid, interp_values


def aupc(curve_points: list[tuple[float, float]]) -> float:
    """
    Compute Area Under the Perturbation Curve.
    AUPC = trapz(y_values, x_fractions) * 100
    Higher is better.
    """
    if len(curve_points) < 2:
        return 0.0

    fracs = np.array([p[0] for p in curve_points])
    decreases = np.array([p[1] for p in curve_points])

    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    return float(_trapz(decreases, fracs) * 100)


def average_curves(
    all_curves: list[list[tuple[float, float]]],
    grid: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Average perturbation curves across samples.

    Returns:
        (grid, mean_values, stderr_values)
    """
    if grid is None:
        grid = np.linspace(0, 0.20, 21)

    all_interp = []
    for curve in all_curves:
        _, interp = interpolate_curve(curve, grid)
        all_interp.append(interp)

    all_interp = np.array(all_interp)
    mean_vals = np.mean(all_interp, axis=0)
    stderr_vals = np.std(all_interp, axis=0) / np.sqrt(len(all_interp))

    return grid, mean_vals, stderr_vals


if __name__ == "__main__":
    # Quick test with synthetic data
    units = ["Sentence one is important.", "Sentence two is filler.", "Sentence three matters."]
    scores = np.array([0.8, 0.1, 0.6])

    def dummy_scalarizer(perturbed_input, original_output):
        return len(perturbed_input) / 100.0

    original_input = " ".join(units)
    original_output = "Summary"

    print("=== Perturbation Curve ===")
    curve = drop_top_k_and_score(units, scores, dummy_scalarizer, original_output, original_input)
    for frac, dec in curve:
        print(f"  {frac:.2%} removed -> decrease = {dec:.4f}")

    print(f"\nAUPC = {aupc(curve):.4f}")
