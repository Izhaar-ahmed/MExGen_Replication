"""
LLM Self-Explanation: prompt the model to rank input units by importance.
Uses Flan-T5-Large for both XSUM and SQuAD.
"""

import re
import random
import numpy as np
from typing import Optional


PROMPT_SUMMARIZATION = """You generated the following summary from the document below.

Document (with numbered sentences):
{numbered_units}

Generated summary:
{generated_summary}

Task: Rank the sentence numbers from MOST important to LEAST important in causing the summary above. Return ONLY a comma-separated list of numbers, e.g.: 3,1,5,2,4

Ranking:"""


PROMPT_QA = """You answered a question using the context below.

Context (with numbered sentences):
{numbered_units}

Question: {question}
Your answer: {generated_answer}

Task: Rank the sentence numbers from MOST important to LEAST important for producing your answer. Return ONLY a comma-separated list of numbers, e.g.: 2,4,1,3

Ranking:"""


def number_units(units: list[str]) -> str:
    """Number each unit with [1], [2], etc."""
    return " ".join(f"[{i + 1}] {u}" for i, u in enumerate(units))


def parse_ranking(raw_output: str, d: int) -> list[int]:
    """
    Parse model output to extract ranking.
    Returns a list of 0-indexed unit positions ordered by importance.
    Falls back to random order for missing/invalid numbers.
    """
    numbers = re.findall(r'\d+', raw_output)
    ranking = []
    seen = set()

    for n in numbers:
        idx = int(n) - 1  # Convert to 0-indexed
        if 0 <= idx < d and idx not in seen:
            ranking.append(idx)
            seen.add(idx)
        if len(ranking) == d:
            break

    # Fill missing ranks randomly
    missing = [i for i in range(d) if i not in seen]
    random.shuffle(missing)
    ranking.extend(missing)

    return ranking


def rank_to_scores(ranking: list[int], d: int) -> np.ndarray:
    """
    Convert rank positions to attribution scores.
    score_i = (d - rank_position_i) / (d - 1)
    Highest-ranked unit gets 1.0, lowest gets 0.0.
    """
    scores = np.zeros(d)
    for pos, unit_idx in enumerate(ranking):
        if d > 1:
            scores[unit_idx] = (d - pos - 1) / (d - 1)
        else:
            scores[unit_idx] = 1.0
    return scores


def self_explain(
    units: list[str],
    generated_output: str,
    model_wrapper,
    task: str = "summarization",
    question: Optional[str] = None,
) -> tuple[np.ndarray, bool]:
    """
    Get self-explanation scores for input units.

    Args:
        units: List of input text units
        generated_output: The model's generated output text
        model_wrapper: ModelWrapper instance (Flan-T5-Large)
        task: 'summarization' or 'qa'
        question: Question text (for QA task only)

    Returns:
        (scores, success): Attribution scores array and whether parsing was fully successful
    """
    d = len(units)
    numbered = number_units(units)

    if task == "summarization":
        prompt = PROMPT_SUMMARIZATION.format(
            numbered_units=numbered,
            generated_summary=generated_output,
        )
    elif task == "qa":
        prompt = PROMPT_QA.format(
            numbered_units=numbered,
            question=question or "",
            generated_answer=generated_output,
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    # Generate ranking
    raw_output = model_wrapper.generate(prompt, max_new_tokens=100)
    ranking = parse_ranking(raw_output, d)

    # Check if we got a complete parse
    success = (len(re.findall(r'\d+', raw_output)) >= d)

    scores = rank_to_scores(ranking, d)
    return scores, success


if __name__ == "__main__":
    # Test parsing
    print("=== Test parse_ranking ===")
    r1 = parse_ranking("3,1,5,2,4", 5)
    print(f"  Input: '3,1,5,2,4' -> {r1}")

    r2 = parse_ranking("3, 1", 4)
    print(f"  Input: '3, 1' (d=4) -> {r2} (partially random)")

    r3 = parse_ranking("garbage text", 3)
    print(f"  Input: 'garbage text' (d=3) -> {r3} (fully random)")

    print("\n=== Test rank_to_scores ===")
    scores = rank_to_scores([2, 0, 1], 3)
    print(f"  Ranking [2, 0, 1] -> scores {scores}")
    # Unit 2 gets score 1.0, unit 0 gets 0.5, unit 1 gets 0.0

    print("\n=== Test number_units ===")
    print(number_units(["Hello world.", "How are you?", "Fine thanks."]))
