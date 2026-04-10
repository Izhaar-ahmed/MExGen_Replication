"""
Data loader for XSUM and SQuAD datasets.
Loads, formats, and caches data samples for the MExGen replication.
"""

import json
import os
from pathlib import Path
from datasets import load_dataset


CACHE_DIR = Path(__file__).parent.parent / "results" / "raw"


def load_xsum(n_samples: int = 200, split: str = "test") -> list[dict]:
    """Load XSUM samples. Returns list of dicts with 'input' and 'output' keys."""
    cache_path = CACHE_DIR / f"xsum_{split}_{n_samples}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    ds = load_dataset("EdinburghNLP/xsum", split=f"{split}[:{n_samples}]", trust_remote_code=True)
    samples = []
    for item in ds:
        samples.append({
            "input": item["document"],
            "output": item["summary"],
            "dataset": "xsum",
        })

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(samples, f, indent=2)
    return samples


def load_squad(n_samples: int = 200, split: str = "validation") -> list[dict]:
    """Load SQuAD samples. Returns list of dicts with 'input', 'question', 'output' keys."""
    cache_path = CACHE_DIR / f"squad_{split}_{n_samples}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    ds = load_dataset("rajpurkar/squad", split=f"{split}[:{n_samples}]", trust_remote_code=True)
    samples = []
    for item in ds:
        samples.append({
            "input": item["context"],
            "question": item["question"],
            "output": item["answers"]["text"][0] if item["answers"]["text"] else "",
            "dataset": "squad",
        })

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(samples, f, indent=2)
    return samples


if __name__ == "__main__":
    print("=== Loading XSUM ===")
    xsum = load_xsum(n_samples=3)
    for i, s in enumerate(xsum):
        print(f"\n--- XSUM Sample {i} ---")
        print(f"Input (first 200 chars): {s['input'][:200]}...")
        print(f"Output: {s['output'][:200]}")

    print("\n=== Loading SQuAD ===")
    squad = load_squad(n_samples=3)
    for i, s in enumerate(squad):
        print(f"\n--- SQuAD Sample {i} ---")
        print(f"Context (first 200 chars): {s['input'][:200]}...")
        print(f"Question: {s['question']}")
        print(f"Answer: {s['output']}")
