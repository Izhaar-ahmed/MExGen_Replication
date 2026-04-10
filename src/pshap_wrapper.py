"""
PartitionSHAP wrapper using the `shap` library.
Sums attribution values across output tokens to get one score per input span.
"""

import numpy as np
import shap
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


def explain_pshap(
    input_text: str,
    model_wrapper,
    units: list[str],
    max_evals: int = 500,
) -> np.ndarray:
    """
    Compute PartitionSHAP attributions for input_text.

    Uses the shap library's Partition explainer with a text masker.
    Sums attribution values across all output token positions to get
    one score per input span.

    Args:
        input_text: Full input text
        model_wrapper: ModelWrapper with model and tokenizer attributes
        units: List of segmented text units (for mapping back)
        max_evals: Max number of model evaluations (for fair comparison)

    Returns:
        np.ndarray of attribution scores, one per unit
    """
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer
    device = model_wrapper.device

    def model_predict(texts):
        """Predict function for SHAP — returns log probabilities."""
        results = []
        for text in texts:
            if not text.strip():
                text = " "
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    num_beams=1,
                    do_sample=False,
                )
            # Get logits for the generated sequence
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=output_ids[:, 1:] if output_ids.shape[1] > 1 else output_ids,
                )
            results.append(-outputs.loss.item())
        return np.array(results)

    # Use SHAP's text masker
    masker = shap.maskers.Text(tokenizer)

    # Create Partition explainer
    explainer = shap.Explainer(model_predict, masker, output_names=["log_prob"])

    try:
        shap_values = explainer([input_text], max_evals=max_evals)

        # shap_values.values shape: (1, n_input_tokens)
        token_attributions = shap_values.values[0]
        if isinstance(token_attributions, np.ndarray) and token_attributions.ndim > 1:
            token_attributions = token_attributions.sum(axis=-1)

        # Map token-level attributions back to unit-level
        # by tokenizing each unit and summing the corresponding token attributions
        unit_scores = np.zeros(len(units))

        # Simple approach: distribute token attributions proportionally to units
        # based on character position mapping
        total_tokens = len(token_attributions)
        unit_token_counts = []
        for u in units:
            n_tokens = len(tokenizer.encode(u, add_special_tokens=False))
            unit_token_counts.append(n_tokens)

        total_unit_tokens = sum(unit_token_counts)

        # Assign attributions proportionally
        token_idx = 0
        for i, count in enumerate(unit_token_counts):
            # Scale to match actual token count
            scaled_count = int(count * total_tokens / max(total_unit_tokens, 1))
            end_idx = min(token_idx + max(scaled_count, 1), total_tokens)
            if token_idx < total_tokens:
                unit_scores[i] = np.sum(token_attributions[token_idx:end_idx])
            token_idx = end_idx

        return unit_scores

    except Exception as e:
        print(f"  [P-SHAP] Error: {e}, returning zeros")
        return np.zeros(len(units))


if __name__ == "__main__":
    print("P-SHAP wrapper module loaded successfully.")
    print("Run with: explain_pshap(input_text, model_wrapper, units)")
