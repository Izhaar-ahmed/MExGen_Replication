"""
Scalarizers for comparing perturbed model outputs to original outputs.
All scalarizers follow the signature: fn(perturbed_out, original_out, **kwargs) -> float
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ---------------------------------------------------------------------------
# Log Prob scalarizer
# ---------------------------------------------------------------------------
def log_prob_scalarizer(
    perturbed_input: str,
    original_output: str,
    model_wrapper=None,
) -> float:
    """
    Average log probability of generating original_output given perturbed_input.
    Requires model_wrapper with a log_prob method.
    """
    if model_wrapper is None:
        raise ValueError("log_prob_scalarizer requires a model_wrapper argument")
    return model_wrapper.log_prob(perturbed_input, original_output)


# ---------------------------------------------------------------------------
# BERTScore scalarizer
# ---------------------------------------------------------------------------
_bertscore_cache = {}


def bert_scalarizer(
    perturbed_output: str,
    original_output: str,
    **kwargs,
) -> float:
    """BERTScore F1 between perturbed output and original output."""
    from bert_score import score as bert_score_fn

    P, R, F1 = bert_score_fn(
        [perturbed_output],
        [original_output],
        model_type="bert-base-uncased",
        verbose=False,
    )
    return F1.item()


# ---------------------------------------------------------------------------
# BARTScore scalarizer
# ---------------------------------------------------------------------------
_bart_model = None
_bart_tokenizer = None


def _get_bart_scorer():
    global _bart_model, _bart_tokenizer
    if _bart_model is None:
        from src.model_wrapper import select_device
        device = select_device()
        _bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        _bart_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn").to(device)
        _bart_model.eval()
    return _bart_model, _bart_tokenizer


def bart_scalarizer(
    perturbed_output: str,
    original_output: str,
    **kwargs,
) -> float:
    """
    BARTScore: log prob of original_output given perturbed_output as source,
    using facebook/bart-large-cnn.
    """
    model, tokenizer = _get_bart_scorer()
    device = next(model.parameters()).device

    inputs = tokenizer(perturbed_output, return_tensors="pt", truncation=True, max_length=1024).to(device)
    labels = tokenizer(original_output, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels["input_ids"],
        )
    return -outputs.loss.item()


# ---------------------------------------------------------------------------
# SUMM scalarizer  (uses DistilBART as auxiliary model)
# ---------------------------------------------------------------------------
_summ_model = None
_summ_tokenizer = None


def _get_summ_scorer():
    global _summ_model, _summ_tokenizer
    if _summ_model is None:
        from src.model_wrapper import select_device
        device = select_device()
        _summ_tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-xsum-12-6", add_prefix_space=True)
        _summ_model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-xsum-12-6").to(device)
        _summ_model.eval()
    return _summ_model, _summ_tokenizer


def summ_scalarizer(
    perturbed_output: str,
    original_output: str,
    **kwargs,
) -> float:
    """
    SUMM scalarizer: log prob of original_output given perturbed_output,
    using DistilBART (same model used for XSUM summarization).
    """
    model, tokenizer = _get_summ_scorer()
    device = next(model.parameters()).device

    inputs = tokenizer(perturbed_output, return_tensors="pt", truncation=True, max_length=1024).to(device)
    labels = tokenizer(original_output, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels["input_ids"],
        )
    return -outputs.loss.item()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
SCALARIZERS = {
    "log_prob": log_prob_scalarizer,
    "bert": bert_scalarizer,
    "bart": bart_scalarizer,
    "summ": summ_scalarizer,
}


if __name__ == "__main__":
    orig = "A cat sat on the mat."
    pert = "A cat was on the mat."

    print("=== BERTScore ===")
    print(f"  F1 = {bert_scalarizer(pert, orig):.4f}")

    print("\n=== BARTScore ===")
    print(f"  Score = {bart_scalarizer(pert, orig):.4f}")

    print("\n=== SUMM Score ===")
    print(f"  Score = {summ_scalarizer(pert, orig):.4f}")

    print("\n=== Log Prob (requires model) ===")
    from src.model_wrapper import load_distilbart
    m = load_distilbart()
    doc = "The weather is sunny today. Birds are flying."
    out = m.generate(doc)
    print(f"  Score = {log_prob_scalarizer(doc, out, model_wrapper=m):.4f}")
