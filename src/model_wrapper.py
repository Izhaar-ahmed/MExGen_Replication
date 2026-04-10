"""
Model wrappers for DistilBART (XSUM summarization) and Flan-T5-Large (SQuAD QA).
Exposes generate() and log_prob() methods.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
)


def select_device() -> torch.device:
    """Auto-select the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = select_device()


class ModelWrapper:
    """Unified wrapper for seq2seq models."""

    def __init__(self, model_name: str, device: torch.device = DEVICE):
        self.model_name = model_name
        self.device = device

        if "flan-t5" in model_name.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

        self.model.eval()

    def generate(self, text: str, max_new_tokens: int = 128) -> str:
        """Generate output text from input text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def log_prob(self, input_text: str, output_text: str) -> float:
        """Compute average log probability of output_text given input_text."""
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        labels = self.tokenizer(output_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels["input_ids"],
            )
        # outputs.loss is average negative log-likelihood per token
        # We return the average log probability (negative of loss)
        return -outputs.loss.item()

    def generate_for_qa(self, context: str, question: str, max_new_tokens: int = 64) -> str:
        """Generate answer for QA task with prompt template."""
        prompt = (
            f"Answer the question based on the context.\n"
            f"Context: {context}\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        return self.generate(prompt, max_new_tokens=max_new_tokens)


# Convenience factory functions
def load_distilbart(device: torch.device = DEVICE) -> ModelWrapper:
    return ModelWrapper("sshleifer/distilbart-xsum-12-6", device)


def load_flan_t5_large(device: torch.device = DEVICE) -> ModelWrapper:
    return ModelWrapper("google/flan-t5-large", device)


if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    print("\n=== DistilBART ===")
    m1 = load_distilbart()
    doc = "The cat sat on the mat. It was a sunny day. The birds were singing outside."
    gen = m1.generate(doc)
    lp = m1.log_prob(doc, gen)
    print(f"Input: {doc}")
    print(f"Generated: {gen}")
    print(f"Log prob: {lp:.4f}")

    print("\n=== Flan-T5-Large ===")
    m2 = load_flan_t5_large()
    ctx = "Albert Einstein was born in Ulm, Germany in 1879."
    q = "Where was Albert Einstein born?"
    ans = m2.generate_for_qa(ctx, q)
    lp2 = m2.log_prob(f"Answer the question based on the context.\nContext: {ctx}\nQuestion: {q}\nAnswer:", ans)
    print(f"Context: {ctx}")
    print(f"Question: {q}")
    print(f"Answer: {ans}")
    print(f"Log prob: {lp2:.4f}")
