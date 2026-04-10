# MExGen: Multi-Level Explanations for Generative Models
## Comprehensive Project Overview (ACL 2025 Replication)

This document is a "Master Summary" designed to explain everything happening in this replication project—from the underlying theory to the specific implementation details.

---

### 1. The Big Picture: What is MExGen?
Generative Language Models (like GPT, BART, or T5) are "black boxes." While they produce impressive summaries, we often don't know **why** they chose specific words.
- **Problem**: Asking the model to explain itself ("Introspection") is often unreliable.
- **Solution**: **MExGen** is a framework that systematically pokes the model with "What if?" questions to discover exactly which parts of the input text it truly relies on.

---

### 2. The Step-by-Step Process (for One Document)
Every time we process a document, we follow this sequence:

1. **Segmentation**: We split the long input document into "units" (usually sentences). These are our features.
2. **Perturbation**: We create dozens of "broken" versions of the document by randomly removing sentences.
3. **Observation**: We send each broken version through the model and see how the summary changes.
4. **Scoring (Scalarizers)**: We measure that change using three "lenses":
   - **Log Prob**: Does the model lose confidence in the original summary?
   - **BERTScore**: Does the summary's meaning stay the same?
   - **BARTScore**: Does the summary's logic and fluency break?
5. **Attribution**: We use math to calculate **Importance Scores** for each sentence based on how much its removal hurt the summary.

---

### 3. The Three Attribution "Detectives"
We use three different algorithms to calculate weights:
- **LOO (Leave-One-Out)**: Simply deletes one sentence at a time. It’s the easiest to understand but misses "teamwork" between sentences.
- **C-LIME (Constrained LIME)**: Creates a "mini-model" to approximate the big model's behavior. It’s very good at finding the primary focus of the model.
- **L-SHAP (Local Shapley)**: Based on **Game Theory**. It treats each sentence as a "player" and calculates its "fair share" of the summary's quality. It is widely considered the most mathematically rigorous method.

---

### 4. How We Measure Success: AUPC
**AUPC (Area Under the Perturbation Curve)** is our grading system for the methods.
- We remove sentences starting with the "most important" one.
- If the summary quality drops **fast**, it means the method correctly identified the most important parts.
- A **higher AUPC score** indicates a more accurate attribution method.

---

### 5. Spearman Rank Correlation: The Consistency Check
We compute Spearman coefficients to see if our different methods (C-LIME vs L-SHAP) or different lenses (LogProb vs BERT) **agree** on what's important. 
- A high score (e.g., 0.85) means both methods are identifying the same information, giving us high confidence in the results.

---

### 6. The "Self-Explanation" Baseline
The project includes a critical comparison:
- **Algorithmic Explanation (MExGen)** vs. **LLM Self-Explanation**.
- We use **Flan-T5-Large** to "explain itself" by asking it to rank its own input importance.
- **Key Result**: MExGen methods (L-SHAP/C-LIME) consistently beat Self-Explanation by a large margin (~3-4× better AUPC), showing that systematic testing is better than model introspection.

---

### 7. Implementation Technicalities
- **Toolkit**: IBM ICX360.
- **Dataset**: XSUM (Extreme Summarization).
- **Inference**: Uses Apple's **MPS (Metal Performance Shaders)** for GPU-accelerated generation.
- **Pipeline**: Fully automated Python scripts (`run_fast_demo.py`) that handle everything from data loading to final plot generation.
