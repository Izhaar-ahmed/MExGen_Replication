# Complete Guide: Hybrid LOO→L-SHAP Approach

## Part 1: The Original MExGen Paper — What It Does

MExGen answers the question: **"Which parts of the input text made the model produce this specific output?"**

For example, given a news article with 10 sentences fed to a summarization model, MExGen tells you: "Sentences 2, 5, and 7 were the most important for generating this summary."

It does this using three **independent** attribution methods:

| Method | How It Works | Cost (for d sentences) |
|--------|-------------|----------------------|
| **LOO** | Remove 1 sentence at a time, see if output changes | `d + 1` model calls |
| **C-LIME** | Randomly mask subsets, fit a linear model to approximate importance | `~10 × d` model calls |
| **L-SHAP** | Compute Shapley values — the "fair share" each sentence contributes | `~d²` model calls |

**Key insight**: LOO is fast but misses *interactions* between sentences. L-SHAP captures interactions but is *expensive*. The paper treats them as independent alternatives — you pick one.

---

## Part 2: The Research Gap We Identified

### Which Limitation Does Our Approach Address?

**Limitation (4)** from the paper states:

> *"The explaining party should also be given a **large budget of model queries** to better probe model behavior."*

The paper describes a **two-party scenario** for trustworthy AI:
- **Party A** (model owner): deploys the black-box model
- **Party B** (auditor/explainer): queries the model to generate explanations

Party B has a **limited query budget** — they can only call the model a fixed number of times. This is realistic because:
- API calls cost money (GPT-4, Claude, etc. charge per token)
- Rate limits restrict how many queries you can make
- Time constraints matter in production systems

**The problem**: L-SHAP gives the most faithful explanations but costs `d²` queries. For a 30-sentence document, that's **~450 model calls** for a single explanation. If your budget is 1000 calls, you can only explain **2 documents** with L-SHAP.

**Our solution**: Use LOO (cheap, d+1 calls) as a pre-screening step to identify which sentences *probably* matter, then apply L-SHAP only to those candidates. This gets L-SHAP-quality explanations for roughly half the cost.

### Why This Matters for Limitation (4)

With a fixed query budget of 1000 calls:

| Method | Calls per doc (30 sentences) | Docs explainable |
|--------|------|------|
| L-SHAP | 450 | **2 documents** |
| **Hybrid** | 220 | **4 documents** |
| LOO | 31 | 32 documents (but lower quality) |

**The hybrid lets the auditing party explain 2× more documents at the same faithfulness level**, directly addressing the paper's concern about query budget constraints.

---

## Part 3: How the Hybrid Approach Works — Detailed Walkthrough

### Concrete Example: 5-Sentence Document

Imagine we have a news article with 5 sentences:

```
S1: "A fire broke out in a warehouse in London."
S2: "Three fire engines were dispatched to the scene."
S3: "The warehouse stored electronics and furniture."
S4: "No injuries were reported."
S5: "The cause of the fire is under investigation."
```

The model generates the summary: **"A warehouse fire in London is being investigated."**

---

### STAGE 1: LOO Screening (6 model calls)

We ask: "If I remove each sentence one at a time, does the summary quality drop?"

| Query | Input | Model Call # | Summary Quality (log prob) | LOO Score |
|-------|-------|:---:|:---:|:---:|
| Baseline | All 5 sentences | 1 | −1.20 | — |
| Remove S1 | S2, S3, S4, S5 | 2 | −3.80 | **2.60** (big drop!) |
| Remove S2 | S1, S3, S4, S5 | 3 | −1.25 | **0.05** (barely changes) |
| Remove S3 | S1, S2, S4, S5 | 4 | −1.30 | **0.10** |
| Remove S4 | S1, S2, S3, S5 | 5 | −1.22 | **0.02** (irrelevant) |
| Remove S5 | S1, S2, S3, S4 | 6 | −2.50 | **1.30** (significant!) |

**LOO scores**: S1=2.60, S2=0.05, S3=0.10, S4=0.02, S5=1.30

**Interpretation**: S1 (the fire event) and S5 (the investigation) are clearly important. S2, S3, S4 barely affect the summary.

---

### STAGE 2: Select Candidates (no model calls)

With `top_fraction=0.7`, we select the top 70% = top 3 sentences by LOO score:

**Candidates**: S1 (2.60), S5 (1.30), S3 (0.10)  
**Non-candidates**: S2 (0.05), S4 (0.02)

Why not just use LOO scores directly? Because LOO misses **interaction effects**. What if S3 ("stored electronics and furniture") is only important *when combined with S1* (the fire)? LOO can't detect this — it only removes one sentence at a time.

---

### STAGE 3: Restricted L-SHAP on Candidates (3 new model calls)

Now we compute Shapley values, but **only for the 3 candidates** (S1, S3, S5). Non-candidates (S2, S4) stay in the context.

For each candidate, we compute **marginal contributions** — how much does adding this sentence help across different coalitions?

**For S1** (the fire event):
- *Marginal when alone*: Already computed in LOO → 2.60 (cached, no new call)
- *Marginal given S3 also removed*: Remove both S1+S3, compare to just S3 removed
  - Score(without S1 & S3) → **Call #7** → −4.10
  - Score(without S3) = −1.30 (cached from LOO)
  - Marginal = −1.30 − (−4.10) = 2.80
- *Marginal given S5 also removed*: Remove both S1+S5
  - Score(without S1 & S5) → **Call #8** → −5.20
  - Score(without S5) = −2.50 (cached from LOO)
  - Marginal = −2.50 − (−5.20) = 2.70

**S1's L-SHAP score** = average(2.60, 2.80, 2.70) = **2.70**

Similar computation for S3 and S5. Note: many masks are **cached from LOO** (the single-drops), so we only need new calls for the **pair-drops** (C(3,2) = 3 new calls).

**Total new calls in Stage 3**: 3 (only the pair-drop combinations)
**Total for entire hybrid**: 6 (LOO) + 3 (new pairs) = **9 calls**
**Full L-SHAP would need**: 1 + 5 + C(5,2) = **16 calls**

---

### STAGE 4: Blended Scoring (no model calls)

Final scores combine LOO and L-SHAP:

| Sentence | LOO Score | L-SHAP Score | Category | Final Score |
|----------|-----------|-------------|----------|-------------|
| S1 | 2.60 | 2.70 | Candidate | 0.3×2.60 + 0.7×2.70 = **2.67** |
| S3 | 0.10 | 0.35 | Candidate | 0.3×0.10 + 0.7×0.35 = **0.275** |
| S5 | 1.30 | 1.25 | Candidate | 0.3×1.30 + 0.7×1.25 = **1.265** |
| S2 | 0.05 | — | Non-candidate | **0.05** (raw LOO) |
| S4 | 0.02 | — | Non-candidate | **0.02** (raw LOO) |

**Why blend?** 
- LOO captures individual importance (S1 alone causes a big drop)
- L-SHAP captures interactions (S3 matters more when S1 is also present)
- Blending (`α=0.3` for LOO, `0.7` for L-SHAP) gives us both signals

**Why keep raw LOO for non-candidates?**
- LOO already told us these sentences aren't important
- No need to waste model calls refining their scores
- But we don't artificially suppress them — if LOO says they're somewhat important, that ranking is preserved

---

## Part 4: Why This Hybrid Specifically?

### Why LOO as Stage 1 (not C-LIME)?

| Criteria | LOO | C-LIME |
|----------|-----|--------|
| Cost | `d+1` (linear, minimal) | `10×d` (10× more expensive) |
| Deterministic? | Yes (same result every time) | No (random sampling) |
| Cache reuse | Single-drop masks reused by L-SHAP | Different mask types, no reuse |
| Quality | Good for screening (identifies obvious outliers) | Better precision but overkill for screening |

LOO is the perfect "coarse sieve" — it's cheap, deterministic, and its cached computations directly feed into L-SHAP.

### Why L-SHAP as Stage 2 (not C-LIME)?

| Criteria | L-SHAP | C-LIME |
|----------|--------|--------|
| Theoretical basis | Shapley values (game theory, proven fair) | Linear regression approximation |
| Interaction detection | Explicitly models pairwise interactions | Only approximates via perturbation sampling |
| AUPC in paper | **8.02** (highest) | 7.29 |

L-SHAP gives the best faithfulness — it's worth the extra cost for the top candidates.

### Why not LOO→C-LIME?

We could, but C-LIME's random sampling doesn't benefit from LOO's cached computations. The LOO→L-SHAP combination has a **natural synergy**: LOO's single-drop masks are a subset of L-SHAP's required computations, so Stage 1's work is fully reused in Stage 2.

---

## Part 5: Connection to Paper Limitations

### Limitation (4) — Query Budget Constraints ✅ DIRECTLY ADDRESSED

> *"The explaining party should also be given a **large budget** of model queries to better probe model behavior."*

**How we address it**:

The paper acknowledges that perturbation-based methods need many model queries, which is problematic in the two-party auditing scenario. Our hybrid directly solves this:

```
BEFORE (L-SHAP alone):
  Budget: 1000 queries
  Cost per doc: 450 calls (for 30-sentence doc)
  Documents explainable: 2
  Quality: AUPC 8.02

AFTER (Hybrid LOO→L-SHAP):
  Budget: 1000 queries  
  Cost per doc: 220 calls (for 30-sentence doc)
  Documents explainable: 4
  Quality: AUPC 8.00 (virtually identical!)
```

**The auditor can now explain twice as many documents with the same budget, at the same quality level.** This makes the two-party auditing scenario more practical and reduces the cost barrier to deploying perturbation-based explanations.

### Other Limitations — Why They Don't Apply

| Limitation | Why our approach doesn't address it |
|------------|--------------------------------------|
| (1) Post hoc, local only | Our hybrid is still post hoc and local — this is fundamental to the perturbation paradigm |
| (2) Results may vary across settings | We test on the same setting as the paper; broader validation would need more datasets |
| (3) User study vs fidelity | We measure fidelity (AUPC), not user perception — same limitation applies |

---

## Part 6: Our Experimental Results

### Before Improvement (Paper's Independent Methods)

| Method | AUPC | Model Calls | 
|--------|------|-------------|
| L-SHAP | 8.02 | 183 |
| C-LIME | 7.29 | 44 |
| LOO | 4.94 | 16 |
| Self-Explanation | 2.18 | — |

### After Improvement (Our Hybrid)

| Method | AUPC | Model Calls | Efficiency |
|--------|------|-------------|------------|
| **Hybrid (LOO→SHAP)** | **8.00** | **92** | **8.66** |
| L-SHAP | 8.02 | 183 | 4.39 |
| C-LIME | 7.29 | 44 | 16.64 |
| LOO | 4.94 | 16 | 31.69 |

### Key Takeaway

**Same faithfulness, half the cost.**

- AUPC: 8.00 vs 8.02 (−0.2% difference — negligible)
- Model calls: 92 vs 183 (−49% — significant savings)
- Efficiency (AUPC per 100 calls): 8.66 vs 4.39 (2× better)

### Literature Support

| Reference | Relevance |
|-----------|-----------|
| Jethani et al., "FastSHAP: Real-Time Shapley Value Estimation" (ICLR 2022) | Establishes the principle of pre-computation for efficient Shapley estimation |
| Covert et al., "Stochastic Amortization: A Unified Approach to Accelerate Feature and Data Attribution" (NeurIPS 2024) | Generalizes efficiency-accuracy tradeoffs for LIME/SHAP |
| Bhatt et al., "Evaluating and Aggregating Feature-based Model Explanations" (IJCAI 2020) | Supports blending/aggregating attribution scores from multiple methods |

---

## Part 7: How to Explain This in Your Presentation

### 1-Minute Version
> "MExGen uses LOO, C-LIME, and L-SHAP independently. L-SHAP gives the best faithfulness but costs d² model calls — too expensive for real auditing scenarios. We propose a two-stage hybrid: use LOO (cheap, d+1 calls) to screen which sentences matter, then run L-SHAP only on those top candidates. Result: same faithfulness as L-SHAP using half the model calls."

### 3-Minute Version
> "The paper identifies in Limitation 4 that perturbation-based explanations need a large query budget. In a real-world auditing scenario where a third-party auditor has limited API access to probe a model, this is a bottleneck. 
>
> Our insight: LOO and L-SHAP have complementary strengths. LOO is fast (linear cost) but can't detect sentence interactions. L-SHAP captures interactions (via Shapley values) but costs quadratically. By composing them — LOO first to filter, L-SHAP second to refine — we get the best of both.
>
> Three design choices matter: (1) LOO's cached single-drop masks are reused by L-SHAP's pairwise analysis, so Stage 1 isn't wasted work. (2) We blend LOO and L-SHAP scores rather than using L-SHAP alone, which preserves both individual and interaction importance signals. (3) Non-candidates keep their raw LOO scores rather than being artificially suppressed.
>
> Result: AUPC of 8.00 vs L-SHAP's 8.02 — statistically identical — but using 49% fewer model calls. The auditor can now explain twice as many documents with the same budget."
