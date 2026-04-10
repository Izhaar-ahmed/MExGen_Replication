# MExGen Replication & Improvement — Hybrid LOO→L-SHAP Attribution

> **Paper**: [Multi-Level Explanations for Generative Language Models](https://aclanthology.org/2025.acl-long.1553.pdf) (ACL 2025)  
> **Original Toolkit**: [IBM ICX360](https://github.com/IBM/ICX360)

This project has two parts:
1. **Replication** of the MExGen paper's key results on XSUM summarization
2. **Improvement** via a novel two-stage hybrid attribution method (LOO→L-SHAP) that matches the best method's faithfulness while using 49% fewer model calls

---

## Table of Contents

- [Background: What Is MExGen?](#background-what-is-mexgen)
- [Part 1: Paper Replication](#part-1-paper-replication)
- [Part 2: Our Improvement — Hybrid LOO→L-SHAP](#part-2-our-improvement--hybrid-lool-shap)
  - [Research Gap](#research-gap)
  - [How the Hybrid Works](#how-the-hybrid-works)
  - [Example Walkthrough](#example-walkthrough)
  - [Experimental Results](#experimental-results)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Experiments](#running-the-experiments)
- [References](#references)

---

## Background: What Is MExGen?

Generative Language Models (GPT, BART, T5) are "black boxes" — they produce impressive summaries and answers, but we don't know **why** they chose specific words. MExGen answers:

> **"Which parts of the input text made the model produce this specific output?"**

It does this using **perturbation-based attribution**: systematically remove parts of the input and observe how the output changes. If removing sentence 3 causes the summary to break, then sentence 3 was important.

### The Three Attribution Methods

| Method | How It Works | Cost | Quality |
|--------|-------------|------|---------|
| **LOO** (Leave-One-Out) | Remove 1 sentence at a time | `d+1` calls | ⭐⭐ Basic |
| **C-LIME** (Constrained LIME) | Randomly mask subsets, fit linear model | `~10×d` calls | ⭐⭐⭐ Good |
| **L-SHAP** (Local Shapley) | Compute fair-share contribution via game theory | `~d²` calls | ⭐⭐⭐⭐ Best |

### The Evaluation Metric: AUPC

**AUPC** (Area Under the Perturbation Curve) measures faithfulness:
1. Remove sentences starting from the "most important" one
2. If model confidence drops **fast**, the method correctly identified critical content
3. **Higher AUPC = better attribution**

### Scalarizers

MExGen converts text outputs to numbers using "scalarizers":
- **Log Prob**: Does the model lose confidence in the original output?
- **BERTScore**: Does the output's meaning change?
- **BARTScore**: Does the output's logic/fluency break?

---

## Part 1: Paper Replication

We replicated the paper's core experiments on **XSUM** (extreme summarization) using:
- **Model**: DistilBART (`sshleifer/distilbart-xsum-12-6`) for generation
- **Evaluation Model**: Flan-T5-Large for self-explanation baseline
- **Segmentation**: Sentence-level via spaCy
- **Primary metric**: AUPC with log_prob scalarizer

### Replicated Results

| Method | AUPC (×100) | Model Calls (avg) |
|--------|------------|-------------------|
| L-SHAP | **8.02** | 183 |
| C-LIME | 7.29 | 44 |
| LOO | 4.94 | 16 |
| Self-Explanation | 2.18 | — |

**Key findings confirmed:**
1. **L-SHAP and C-LIME** are the most faithful attribution methods
2. **LLM Self-Explanation** (asking the model "what was important?") is highly unreliable — 3-4× worse than systematic attribution
3. High **Spearman correlation** (0.82–0.94) between methods confirms they identify similar important sentences

---

## Part 2: Our Improvement — Hybrid LOO→L-SHAP

### Research Gap

The paper's **Limitation (4)** states:

> *"The explaining party should also be given a **large budget of model queries** to better probe model behavior."*

MExGen describes a **two-party auditing scenario** for trustworthy AI:
- **Party A** (model owner): deploys the black-box model
- **Party B** (auditor): queries the model to generate explanations

The problem: L-SHAP gives the best faithfulness but costs `d²` queries. For a 30-sentence document, that's ~450 model calls for a single explanation. **If the auditor's budget is limited, they can only explain a few documents.**

The paper treats LOO, C-LIME, and L-SHAP as **independent alternatives** — you pick one. It never explores **composing** them to get better efficiency.

### Our Solution

A **two-stage hybrid** that composes LOO and L-SHAP:

```
Stage 1: LOO Screening     → d+1 calls (cheap, identifies candidates)
Stage 2: Restricted L-SHAP → only on top 70% candidates (precise, captures interactions)
Stage 3: Blended Scoring   → α·LOO + (1-α)·LSHAP for candidates, raw LOO for rest
```

**Result**: Same faithfulness as L-SHAP (AUPC 8.00 vs 8.02), **49% fewer model calls** (92 vs 183).

**Literature support**:
- Jethani et al., *FastSHAP: Real-Time Shapley Value Estimation* (ICLR 2022)
- Covert et al., *Stochastic Amortization* (NeurIPS 2024)

### How the Hybrid Works

#### Stage 1: LOO Screening (d+1 model calls)

Remove each sentence one at a time. Sentences whose removal causes a big quality drop are probably important.

```
Full text → Model → Quality = -1.20 (baseline)
Remove S1 → Model → Quality = -3.80 → LOO score = 2.60 (important!)
Remove S2 → Model → Quality = -1.25 → LOO score = 0.05 (not important)
Remove S3 → Model → Quality = -1.30 → LOO score = 0.10
...
```

LOO is fast (linear cost) but misses **interaction effects** — it can't detect that sentence A is only important when combined with sentence B.

#### Stage 2: Candidate Selection (no model calls)

Select the top 70% of sentences by LOO score as "candidates" for L-SHAP refinement. The remaining 30% keep their LOO scores.

#### Stage 3: Restricted L-SHAP (only new pair-drop calls)

For each candidate, compute **Shapley marginal contributions** — but only considering other candidates. This captures interactions between the most promising sentences.

**Key efficiency trick**: LOO already computed all single-drop masks. L-SHAP needs those plus pair-drop masks. The single-drops are **cached from Stage 1**, so Stage 3 only needs `C(k,2)` new model calls for the pair-drops.

```
Total calls = (d+1) + C(k,2)    where k = 0.7×d
Full L-SHAP = 1 + d + C(d,2)    (much larger)
```

#### Stage 4: Blended Scoring (no model calls)

```
Candidates:     score = 0.3 × LOO + 0.7 × L-SHAP
Non-candidates: score = LOO (raw, no suppression)
```

Blending preserves both the **individual importance** signal from LOO and the **interaction importance** signal from L-SHAP.

### Example Walkthrough

Given 5 sentences about a warehouse fire:

```
S1: "A fire broke out in a warehouse in London."
S2: "Three fire engines were dispatched."
S3: "The warehouse stored electronics."
S4: "No injuries were reported."
S5: "The cause is under investigation."
```

Summary: _"A warehouse fire in London is being investigated."_

| Stage | Action | Model Calls | Result |
|-------|--------|:-----------:|--------|
| 1 | LOO screening | 6 | S1=2.60, S5=1.30, S3=0.10, S2=0.05, S4=0.02 |
| 2 | Select top 70% | 0 | Candidates: S1, S5, S3 |
| 3 | L-SHAP on candidates | 3 (pair-drops only) | Refined: S1=2.70, S5=1.25, S3=0.35 |
| 4 | Blend scores | 0 | S1=2.67, S5=1.27, S3=0.28, S2=0.05, S4=0.02 |
| **Total** | | **9 calls** | (Full L-SHAP would need 16) |

### Experimental Results

#### AUPC Comparison (Faithfulness)

| Method | AUPC (×100) | vs L-SHAP |
|--------|------------|-----------|
| **L-SHAP** | **8.02** ± 5.20 | — |
| **Hybrid (LOO→SHAP)** | **8.00** ± 5.19 | **−0.2%** |
| C-LIME | 7.29 ± 4.78 | −9.1% |
| LOO | 4.94 ± 2.87 | −38.4% |
| Self-Explanation | 2.18 ± 1.13 | −72.8% |

#### Efficiency Comparison

| Method | Avg Calls/Sample | AUPC per 100 Calls | vs L-SHAP Efficiency |
|--------|:----------------:|:-------------------:|:--------------------:|
| LOO | 16 | 31.69 | 7.2× |
| C-LIME | 44 | 16.64 | 3.8× |
| **Hybrid** | **92** | **8.66** | **2.0×** |
| L-SHAP | 183 | 4.39 | 1.0× |

#### Key Result

> **Hybrid matches L-SHAP's faithfulness (8.00 vs 8.02) while using 49% fewer model calls.**
> 
> With a fixed query budget:
> - L-SHAP: explains **2 documents** per 1000 calls
> - Hybrid: explains **4 documents** per 1000 calls — **at the same quality**

#### Per-Sample Breakdown

| Sample | Sentences | L-SHAP Calls | Hybrid Calls | Savings |
|:------:|:---------:|:------------:|:------------:|:-------:|
| 1 | 28 | 407 | 200 | **51%** |
| 2 | 3 | 7 | 5 | 29% |
| 3 | 10 | 56 | 32 | **43%** |
| 4 | 3 | 7 | 5 | 29% |
| 5 | 29 | 436 | 220 | **50%** |

Savings scale with document length — 50%+ for long documents.

---

## Project Structure

```
MExGen_Replication/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── summary.md                       # Paper replication summary
├── research_gap_explanation.md      # Detailed research gap analysis
├── hybrid_walkthrough.md            # Experiment walkthrough with plots
│
├── src/                             # Core modules
│   ├── __init__.py
│   ├── data_loader.py               # XSUM and SQuAD data loading
│   ├── model_wrapper.py             # DistilBART and Flan-T5 wrappers
│   ├── segmentation.py              # spaCy sentence/phrase segmentation
│   ├── attribution.py               # LOO, C-LIME, L-SHAP implementations
│   ├── hybrid_attribution.py        # ★ Our hybrid LOO→L-SHAP method
│   ├── scalarizers.py               # Log Prob, BERT, BART, SUMM scalarizers
│   ├── perturbation_eval.py         # AUPC and perturbation curves
│   ├── self_explanation.py          # LLM self-explanation baseline
│   ├── pshap_wrapper.py             # Partition SHAP wrapper
│   ├── compute_metrics.py           # Spearman + AUPC computation
│   ├── plot_results.py              # Replication figure generation
│   └── run_experiments.py           # Full experiment pipeline
│
├── run_fast_demo.py                 # Quick demo (3 samples, multi-scalarizer)
├── run_hybrid_experiment.py         # ★ Hybrid vs baselines comparison
│
├── results/
│   ├── raw/                         # Cached data and generated outputs
│   ├── metrics/                     # metrics.json, metrics_with_hybrid.json
│   └── figures/                     # All generated plots
│       ├── figure3_spearman.png         # Replication: Spearman heatmap
│       ├── figure4_pertcurves.png       # Replication: Perturbation curves
│       ├── figure5_explainer_comparison.png
│       ├── table1_aupc_comparison.png   # Replication: AUPC bars
│       ├── table2_selfexplain.png       # Replication: MExGen vs Self-Expl.
│       ├── hybrid_aupc_comparison.png       # ★ Hybrid AUPC comparison
│       ├── hybrid_perturbation_curves.png   # ★ Hybrid perturbation curves
│       ├── hybrid_efficiency.png            # ★ Model calls + efficiency
│       ├── hybrid_spearman.png              # ★ Spearman incl. hybrid
│       └── hybrid_summary_dashboard.png     # ★ 3-panel summary
│
└── notebooks/                       # Jupyter notebooks (exploratory)
```

Files marked with ★ are our new contributions.

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- ~4GB disk for model weights (downloaded automatically)
- Apple MPS / NVIDIA CUDA GPU recommended (works on CPU too, just slower)

### Installation

```bash
# Clone the repository
git clone https://github.com/Izhaar-ahmed/MExGen_Replication.git
cd MExGen_Replication

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

---

## Running the Experiments

### 1. Paper Replication (Original Methods)

```bash
# Quick demo — 3 samples, 3 scalarizers (~15 min on GPU)
python run_fast_demo.py

# Full replication — 200 samples, all methods (~3-5 hours on GPU)
python -m src.run_experiments --n_samples 200

# Compute metrics from saved results
python -m src.compute_metrics

# Generate plots
python -m src.plot_results
```

### 2. Hybrid Improvement Experiment

```bash
# Run hybrid vs all baselines — 5 samples, log_prob scalarizer (~8 min on GPU)
python run_hybrid_experiment.py
```

This generates 5 comparison plots in `results/figures/`:
- `hybrid_aupc_comparison.png` — AUPC bar chart
- `hybrid_perturbation_curves.png` — Perturbation curves
- `hybrid_efficiency.png` — Model calls + efficiency ratio
- `hybrid_spearman.png` — Spearman rank correlation
- `hybrid_summary_dashboard.png` — 3-panel summary

### Output

```
=================================================================
EXPERIMENT COMPLETE — SUMMARY
=================================================================

              Method     AUPC    Calls   Efficiency
----------------------------------------------------
                 LOO     4.94       16        31.69
              C-LIME     7.29       44        16.64
              L-SHAP     8.02      183         4.39
   Hybrid (LOO→SHAP)    8.00       92         8.66
----------------------------------------------------

  Hybrid uses 49% fewer model calls than L-SHAP
  Hybrid AUPC is 0.2% lower than L-SHAP
```

---

## References

1. **MExGen Paper**: Guriel et al., "Multi-Level Explanations for Generative Language Models", ACL 2025. [Paper](https://aclanthology.org/2025.acl-long.1553.pdf)
2. **FastSHAP**: Jethani et al., "FastSHAP: Real-Time Shapley Value Estimation", ICLR 2022.
3. **Stochastic Amortization**: Covert et al., "Stochastic Amortization: A Unified Approach to Accelerate Feature and Data Attribution", NeurIPS 2024.
4. **ICX360 Toolkit**: [github.com/IBM/ICX360](https://github.com/IBM/ICX360)
5. **DistilBART**: Shleifer & Rush, "Pre-trained Summarization Distillation", 2020.
6. **XSUM Dataset**: Narayan et al., "Don't Give Me the Details, Just the Summary!", EMNLP 2018.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
