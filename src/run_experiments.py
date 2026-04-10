"""
Master experiment loop for MExGen replication (XSUM only).
Runs all attribution methods (C-LIME, L-SHAP, LOO, P-SHAP, Self-Explanation)
with all 4 scalarizers on DistilBART/XSUM.
Saves results to results/raw/ as .npz files.
"""

import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.data_loader import load_xsum
from src.model_wrapper import load_distilbart, load_flan_t5_large, select_device
from src.segmentation import segment_text
from src.attribution import explain_loo, explain_clime, explain_lshap
from src.self_explanation import self_explain
from src.pshap_wrapper import explain_pshap
from src.perturbation_eval import drop_top_k_and_score, aupc


RESULTS_DIR = Path(__file__).parent.parent / "results" / "raw"
SCALARIZER_NAMES = ["log_prob", "bert", "bart", "summ"]
EXPLAINER_NAMES = ["clime", "lshap", "loo"]


def make_log_prob_scalarizer(model_wrapper):
    """Create a log_prob scalarizer bound to a specific model."""
    def scalarizer(perturbed_input, original_output):
        return model_wrapper.log_prob(perturbed_input, original_output)
    return scalarizer


def make_text_scalarizer(name: str):
    """Create a text-based scalarizer (bert, bart, summ)."""
    from src.scalarizers import SCALARIZERS
    fn = SCALARIZERS[name]

    def scalarizer(perturbed_input, original_output, model_wrapper=None):
        """For text scalarizers, we first generate output from perturbed input, then compare."""
        if model_wrapper is not None:
            perturbed_output = model_wrapper.generate(perturbed_input)
        else:
            perturbed_output = perturbed_input  # fallback
        return fn(perturbed_output, original_output)
    return scalarizer


def run_mexgen_experiments(n_samples: int = 200):
    """Run all MExGen experiments on XSUM."""
    print("=" * 60)
    print("MExGen Replication — XSUM / DistilBART")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[1/6] Loading XSUM data...")
    samples = load_xsum(n_samples=n_samples)
    print(f"  Loaded {len(samples)} samples")

    # Load models
    print("\n[2/6] Loading DistilBART model...")
    distilbart = load_distilbart()
    print(f"  DistilBART loaded on {distilbart.device}")

    print("\n[3/6] Loading Flan-T5-Large (for self-explanation)...")
    flan_t5 = load_flan_t5_large()
    print(f"  Flan-T5-Large loaded on {flan_t5.device}")

    # -----------------------------------------------------------------------
    # Phase 1: Generate outputs and segment inputs
    # -----------------------------------------------------------------------
    print("\n[4/6] Generating outputs and segmenting inputs...")
    all_units = []
    all_outputs = []
    all_inputs = []

    for sample in tqdm(samples, desc="Generating"):
        input_text = sample["input"]
        gen_output = distilbart.generate(input_text)
        units = segment_text(input_text, mode="sentence_phrase")

        all_inputs.append(input_text)
        all_outputs.append(gen_output)
        all_units.append(units)

    # Save generated outputs
    gen_path = RESULTS_DIR / "xsum_generated_outputs.json"
    with open(gen_path, "w") as f:
        json.dump({"inputs": all_inputs, "outputs": all_outputs, "units": all_units}, f)
    print(f"  Saved to {gen_path}")

    # -----------------------------------------------------------------------
    # Phase 2: Run MExGen attributions (C-LIME, L-SHAP, LOO) × 4 scalarizers
    # -----------------------------------------------------------------------
    print("\n[5/6] Running attribution methods...")

    for scal_name in SCALARIZER_NAMES:
        print(f"\n  --- Scalarizer: {scal_name} ---")

        # Create appropriate scalarizer
        if scal_name == "log_prob":
            scal_fn = make_log_prob_scalarizer(distilbart)
        else:
            base_fn = make_text_scalarizer(scal_name)
            # Wrap to include model for text generation
            def scal_fn(perturbed_input, original_output, _model=distilbart, _base=base_fn):
                return _base(perturbed_input, original_output, model_wrapper=_model)

        for exp_name in EXPLAINER_NAMES:
            print(f"    Explainer: {exp_name}")
            all_scores = []
            all_curves = []

            for i in tqdm(range(len(samples)), desc=f"      {exp_name}/{scal_name}", leave=False):
                units = all_units[i]
                original_output = all_outputs[i]
                original_input = all_inputs[i]

                if len(units) == 0:
                    all_scores.append(np.array([]))
                    all_curves.append([(0.0, 0.0)])
                    continue

                # Run attribution
                if exp_name == "loo":
                    scores = explain_loo(units, None, scal_fn, original_output, original_input)
                elif exp_name == "clime":
                    scores = explain_clime(units, None, scal_fn, original_output, original_input)
                elif exp_name == "lshap":
                    scores = explain_lshap(units, None, scal_fn, original_output, original_input)

                all_scores.append(scores)

                # Compute perturbation curve
                curve = drop_top_k_and_score(
                    units, scores, scal_fn, original_output, original_input
                )
                all_curves.append(curve)

            # Save results
            save_path = RESULTS_DIR / f"xsum_distilbart_{exp_name}_{scal_name}.npz"
            np.savez(
                save_path,
                scores=np.array(all_scores, dtype=object),
                curves=np.array(all_curves, dtype=object),
            )
            print(f"      Saved: {save_path.name}")

    # -----------------------------------------------------------------------
    # Phase 3: Run P-SHAP
    # -----------------------------------------------------------------------
    print("\n  --- P-SHAP ---")
    pshap_scores_all = []
    pshap_curves_all = []
    log_prob_scal = make_log_prob_scalarizer(distilbart)

    for i in tqdm(range(len(samples)), desc="    P-SHAP"):
        units = all_units[i]
        original_output = all_outputs[i]
        original_input = all_inputs[i]

        if len(units) == 0:
            pshap_scores_all.append(np.array([]))
            pshap_curves_all.append([(0.0, 0.0)])
            continue

        scores = explain_pshap(original_input, distilbart, units)
        pshap_scores_all.append(scores)

        curve = drop_top_k_and_score(
            units, scores, log_prob_scal, original_output, original_input
        )
        pshap_curves_all.append(curve)

    save_path = RESULTS_DIR / "xsum_distilbart_pshap_log_prob.npz"
    np.savez(save_path, scores=np.array(pshap_scores_all, dtype=object),
             curves=np.array(pshap_curves_all, dtype=object))
    print(f"    Saved: {save_path.name}")

    # -----------------------------------------------------------------------
    # Phase 4: Run Self-Explanation
    # -----------------------------------------------------------------------
    print("\n[6/6] Running self-explanation...")
    selfexp_scores_all = []
    selfexp_curves_all = []
    success_count = 0

    for i in tqdm(range(len(samples)), desc="  Self-Explain"):
        units = all_units[i]
        original_output = all_outputs[i]
        original_input = all_inputs[i]

        if len(units) == 0:
            selfexp_scores_all.append(np.array([]))
            selfexp_curves_all.append([(0.0, 0.0)])
            continue

        scores, success = self_explain(
            units, original_output, flan_t5, task="summarization"
        )
        selfexp_scores_all.append(scores)
        if success:
            success_count += 1

        curve = drop_top_k_and_score(
            units, scores, log_prob_scal, original_output, original_input
        )
        selfexp_curves_all.append(curve)

    save_path = RESULTS_DIR / "xsum_distilbart_selfexplain_log_prob.npz"
    np.savez(save_path, scores=np.array(selfexp_scores_all, dtype=object),
             curves=np.array(selfexp_curves_all, dtype=object))
    print(f"    Saved: {save_path.name}")
    print(f"    Parse success rate: {success_count}/{len(samples)} ({100*success_count/len(samples):.1f}%)")

    print("\n" + "=" * 60)
    print("All experiments complete!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=200)
    args = parser.parse_args()
    run_mexgen_experiments(n_samples=args.n_samples)
