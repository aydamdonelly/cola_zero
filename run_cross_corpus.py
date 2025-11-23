"""
Cross-Corpus Calibration Experiment (OPT 6.7B focus)

Runs COLA-Zero samplers across multiple calibration sources to validate coverage fixes.
Configuration:
- Model: facebook/opt-6.7b
- Methods: random, cola_zero, cola_zero_vanille
- Calibration size: 128 documents, seq_len 2048
- Seeds: 1–10
- Downstream tasks: arc_easy, hellaswag, piqa, math_qa
"""

import sys
import os

# Allow HuggingFace datasets with custom code (needed for math_qa)
os.environ['HF_DATASETS_TRUST_REMOTE_CODE'] = 'true'

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.runner import run_experiment_suite

if __name__ == "__main__":
    # Calibration sources to test
    calibration_sources = ['wikitext', 'mathqa']

    # Methods to test
    methods = ['random', 'cola_zero', 'cola_zero_vanille']

    # Seeds
    seeds = list(range(1, 11))  # 1-10

    print("="*80)
    print("CROSS-CORPUS CALIBRATION EXPERIMENT")
    print("="*80)
    print(f"Calibration sources: {calibration_sources}")
    print(f"Methods: {methods}")
    print(f"Seeds: {seeds}")
    print(f"Total experiments: {len(calibration_sources)} × {len(methods)} × {len(seeds)} = {len(calibration_sources) * len(methods) * len(seeds)}")
    print(f"\nShared configuration:")
    print(f"  - Model: facebook/opt-6.7b")
    print(f"  - n_samples: 128 (standard COLA token budget)")
    print(f"  - seq_len: 2048")
    print(f"  - Methods: {methods}")
    print("="*80)
    print()

    # Run experiment for each calibration source
    for calib_source in calibration_sources:
        print(f"\n{'='*80}")
        print(f"CALIBRATION SOURCE: {calib_source.upper()}")
        print(f"{'='*80}\n")

        # Adapt parameters for MathQA due to extremely short document lengths
        # Following COLA framework principle: "adaptable based on compression method and source"
        # NOTE: COLA paper uses MathQA (Table 1) but does not document how they handle this
        if calib_source == 'mathqa':
            # MathQA documents average ~133 tokens (much shorter than WikiText/C4)
            n_samples = 128
            seq_len = 2048
            print(f"Using standard COLA parameters for MathQA: n_samples={n_samples}, seq_len={seq_len}")
            print(f"   Total tokens: {n_samples * seq_len:,}")
            print(f"   Runner enforces min_length=50 tokens for MathQA.\n")
        else:
            n_samples = 128
            seq_len = 2048
            print(f"Using standard COLA parameters for {calib_source}: n_samples={n_samples}, seq_len={seq_len}")
            print(f"   Total tokens: {n_samples * seq_len:,}\n")

        config = {
            "models": ["facebook/opt-6.7b"],
            "methods": methods,
            "seeds": seeds,
            "calibration_source": calib_source,  # KEY: different source per run
            "n_calibration_samples": n_samples,  # Adapted for source characteristics
            "seq_len": seq_len,  # Adapted for source characteristics
            "quant_bits": 4,
            "group_size": 128,
            "do_downstream": True,
            "eval_batch_size": 8,
            # Multi-corpus PPL evaluation
            "ppl_corpora": ["wikitext2", "c4"],  # Evaluate on BOTH corpora
            # Updated downstream tasks (replacing WinoGrande with MathQA)
            "downstream_tasks": ["arc_easy", "hellaswag", "piqa", "mathqa"]
        }

        run_experiment_suite(config)

    print()
    print("="*80)
    print("CROSS-CORPUS EXPERIMENT COMPLETE")
    print("="*80)
    print()
    print("Next steps:")
    print("  1. Analyze results:")
    print("     python analyse_cross_corpus.py results/metrics/raw_runs")
    print()
    print("  2. Expected findings:")
    print("     - WikiText calib: Best PPL@WT2, good PPL@C4")
    print("     - C4 calib: Best PPL@C4, good PPL@WT2")
    print("     - MathQA calib: Better math tasks, potentially higher PPL (domain-specific)")
    print("     - COLA-Zero shows robustness across diverse calibration sources")
    print()
    print("  3. Key metrics:")
    print("     - Transfer Score: avg PPL improvement across corpora")
    print("     - Generalization Gap: in-domain gain - out-domain gain")
    print("     - Domain Correspondence: MathQA calib → math task boost")
    print("     - Downstream Performance: arc_easy, hellaswag, piqa, mathqa")
    print()
    print("  4. Note on MathQA calibration:")
    print("     - n_samples=128, seq_len=2048 (matches WikiText/C4 budget)")
    print("     - min_length set to 50 tokens inside runner")
    print("     - Index mapping bug fixed: metadata aligns with original documents")
    print("     - Expect standard token budget without fallback overlap")
    print("="*80)
