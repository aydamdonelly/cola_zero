"""
Re-run seeds 1 & 7 with fixed feature weights and coverage guard.

This script only runs the 2 problematic seeds that had high perplexity.
Expected runtime: ~4 hours
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.runner import run_experiment_suite

if __name__ == "__main__":
    # Configuration for re-running bad seeds
    config = {
        "models": [
            "meta-llama/Meta-Llama-3-8B-Instruct"
        ],
        "methods": ["cola_zero"],  # Only re-run cola_zero (random is already fine)
        "seeds": [1, 7],  # The two problematic seeds
        "n_calibration_samples": 128,
        "seq_len": 2048,
        "quant_bits": 4,
        "group_size": 128,
        "do_downstream": True,
        "eval_batch_size": 8
    }

    print("="*80)
    print("RE-RUNNING BAD SEEDS WITH FIXES")
    print("="*80)
    print("Seeds: 1, 7")
    print("Fixes applied:")
    print("  1. ✅ Corrected feature weights (sqrt-dim rule)")
    print("  2. ✅ Coverage guard (enforces ≥110% before tokenization)")
    print("  3. ✅ K-Means n_init=10 with k-means++")
    print("="*80)
    print()

    # Run
    run_experiment_suite(config)

    print()
    print("="*80)
    print("RE-RUN COMPLETE")
    print("="*80)
    print("Next steps:")
    print("  1. Check results/metrics/raw_runs/ for new seed1 and seed7 results")
    print("  2. Run: python -m experiments.aggregate_results")
    print("  3. Compare with previous results")
    print("="*80)
