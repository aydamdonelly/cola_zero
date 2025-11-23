"""
Quick test of C4 calibration fix

Tests ONE seed with random + cola_zero to verify fix works
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Allow HuggingFace datasets with custom code (needed for math_qa)
os.environ['HF_DATASETS_TRUST_REMOTE_CODE'] = 'true'

from experiments.runner import run_experiment_suite

if __name__ == "__main__":
    print("="*80)
    print("C4 CALIBRATION FIX TEST (1 seed)")
    print("="*80)

    config = {
        "models": ["meta-llama/Meta-Llama-3-8B-Instruct"],
        "methods": ["random", "cola_zero"],  # Only these 2
        "seeds": [1],  # Only 1 seed
        "calibration_source": "c4",
        "n_calibration_samples": 128,
        "seq_len": 2048,
        "quant_bits": 4,
        "group_size": 128,
        "do_downstream": False,  # Skip downstream for speed
        "ppl_corpora": ["wikitext2", "c4"]
    }

    run_experiment_suite(config)

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print("\nExpected results (if fix works):")
    print("  Random:     WikiText-2 ~16, C4 ~17")
    print("  COLA-Zero:  WikiText-2 ~10-13, C4 ~12-15")
    print("\nIf COLA-Zero is still >20, the fix didn't work!")
