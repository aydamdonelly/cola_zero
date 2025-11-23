"""
Quick test: Padding-Fix mit 1 seed validieren

Testet Random + COLA-Zero mit C4 Calibration (1 seed).
Erwartet nach Fix:
  - Random:     WikiText-2 ~16, C4 ~17 (unverändert)
  - COLA-Zero:  WikiText-2 ~10-13, C4 ~12-15 (MASSIV besser als vorher!)

Wenn COLA-Zero immer noch >20 PPL → Fix hat nicht funktioniert!
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Allow HuggingFace datasets with custom code
os.environ['HF_DATASETS_TRUST_REMOTE_CODE'] = 'true'

from experiments.runner import run_experiment_suite

if __name__ == "__main__":
    print("="*80)
    print("C4 PADDING-FIX TEST (1 seed, no downstream)")
    print("="*80)
    print()
    print("Testing:")
    print("  - Random baseline (should be ~16 PPL, unchanged)")
    print("  - COLA-Zero (should be ~10-13 PPL after fix!)")
    print()
    print("="*80)

    config = {
        "models": ["meta-llama/Meta-Llama-3-8B-Instruct"],
        "methods": ["random", "cola_zero"],  # Only these 2
        "seeds": [1],  # Only 1 seed for quick validation
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
    print()
    print("Expected results (if padding fix works):")
    print("  Random:     WikiText-2 ~16, C4 ~17")
    print("  COLA-Zero:  WikiText-2 ~10-13, C4 ~12-15")
    print()
    print("Previous (broken) results:")
    print("  Random:     WikiText-2 16.03, C4 16.71")
    print("  COLA-Zero:  WikiText-2 29.33, C4 25.68 ❌")
    print()
    print("If COLA-Zero is still >20 PPL, check:")
    print("  1. calibration_doctor.py imported correctly?")
    print("  2. HAS_CALIB_DOCTOR = True in runner logs?")
    print("  3. Doctor report shows frac_pad = 0.0?")
    print("="*80)
