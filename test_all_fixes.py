"""
Comprehensive Test: Alle Fixes validieren

Tests:
1. C4 Calibration (Random + COLA-Zero, 1 seed)
2. Determinismus (gleicher seed 2×)
3. Doctor-Report Validierung

Erwartete Ergebnisse (nach allen Fixes):
  Random:     WikiText-2 ~16, C4 ~17 (baseline)
  COLA-Zero:  WikiText-2 ~10-13, C4 ~12-15 (✅ FIX ERFOLG!)

Wenn COLA-Zero immer noch >20 PPL → etwas ist noch kaputt!
"""

import sys
import os
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Allow HuggingFace datasets with custom code
os.environ['HF_DATASETS_TRUST_REMOTE_CODE'] = 'true'

from experiments.runner import run_experiment_suite

def test_c4_calibration():
    """Test 1: C4 calibration mit Random + COLA-Zero"""
    print("="*80)
    print("TEST 1: C4 CALIBRATION (1 seed, no downstream)")
    print("="*80)
    print()
    print("Expected after fixes:")
    print("  Random:     WikiText-2 ~16, C4 ~17")
    print("  COLA-Zero:  WikiText-2 ~10-13, C4 ~12-15")
    print()
    print("Previous (broken) results:")
    print("  COLA-Zero:  WikiText-2 29.33, C4 25.68 ❌")
    print("="*80)

    config = {
        "models": ["meta-llama/Meta-Llama-3-8B-Instruct"],
        "methods": ["cola_zero", "random"],  # COLA-Zero first for faster feedback!
        "seeds": [42],  # Single seed for speed
        "calibration_source": "c4",
        "n_calibration_samples": 128,
        "seq_len": 2048,
        "quant_bits": 4,
        "group_size": 128,
        "do_downstream": False,  # Skip for speed
        "ppl_corpora": ["wikitext2", "c4"]
    }

    run_experiment_suite(config)


def test_determinism():
    """Test 2: Determinismus - gleicher seed sollte identische PPL geben"""
    print("\n" + "="*80)
    print("TEST 2: DETERMINISMUS (same seed 2×)")
    print("="*80)
    print()
    print("Running COLA-Zero with seed=99 twice...")
    print("Expected: PPL should match within ±0.05")
    print("="*80)

    config = {
        "models": ["meta-llama/Meta-Llama-3-8B-Instruct"],
        "methods": ["cola_zero"],
        "seeds": [99, 99],  # Same seed twice
        "calibration_source": "c4",
        "n_calibration_samples": 128,
        "seq_len": 2048,
        "quant_bits": 4,
        "group_size": 128,
        "do_downstream": False,
        "ppl_corpora": ["wikitext2"]
    }

    run_experiment_suite(config)


def validate_results():
    """Validate results from JSON files"""
    print("\n" + "="*80)
    print("VALIDATION: Checking Results")
    print("="*80)

    results_dir = Path("./results/metrics/raw_runs")
    if not results_dir.exists():
        print("❌ Results directory not found!")
        return

    # Check Test 1: C4 results
    c4_files = list(results_dir.glob("*__c4__seed42.json"))
    if not c4_files:
        print("⚠️  No C4 seed42 results found!")
        return

    print("\n--- Test 1: C4 Calibration Results ---")
    for json_file in sorted(c4_files):
        with open(json_file, 'r') as f:
            data = json.load(f)

        method = data.get('method', 'unknown')
        ppl_wt2 = data.get('perplexity', -1)
        ppl_c4 = data.get('ppl_c4', -1)

        print(f"\n{method:15} (seed 42)")
        print(f"  WikiText-2: {ppl_wt2:.2f}")
        print(f"  C4:         {ppl_c4:.2f}")

        # Validation checks
        if method == "cola_zero":
            if ppl_wt2 > 20:
                print(f"  ❌ FAIL: COLA-Zero PPL still >20! Fix didn't work.")
            elif ppl_wt2 < 14:
                print(f"  ✅ PASS: COLA-Zero PPL looks good!")
            else:
                print(f"  ⚠️  MARGINAL: PPL is 14-20 (expected <14)")

        # Check doctor report
        if 'calib_doctor' in data.get('calib_meta', {}):
            doctor = data['calib_meta']['calib_doctor']
            pad_ratio = doctor['specials']['frac_pad']
            diversity = doctor['diversity']

            print(f"  Doctor Report:")
            print(f"    Pad ratio: {pad_ratio:.4f} {'✅' if pad_ratio == 0.0 else '❌ FAIL'}")
            print(f"    Diversity: {diversity:.3f} {'✅' if diversity > 0.3 else '⚠️  LOW'}")

    # Check Test 2: Determinism
    det_files = list(results_dir.glob("*__cola_zero__c4__seed99.json"))
    if len(det_files) >= 2:
        print("\n--- Test 2: Determinism Check ---")
        ppls = []
        for json_file in sorted(det_files)[:2]:
            with open(json_file, 'r') as f:
                data = json.load(f)
            ppls.append(data.get('perplexity', -1))

        if len(ppls) == 2:
            diff = abs(ppls[0] - ppls[1])
            print(f"Run 1: PPL = {ppls[0]:.4f}")
            print(f"Run 2: PPL = {ppls[1]:.4f}")
            print(f"Diff:  {diff:.4f}")

            if diff < 0.05:
                print(f"✅ PASS: Deterministic (diff < 0.05)")
            else:
                print(f"❌ FAIL: Non-deterministic (diff = {diff:.4f})")

    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    print("="*80)
    print("COMPREHENSIVE FIX TEST")
    print("="*80)
    print()
    print("This will run:")
    print("  1. C4 calibration test (Random + COLA-Zero, seed 42)")
    print("  2. Determinism test (COLA-Zero, seed 99 × 2)")
    print("  3. Results validation")
    print()
    print("Total runtime: ~4-5 hours")
    print("="*80)
    print()

    # Run tests
    test_c4_calibration()
    test_determinism()

    # Validate
    validate_results()

    print("\n" + "="*80)
    print("ALL TESTS COMPLETE")
    print("="*80)
    print()
    print("Next steps:")
    print("  1. If Test 1 PASS: Run full cross-corpus (python run_cross_corpus.py)")
    print("  2. If Test 1 FAIL: Debug - check logs for Doctor reports")
    print("  3. Check: results/metrics/raw_runs/*.json for detailed results")
    print("="*80)
