"""
Quick pipeline validation test.

Runs a single experiment end-to-end to catch bugs before full suite.

Usage:
    python test_pipeline.py

Expected runtime: ~10 minutes on GPU
"""

import os
import sys
import json
import time
from pathlib import Path

# Add parent to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.runner import run_single_experiment


def test_pipeline():
    """Run single experiment to validate entire pipeline."""

    print("="*80)
    print("PIPELINE VALIDATION TEST")
    print("="*80)
    print("This runs 1 model × 1 method × 1 seed to catch bugs")
    print("Expected runtime: ~10 minutes")
    print("="*80)
    print()

    # Minimal test config
    config = {
        "models": ["facebook/opt-125m"],  # Smallest model, fastest
        "methods": ["random"],             # Fastest method
        "seeds": [42],                     # Single seed
        "n_calibration_samples": 32,      # Reduced for speed
        "seq_len": 2048,
        "quant_bits": 4,
        "group_size": 128,
        "do_downstream": False
    }

    print("[TEST] Configuration:")
    print(f"  Model: {config['models'][0]}")
    print(f"  Method: {config['methods'][0]}")
    print(f"  Seed: {config['seeds'][0]}")
    print(f"  Samples: {config['n_calibration_samples']}")
    print()

    # Run single experiment
    print("[TEST] Starting experiment...")
    start_time = time.time()

    try:
        result = run_single_experiment(
            model_name=config['models'][0],
            method=config['methods'][0],
            seed=config['seeds'][0],
            config=config
        )

        elapsed = time.time() - start_time

        print()
        print("="*80)
        print("[TEST] ✓ EXPERIMENT COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Runtime: {elapsed/60:.1f} minutes")
        print()
        print("Results:")
        print(f"  PPL: {result['perplexity']:.2f}")
        print(f"  Selection time: {result['selection_time_sec']:.1f}s")
        print(f"  Quant time: {result['quant_time_sec']:.1f}s")
        print()

        # Validate JSON output
        model_clean = config['models'][0].replace("/", "-")
        json_path = f"./results/metrics/raw_runs/{model_clean}__random__seed42.json"

        print("[TEST] Validating JSON output...")
        with open(json_path, 'r') as f:
            saved_result = json.load(f)

        required_keys = [
            'model', 'method', 'seed', 'perplexity',
            'selection_time_sec', 'quant_time_sec', 'total_time_sec',
            'n_calibration_samples', 'seq_len', 'quant_bits', 'group_size',
            'downstream'
        ]

        missing_keys = [k for k in required_keys if k not in saved_result]
        if missing_keys:
            print(f"[TEST] ✗ Missing keys in JSON: {missing_keys}")
            return False

        print(f"[TEST] ✓ JSON valid: {json_path}")
        print()

        # Test aggregation
        print("[TEST] Testing aggregation...")
        from experiments.aggregate_results import load_all_results, group_results, aggregate_and_analyze

        results = load_all_results()
        if len(results) == 0:
            print("[TEST] ✗ No results found for aggregation")
            return False

        grouped = group_results(results)
        summary = aggregate_and_analyze(grouped)

        print("[TEST] ✓ Aggregation successful")
        print()

        # Final validation
        print("="*80)
        print("[TEST] ✓✓✓ ALL VALIDATION PASSED ✓✓✓")
        print("="*80)
        print()
        print("Pipeline is bug-free and ready for full suite!")
        print()
        print("Next steps:")
        print("  1. Edit experiments/runner.py config for your needs")
        print("  2. Run: bash run_full_suite.sh")
        print("="*80)

        return True

    except Exception as e:
        print()
        print("="*80)
        print("[TEST] ✗✗✗ VALIDATION FAILED ✗✗✗")
        print("="*80)
        print(f"Error: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print()
        print("Fix this error before running full suite!")
        print("="*80)

        import traceback
        traceback.print_exc()

        return False


if __name__ == "__main__":
    print("called")
    success = test_pipeline()
    sys.exit(0 if success else 1)
