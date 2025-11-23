"""
Full comparison pipeline for COLA-Zero evaluation.

This script runs the complete evaluation:
1. Quantize with COLA-Zero
2. Quantize with Random baseline
3. Evaluate perplexity
4. (Optional) Evaluate zero-shot tasks
5. Generate comparison report
"""

import torch
import numpy as np
import json
import os
import argparse
import sys
import random

# Import quantization functions using importlib for compatibility with dash-named files
import importlib.util

# Helper function to import module from file path
def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import functions from experiment scripts
exp_dir = os.path.dirname(os.path.abspath(__file__))
quantize_cola_zero_module = import_from_path("quantize_cola_zero", os.path.join(exp_dir, "01_quantize_cola_zero.py"))
quantize_random_module = import_from_path("quantize_random", os.path.join(exp_dir, "02_quantize_random.py"))
evaluate_perplexity_module = import_from_path("evaluate_perplexity", os.path.join(exp_dir, "03_evaluate_perplexity.py"))

quantize_with_cola_zero = quantize_cola_zero_module.quantize_with_cola_zero
quantize_with_random = quantize_random_module.quantize_with_random
evaluate_perplexity = evaluate_perplexity_module.evaluate_perplexity

# Try to import task evaluation (optional)
try:
    evaluate_tasks_module = import_from_path("evaluate_tasks", os.path.join(exp_dir, "04_evaluate_tasks.py"))
    compare_task_performance = evaluate_tasks_module.compare_task_performance
    HAS_TASK_EVAL = True
except Exception:
    HAS_TASK_EVAL = False


def set_all_seeds(seed: int) -> None:
    """Set RNG seeds across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_full_comparison(
    model_name='facebook/opt-125m',
    output_base='./results',
    n_samples=128,
    seq_len=2048,
    bits=4,
    include_perplexity_feature=True,
    skip_task_eval=True,
    seed: int = 42
):
    """
    Run complete comparison pipeline.

    Args:
        model_name: HuggingFace model name
        output_base: Base directory for results
        n_samples: Number of calibration samples
        seq_len: Sequence length
        bits: Quantization bits
        include_perplexity_feature: Whether to include perplexity in COLA-Zero features
        skip_task_eval: Skip zero-shot task evaluation (faster)

    Returns:
        dict: Complete comparison report
    """
    print(f"\n{'='*80}")
    print(f"FULL COMPARISON PIPELINE")
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    print(f"Calibration samples: {n_samples}")
    print(f"Sequence length: {seq_len}")
    print(f"Quantization bits: {bits}")
    print(f"Include perplexity feature: {include_perplexity_feature}")
    print(f"Skip task evaluation: {skip_task_eval}")
    print(f"Seed: {seed}")
    print(f"{'='*80}\n")

    set_all_seeds(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Paths
    model_short_name = model_name.split('/')[-1]
    cola_zero_path = f"{output_base}/quantized_models/{model_short_name}-cola-zero-{bits}bit"
    random_path = f"{output_base}/quantized_models/{model_short_name}-random-{bits}bit"

    # Step 1: Quantize with COLA-Zero
    print("\n" + "="*80)
    print("STEP 1: Quantizing with COLA-Zero")
    print("="*80)
    cola_timing = quantize_with_cola_zero(
        model_name=model_name,
        output_dir=cola_zero_path,
        n_calibration_samples=n_samples,
        seq_len=seq_len,
        bits=bits,
        device=device,
        include_perplexity=include_perplexity_feature,
        seed=seed
    )

    # Step 2: Quantize with Random
    print("\n" + "="*80)
    print("STEP 2: Quantizing with Random Baseline")
    print("="*80)
    random_timing = quantize_with_random(
        model_name=model_name,
        output_dir=random_path,
        n_calibration_samples=n_samples,
        seq_len=seq_len,
        bits=bits,
        device=device,
        seed=seed
    )

    # Step 3: Evaluate Perplexity
    print("\n" + "="*80)
    print("STEP 3: Evaluating Perplexity")
    print("="*80)

    ppl_cola = evaluate_perplexity(cola_zero_path, device=device, seed=seed)
    ppl_random = evaluate_perplexity(random_path, device=device, seed=seed)

    # Step 4: Evaluate Tasks (optional)
    task_results = None
    if not skip_task_eval and HAS_TASK_EVAL:
        print("\n" + "="*80)
        print("STEP 4: Evaluating Zero-Shot Tasks")
        print("="*80)

        task_results = compare_task_performance(
            baseline_path=model_name,
            quantized_paths={
                'cola_zero': cola_zero_path,
                'random': random_path
            },
            output_file=f"{output_base}/metrics/task_scores.json",
            seed=seed
        )
    else:
        if skip_task_eval:
            print("\n" + "="*80)
            print("STEP 4: Skipping Zero-Shot Tasks (--skip_task_eval)")
            print("="*80)
        else:
            print("\n" + "="*80)
            print("STEP 4: Zero-Shot Tasks Not Available (lm-eval not installed)")
            print("="*80)

    # Step 5: Generate Report
    print("\n" + "="*80)
    print("STEP 5: Generating Final Report")
    print("="*80)

    report = {
        'model': model_name,
        'calibration_samples': n_samples,
        'sequence_length': seq_len,
        'quantization_bits': bits,
        'include_perplexity_feature': include_perplexity_feature,
        'seed': seed,
        'timing': {
            'cola_zero': cola_timing,
            'random': random_timing
        },
        'perplexity': {
            'cola_zero': ppl_cola,
            'random': ppl_random,
            'improvement': ppl_random - ppl_cola,
            'improvement_pct': ((ppl_random - ppl_cola) / ppl_random) * 100
        },
        'calibration_meta': {
            'cola_zero': cola_timing.get('calibration_meta'),
            'random': random_timing.get('calibration_meta')
        }
    }

    if task_results:
        report['task_accuracy'] = task_results

    # Save report
    report_path = f"{output_base}/metrics/comparison_report.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    print(f"\nTiming:")
    print(f"  COLA-Zero selection:  {cola_timing['selection_time']:>8.2f}s")
    print(f"  Random selection:     {random_timing['selection_time']:>8.2f}s")
    print(f"  Selection overhead:   {cola_timing['selection_time'] - random_timing['selection_time']:>8.2f}s")

    print(f"\n  COLA-Zero quantization:  {cola_timing['quantization_time']:>8.2f}s")
    print(f"  Random quantization:     {random_timing['quantization_time']:>8.2f}s")

    print(f"\n  COLA-Zero total:  {cola_timing['total_time']:>8.2f}s")
    print(f"  Random total:     {random_timing['total_time']:>8.2f}s")

    print(f"\nPerplexity:")
    print(f"  COLA-Zero:  {ppl_cola:>8.2f}")
    print(f"  Random:     {ppl_random:>8.2f}")

    improvement = ppl_random - ppl_cola
    improvement_pct = (improvement / ppl_random) * 100
    sign = "↓" if improvement > 0 else "↑"
    print(f"  {sign} Improvement: {abs(improvement):>7.2f} ({abs(improvement_pct):>5.1f}%)")

    print(f"\nReport saved to: {report_path}")
    print("="*80 + "\n")

    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run full COLA-Zero comparison pipeline')
    parser.add_argument(
        '--model',
        type=str,
        default='facebook/opt-125m',
        help='Model name (e.g., facebook/opt-125m, meta-llama/Meta-Llama-3-8B-Instruct)'
    )
    parser.add_argument(
        '--output_base',
        type=str,
        default='./results',
        help='Base directory for results'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=128,
        help='Number of calibration samples'
    )
    parser.add_argument(
        '--seq_len',
        type=int,
        default=2048,
        help='Sequence length'
    )
    parser.add_argument(
        '--bits',
        type=int,
        default=4,
        help='Number of bits for quantization'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--no_perplexity_feature',
        action='store_true',
        help='Disable perplexity feature in COLA-Zero (for iteration 1)'
    )
    parser.add_argument(
        '--with_task_eval',
        action='store_true',
        help='Include zero-shot task evaluation (slower, optional)'
    )

    args = parser.parse_args()

    # Run full comparison
    run_full_comparison(
        model_name=args.model,
        output_base=args.output_base,
        n_samples=args.n_samples,
        seq_len=args.seq_len,
        bits=args.bits,
        include_perplexity_feature=not args.no_perplexity_feature,
        skip_task_eval=not args.with_task_eval,
        seed=args.seed
    )
