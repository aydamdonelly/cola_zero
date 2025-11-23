"""
Evaluate quantized models on zero-shot tasks using lm-evaluation-harness.

This is an optional evaluation script for benchmarking on standard tasks.
Note: For small models like OPT-125M, scores will be noisy.
This is most useful for larger models (7B+).
"""

import argparse
import json
import os
import random

import numpy as np
import torch

try:
    from lm_eval import evaluator
    HAS_LM_EVAL = True
except ImportError:
    HAS_LM_EVAL = False
    print("Warning: lm-eval not installed. Install with: pip install lm-eval")


def set_all_seeds(seed: int) -> None:
    """Set RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate_zero_shot_tasks(model_path, tasks=None, batch_size=8, seed: int = 42):
    """
    Evaluate quantized model on zero-shot tasks.

    Args:
        model_path: Path to quantized model
        tasks: List of task names (default: standard benchmarks)
        batch_size: Batch size for evaluation

    Returns:
        dict: Task scores
    """
    if not HAS_LM_EVAL:
        print("lm-eval not available, skipping task evaluation")
        return {}

    if tasks is None:
        tasks = [
            "hellaswag",
            "arc_easy",
            "arc_challenge",
            "winogrande",
            "piqa",
            "lambada_openai"
        ]

    print(f"\nEvaluating {model_path} on {len(tasks)} tasks...")
    print(f"Tasks: {tasks}")

    set_all_seeds(seed)

    try:
        results = evaluator.simple_evaluate(
            model="hf",
            model_args=f"pretrained={model_path},dtype=float16,trust_remote_code=True",
            tasks=tasks,
            num_fewshot=0,  # Zero-shot
            batch_size=batch_size,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Extract scores
        scores = {}
        for task in tasks:
            task_results = results['results'][task]

            # Different tasks use different metric names
            if 'acc' in task_results:
                scores[task] = task_results['acc']
            elif 'acc_norm' in task_results:
                scores[task] = task_results['acc_norm']
            else:
                # Fallback: use first available metric
                scores[task] = list(task_results.values())[0]

        return scores

    except Exception as e:
        print(f"Error during evaluation: {e}")
        return {}


def compare_task_performance(baseline_path, quantized_paths, output_file=None, seed: int = 42):
    """
    Compare baseline vs quantized models on tasks.

    Args:
        baseline_path: Path to baseline (FP16) model
        quantized_paths: Dict of {method_name: model_path}
        output_file: Optional JSON file to save results
    """
    if not HAS_LM_EVAL:
        print("lm-eval not available, skipping task evaluation")
        return {}

    print(f"\n{'='*80}")
    print(f"Zero-Shot Task Evaluation")
    print(f"{'='*80}\n")

    # Evaluate baseline
    print("Evaluating baseline (FP16)...")
    baseline_scores = evaluate_zero_shot_tasks(baseline_path, seed=seed)

    if not baseline_scores:
        print("Baseline evaluation failed, aborting")
        return {}

    # Evaluate quantized models
    all_results = {'baseline': baseline_scores}

    for method_name, model_path in quantized_paths.items():
        print(f"\nEvaluating {method_name}...")
        scores = evaluate_zero_shot_tasks(model_path, seed=seed)
        if scores:
            all_results[method_name] = scores

    # Print comparison table
    tasks = list(baseline_scores.keys())

    print(f"\n{'='*80}")
    print(f"Results Comparison")
    print(f"{'='*80}\n")

    # Header
    header = f"{'Task':<20}"
    for method in all_results.keys():
        header += f"{method.capitalize():<12}"
    print(header)
    print("-" * 80)

    # Rows
    for task in tasks:
        row = f"{task:<20}"
        for method in all_results.keys():
            if task in all_results[method]:
                score = all_results[method][task] * 100  # Convert to percentage
                row += f"{score:>10.2f}% "
            else:
                row += f"{'N/A':>11} "
        print(row)

    # Average
    print("-" * 80)
    avg_row = f"{'Average':<20}"
    for method in all_results.keys():
        method_scores = [all_results[method][task] for task in tasks if task in all_results[method]]
        if method_scores:
            avg_score = sum(method_scores) / len(method_scores) * 100
            avg_row += f"{avg_score:>10.2f}% "
        else:
            avg_row += f"{'N/A':>11} "
    print(avg_row)
    print(f"{'='*80}\n")

    # Degradation analysis
    if len(all_results) > 1:
        print(f"Accuracy Degradation (vs Baseline):")
        print("-" * 80)
        for method in quantized_paths.keys():
            if method in all_results:
                degradations = []
                for task in tasks:
                    if task in baseline_scores and task in all_results[method]:
                        baseline_acc = baseline_scores[task] * 100
                        quantized_acc = all_results[method][task] * 100
                        degradation = baseline_acc - quantized_acc
                        degradations.append(degradation)

                if degradations:
                    avg_degradation = sum(degradations) / len(degradations)
                    print(f"{method.capitalize():<15}: {avg_degradation:>6.2f}% average drop")
        print(f"{'='*80}\n")

    # Save results
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {output_file}")

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate models on zero-shot tasks')
    parser.add_argument(
        '--baseline',
        type=str,
        help='Baseline model path or HF model name'
    )
    parser.add_argument(
        '--model_paths',
        type=str,
        nargs='+',
        help='Paths to quantized models (format: method_name:path)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./results/metrics/task_scores.json',
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    if not HAS_LM_EVAL:
        print("\nERROR: lm-eval not installed.")
        print("Install with: pip install lm-eval")
        print("\nThis evaluation is optional. You can proceed with perplexity evaluation only.")
        exit(1)

    if args.baseline and args.model_paths:
        # Parse model paths
        model_paths = {}
        for item in args.model_paths:
            parts = item.split(':')
            if len(parts) == 2:
                method_name, path = parts
                model_paths[method_name] = path
            else:
                print(f"Warning: Invalid format '{item}', expected 'method:path'")

        # Run comparison
        compare_task_performance(
            baseline_path=args.baseline,
            quantized_paths=model_paths,
            output_file=args.output,
            seed=args.seed
        )
    else:
        # Default: compare OPT-125M models
        print("No paths specified, using default OPT-125M models...")
        print("Note: For OPT-125M, zero-shot scores will be noisy.")
        print("This evaluation is most useful for larger models (7B+).\n")

        compare_task_performance(
            baseline_path='facebook/opt-125m',
            quantized_paths={
                'cola_zero': './results/quantized_models/opt-125m-cola-zero-4bit',
                'random': './results/quantized_models/opt-125m-random-4bit'
            },
            output_file=args.output,
            seed=args.seed
        )
