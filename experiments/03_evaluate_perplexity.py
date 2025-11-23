"""
Evaluate perplexity of quantized models on WikiText-2 test set.

This script evaluates model quality using perplexity (PPL) as the primary metric.
Lower perplexity indicates better language modeling performance.
"""

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer
from gptqmodel import GPTQModel
import numpy as np
import argparse
import json
import os
import random


def set_all_seeds(seed: int) -> None:
    """Set RNG seeds across random, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate_perplexity(model_path, device='cuda', seqlen=2048, seed: int = 42):
    """
    Evaluate perplexity on WikiText-2 test set.

    Args:
        model_path: Path to quantized model
        device: 'cuda' or 'cpu'
        seqlen: Sequence length for evaluation

    Returns:
        float: Perplexity score
    """
    print(f"\nEvaluating perplexity for: {model_path}")

    set_all_seeds(seed)

    # Load model and tokenizer
    print("Loading model...")
    model = GPTQModel.load(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load test data
    print("Loading WikiText-2 test set...")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    # Get model's actual device
    model_device = next(model.parameters()).device
    print(f"  Model device: {model_device}")
    testenc = testenc.input_ids.to(model_device)

    # Calculate perplexity
    print("Computing perplexity...")
    nsamples = testenc.numel() // seqlen
    print(f"  Number of samples: {nsamples}")

    model.eval()
    nlls = []

    with torch.no_grad():
        for i in range(nsamples):
            batch = testenc[:, (i * seqlen):((i + 1) * seqlen)]

            attn = torch.ones_like(batch, dtype=torch.long)
            outputs = model(batch, attention_mask=attn)
            logits = outputs.logits

            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()

            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            neg_log_likelihood = loss.float() * (seqlen - 1)
            nlls.append(neg_log_likelihood)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{nsamples} samples")

    ppl = torch.exp(torch.stack(nlls).sum() / ((seqlen - 1) * nsamples))
    eval_tokens = int(nsamples * seqlen)

    print(f"Perplexity: {ppl.item():.2f}")
    print(f"Evaluated tokens: {eval_tokens}")
    return ppl.item()


def evaluate_stability(model_base_path, method_name, n_runs=5, device='cuda'):
    """
    Evaluate stability across multiple runs (Coefficient of Variation).

    NOTE: This requires quantizing the model multiple times with different seeds.
    For simplicity, this assumes you've already done that.

    Args:
        model_base_path: Base path for models
        method_name: Method name (e.g., 'cola_zero', 'random')
        n_runs: Number of runs to evaluate
        device: 'cuda' or 'cpu'

    Returns:
        dict: Stability statistics
    """
    perplexities = []

    for run in range(n_runs):
        model_path = f"{model_base_path}_{method_name}_run{run}"
        if not os.path.exists(model_path):
            print(f"Warning: {model_path} not found, skipping")
            continue

        ppl = evaluate_perplexity(model_path, device=device)
        perplexities.append(ppl)

    if len(perplexities) == 0:
        print(f"No models found for stability analysis")
        return None

    mean_ppl = np.mean(perplexities)
    std_ppl = np.std(perplexities)
    cv = (std_ppl / mean_ppl) * 100  # Coefficient of Variation

    print(f"\n{'='*60}")
    print(f"Stability Analysis: {method_name}")
    print(f"{'='*60}")
    print(f"Mean Perplexity: {mean_ppl:.2f} ± {std_ppl:.2f}")
    print(f"Coefficient of Variation: {cv:.2f}%")
    print(f"Individual runs: {perplexities}")
    print(f"{'='*60}\n")

    return {
        'mean': mean_ppl,
        'std': std_ppl,
        'cv': cv,
        'all_runs': perplexities
    }


def compare_methods(model_paths, device='cuda', output_file=None, seed: int = 42):
    """
    Compare perplexity across multiple methods.

    Args:
        model_paths: Dict of {method_name: model_path}
        device: 'cuda' or 'cpu'
        output_file: Optional JSON file to save results

    Returns:
        dict: Comparison results
    """
    print(f"\n{'='*80}")
    print(f"Perplexity Comparison")
    print(f"{'='*80}\n")

    results = {}

    for method_name, model_path in model_paths.items():
        print(f"\nEvaluating {method_name}...")
        ppl = evaluate_perplexity(model_path, device=device, seed=seed)
        results[method_name] = ppl

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"Results")
    print(f"{'='*80}")

    methods = list(results.keys())
    for method in methods:
        print(f"{method:.<30} {results[method]:>10.2f} PPL")

    # Calculate improvements
    if 'random' in results:
        baseline_ppl = results['random']
        print(f"\n{'='*80}")
        print(f"Improvement over Random Baseline")
        print(f"{'='*80}")

        for method in methods:
            if method != 'random':
                improvement = baseline_ppl - results[method]
                pct_improvement = (improvement / baseline_ppl) * 100
                sign = "↓" if improvement > 0 else "↑"
                print(f"{method:.<30} {sign} {abs(improvement):>6.2f} ({pct_improvement:>6.2f}%)")

    print(f"{'='*80}\n")

    # Save results
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model perplexity')
    parser.add_argument(
        '--model_paths',
        type=str,
        nargs='+',
        help='Paths to models (format: method_name:path)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./results/metrics/perplexity.json',
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device (cuda or cpu)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    if args.model_paths:
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
        compare_methods(
            model_paths=model_paths,
            device=args.device,
            output_file=args.output,
            seed=args.seed
        )
    else:
        # Default: compare OPT-125M models
        print("No model paths specified, using default OPT-125M models...")
        model_paths = {
            'cola_zero': './results/quantized_models/opt-125m-cola-zero-4bit',
            'random': './results/quantized_models/opt-125m-random-4bit'
        }

        compare_methods(
            model_paths=model_paths,
            device=args.device,
            output_file=args.output,
            seed=args.seed
        )
