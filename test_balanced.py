"""
Quick test: Does balanced feature selection work better than random?

This script:
1. Loads a small model (for speed)
2. Quantizes with BALANCED COLA-Zero
3. Quantizes with Random baseline
4. Compares perplexity

Run time: ~10-15 minutes on GPU
"""

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from gptqmodel import GPTQModel, QuantizeConfig
import argparse

from cola_zero.sampler_balanced import COLAZeroBalancedSampler
from cola_zero.baselines import RandomSampler


def load_wikitext2():
    """Load WikiText-2 train split."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split='train')
    documents = [text for text in dataset['text'] if len(text.strip()) > 0]
    print(f"Loaded {len(documents)} documents")
    return documents


def quantize_and_eval(model_name, sampler, sampler_name, seed=42):
    """Quantize model and evaluate perplexity."""
    print(f"\n{'='*80}")
    print(f"Testing: {sampler_name}")
    print(f"{'='*80}\n")

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load calibration data
    print("Loading WikiText-2...")
    documents = load_wikitext2()

    # Select samples
    print(f"\nSelecting samples with {sampler_name}...")
    if isinstance(sampler, str):
        # Instantiate if string
        if sampler == 'balanced':
            sampler = COLAZeroBalancedSampler(tokenizer, device='cuda', random_state=seed)
        elif sampler == 'random':
            sampler = RandomSampler(tokenizer, random_state=seed)

    calibration_data, meta = sampler.select_samples(
        texts=documents,
        n_samples=128,
        seq_len=2048,
        min_length=175
    )

    print(f"Selected {len(calibration_data)} samples")

    # Configure quantization
    quantize_config = QuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        damp_percent=0.01
    )

    # Load model
    print(f"\nLoading model {model_name}...")
    model = GPTQModel.load(model_name, quantize_config=quantize_config)

    # Quantize
    print("\nQuantizing...")
    model.quantize(calibration_dataset=calibration_data, batch_size=1)

    # Evaluate perplexity
    print("\nEvaluating perplexity on WikiText-2 test split...")
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split='test')
    test_text = "\n\n".join([t for t in test_dataset['text'] if t.strip()])

    # Tokenize test set
    encodings = tokenizer(test_text, return_tensors='pt')
    max_length = 2048
    stride = 512

    nlls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i

        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    print(f"\n{'='*80}")
    print(f"RESULT: {sampler_name}")
    print(f"{'='*80}")
    print(f"Perplexity: {ppl:.4f}")
    print(f"{'='*80}\n")

    return {
        'method': sampler_name,
        'perplexity': float(ppl),
        'metadata': meta
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='facebook/opt-125m',
                        help='Model to quantize (default: facebook/opt-125m)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("="*80)
    print("BALANCED FEATURES TEST")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Seed: {args.seed}")
    print("="*80)

    # Test both methods
    results = []

    print("\n\n### TEST 1: BALANCED COLA-Zero ###")
    result_balanced = quantize_and_eval(
        args.model,
        'balanced',
        'COLA-Zero (BALANCED)',
        seed=args.seed
    )
    results.append(result_balanced)

    print("\n\n### TEST 2: Random Baseline ###")
    result_random = quantize_and_eval(
        args.model,
        'random',
        'Random Baseline',
        seed=args.seed
    )
    results.append(result_random)

    # Compare
    print("\n\n")
    print("="*80)
    print("FINAL COMPARISON")
    print("="*80)
    for r in results:
        print(f"{r['method']:30s}  PPL: {r['perplexity']:.4f}")

    diff = result_random['perplexity'] - result_balanced['perplexity']
    print(f"\nDifference: {diff:.4f} (negative = balanced is better)")

    if abs(diff) < 0.05:
        print("\n⚠️  Results are still nearly identical!")
        print("   → Feature balance alone may not be enough")
        print("   → Consider activation-based features")
    elif diff > 0:
        print("\n✅ BALANCED is better!")
        print("   → Feature balance helps!")
    else:
        print("\n❌ Random is better (unexpected)")

    print("="*80)
