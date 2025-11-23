"""
Quantize a model using random calibration data selection (baseline).

This script implements the standard random sampling approach used in
AutoGPTQ examples for comparison with COLA-Zero.
"""

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from gptqmodel import GPTQModel, QuantizeConfig
import sys
import os
import time
import argparse
import warnings
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cola_zero.baselines import RandomSampler


def set_all_seeds(seed: int) -> None:
    """Set RNG seeds across libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def check_torch_version():
    """Check PyTorch version and warn if < 2.6 (due to torch.load CVE)."""
    torch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
    if torch_version < (2, 6):
        warnings.warn(
            f"PyTorch {torch.__version__} detected. PyTorch 2.6+ is recommended "
            f"due to CVE-2025-32434 in torch.load. "
            f"Using PyTorch format may have security implications. "
            f"Consider upgrading: pip install torch>=2.6.0",
            UserWarning
        )
        return False
    return True


def load_wikitext2(split='train'):
    """Load WikiText-2 as individual documents."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    documents = [text for text in dataset['text'] if len(text.strip()) > 0]
    print(f"Loaded {len(documents)} documents from WikiText-2 {split} split")
    return documents


def quantize_with_random(
    model_name: str,
    output_dir: str,
    n_calibration_samples: int = 128,
    seq_len: int = 2048,
    bits: int = 4,
    group_size: int = 128,
    device: str = 'cuda',
    seed: int = 42
):
    """Quantize model using random calibration sample selection."""

    print(f"\n{'='*80}")
    print(f"Random Baseline Quantization Pipeline")
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    print(f"Output: {output_dir}")
    print(f"Calibration samples: {n_calibration_samples}")
    print(f"Sequence length: {seq_len}")
    print(f"Quantization: {bits}-bit, group_size={group_size}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"{'='*80}\n")

    set_all_seeds(seed)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Tokenizer loaded: {tokenizer.__class__.__name__}")

    # Load dataset
    print("\nLoading WikiText-2 dataset...")
    documents = load_wikitext2(split='train')

    # Random sampling
    print("\nRunning random sample selection...")
    start_time = time.time()

    sampler = RandomSampler(tokenizer=tokenizer, random_state=seed)
    calibration_data, meta = sampler.select_samples(
        texts=documents,
        n_samples=n_calibration_samples,
        seq_len=seq_len
    )

    selection_time = time.time() - start_time
    print(f"Selection completed in {selection_time:.2f} seconds")

    # Configure quantization
    print("\nConfiguring quantization...")
    quantize_config = QuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=True,
        damp_percent=0.01
    )
    print(f"  Config: {quantize_config}")

    # Load model
    print(f"\nLoading model {model_name}...")
    model = GPTQModel.load(
        model_name,
        quantize_config=quantize_config
    )
    print("  ✓ Model loaded")

    print("\nStarting quantization...")
    quant_start = time.time()

    model.quantize(
        calibration_dataset=calibration_data,
        batch_size=1
    )

    quant_time = time.time() - quant_start
    print(f"\nQuantization completed in {quant_time:.2f} seconds")

    # Save
    print(f"\nSaving quantized model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model.save(output_dir)
    print("  Model saved")

    print(f"\n{'='*80}")
    print(f"Random Baseline Complete!")
    print(f"{'='*80}")
    print(f"Selection time:     {selection_time:>10.2f}s")
    print(f"Quantization time:  {quant_time:>10.2f}s")
    print(f"Total time:         {selection_time + quant_time:>10.2f}s")
    if meta:
        print(f"Token starts logged: {len(meta.get('start_positions', []))}")
    print(f"Output directory:   {output_dir}")
    print(f"{'='*80}\n")

    return {
        'selection_time': selection_time,
        'quantization_time': quant_time,
        'total_time': selection_time + quant_time,
        'calibration_meta': meta,
        'seed': seed
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quantize model with random baseline')
    parser.add_argument(
        '--model',
        type=str,
        default='facebook/opt-125m',
        help='Model name (e.g., facebook/opt-125m, meta-llama/Meta-Llama-3-8B-Instruct)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (default: ./results/quantized_models/{model_name}-random-4bit)'
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

    args = parser.parse_args()

    # Default output directory
    if args.output is None:
        model_short_name = args.model.split('/')[-1]
        args.output = f'./results/quantized_models/{model_short_name}-random-{args.bits}bit'

    # Run quantization
    quantize_with_random(
        model_name=args.model,
        output_dir=args.output,
        n_calibration_samples=args.n_samples,
        seq_len=args.seq_len,
        bits=args.bits,
        group_size=128,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=args.seed
    )
