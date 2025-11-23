"""
Debug C4 Calibration Issue

Investigates why COLA-Zero performs catastrophically bad with C4 calibration.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer
from experiments.calibration_sources import get_calibration_source
from cola_zero.sampler_balanced import COLAZeroBalancedSampler
from cola_zero.baselines import RandomSampler
import numpy as np

def analyze_samples(samples, label="Samples"):
    """Analyze calibration samples for quality issues."""
    print(f"\n{'='*80}")
    print(f"ANALYSIS: {label}")
    print(f"{'='*80}")

    texts = [s['text'] for s in samples]

    # Basic stats
    print(f"Number of samples: {len(texts)}")
    print(f"Average length: {np.mean([len(t) for t in texts]):.0f} chars")
    print(f"Min length: {min(len(t) for t in texts)} chars")
    print(f"Max length: {max(len(t) for t in texts)} chars")

    # Check for duplicates
    unique_texts = set(texts)
    print(f"Unique samples: {len(unique_texts)} / {len(texts)}")
    if len(unique_texts) < len(texts):
        print(f"⚠️  WARNING: {len(texts) - len(unique_texts)} duplicate samples!")

    # Check for empty/very short
    very_short = [t for t in texts if len(t) < 100]
    if very_short:
        print(f"⚠️  WARNING: {len(very_short)} samples < 100 chars")

    # Check for common junk patterns
    junk_patterns = [
        'click here', 'subscribe', 'cookie', 'privacy policy',
        '©', 'all rights reserved', 'terms of service',
        'advertisement', '|||', '###'
    ]

    junk_count = 0
    for text in texts:
        text_lower = text.lower()
        if any(pattern in text_lower for pattern in junk_patterns):
            junk_count += 1

    if junk_count > 0:
        print(f"⚠️  WARNING: {junk_count} samples contain junk patterns (ads/boilerplate)")

    # Show first 3 samples (truncated)
    print(f"\n--- First 3 samples (first 200 chars each) ---")
    for i, text in enumerate(texts[:3]):
        print(f"\nSample {i+1}:")
        print(text[:200] + "..." if len(text) > 200 else text)
        print(f"Length: {len(text)} chars")

    # Check token coverage
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    total_tokens = sum(len(tokenizer.encode(t)) for t in texts)
    required_tokens = 128 * 2048 * 1.1
    print(f"\n--- Token Coverage ---")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Required tokens (110%): {required_tokens:,.0f}")
    print(f"Coverage ratio: {total_tokens / required_tokens:.2f}x")
    if total_tokens < required_tokens:
        print(f"⚠️  WARNING: Insufficient token coverage!")


def main():
    print("="*80)
    print("C4 CALIBRATION DEBUG")
    print("="*80)

    # Load tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load C4 calibration data
    print("\n[2/5] Loading C4 calibration data...")
    documents = get_calibration_source('c4', split='train')
    print(f"Loaded {len(documents)} C4 documents")

    # Analyze source documents
    print("\n[3/5] Analyzing source C4 documents...")
    print(f"Total documents: {len(documents)}")
    doc_lengths = [len(d) for d in documents]
    print(f"Average doc length: {np.mean(doc_lengths):.0f} chars")
    print(f"Min doc length: {min(doc_lengths)}")
    print(f"Max doc length: {max(doc_lengths)}")

    # Sample with Random
    print("\n[4/5] Testing Random sampler...")
    random_sampler = RandomSampler(tokenizer, random_state=1)
    random_samples, random_meta = random_sampler.select_samples(
        documents, n_samples=128, seq_len=2048
    )
    analyze_samples(random_samples, "Random Baseline")

    # Sample with COLA-Zero
    print("\n[5/5] Testing COLA-Zero sampler...")
    cola_sampler = COLAZeroBalancedSampler(
        tokenizer=tokenizer,
        device='cuda',
        random_state=1,
        tfidf_dims=100
    )

    try:
        cola_samples, cola_meta = cola_sampler.select_samples(
            documents, n_samples=128, seq_len=2048, min_length=175
        )
        analyze_samples(cola_samples, "COLA-Zero")

        # Compare sample overlap
        print("\n" + "="*80)
        print("COMPARISON: Random vs COLA-Zero")
        print("="*80)

        random_texts = set(s['text'] for s in random_samples)
        cola_texts = set(s['text'] for s in cola_samples)
        overlap = random_texts & cola_texts

        print(f"Overlap: {len(overlap)} / 128 samples are identical")
        print(f"COLA-Zero unique samples: {len(cola_texts - random_texts)}")

        # Check if COLA-Zero metadata reveals issues
        print("\n--- COLA-Zero Metadata ---")
        for key, value in cola_meta.items():
            if key not in ['selected_indices', 'selected_texts']:
                print(f"{key}: {value}")

    except Exception as e:
        print(f"❌ ERROR in COLA-Zero sampling: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("DEBUG COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Check if COLA-Zero selected junk/ads/boilerplate")
    print("2. Check if token coverage is sufficient")
    print("3. Check if samples are duplicates/very short")
    print("4. Inspect actual sample quality (first 3 samples above)")


if __name__ == "__main__":
    main()
