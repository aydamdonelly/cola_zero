"""
Installation and smoke test script for COLA-Zero.

This script validates that all dependencies are correctly installed
and runs a quick smoke test with a tiny dataset.
"""

import sys
import os

print("="*80)
print("COLA-Zero Installation Test")
print("="*80)

# Test 1: Check Python version
print("\n[1/10] Checking Python version...")
assert sys.version_info >= (3, 8), f"Python 3.8+ required, found {sys.version}"
print(f"  ✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

# Test 2: Check torch
print("\n[2/10] Checking PyTorch...")
try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  ✓ CUDA version: {torch.version.cuda}")
    else:
        print("  ⚠ CUDA not available (will use CPU)")
except ImportError as e:
    print(f"  ✗ PyTorch not installed: {e}")
    sys.exit(1)

# Test 3: Check transformers
print("\n[3/10] Checking Transformers...")
try:
    import transformers
    print(f"  ✓ Transformers {transformers.__version__}")
except ImportError as e:
    print(f"  ✗ Transformers not installed: {e}")
    sys.exit(1)

# Test 4: Check datasets
print("\n[4/10] Checking Datasets...")
try:
    import datasets
    print(f"  ✓ Datasets {datasets.__version__}")
except ImportError as e:
    print(f"  ✗ Datasets not installed: {e}")
    sys.exit(1)

# Test 5: Check scikit-learn
print("\n[5/10] Checking scikit-learn...")
try:
    import sklearn
    print(f"  ✓ scikit-learn {sklearn.__version__}")
except ImportError as e:
    print(f"  ✗ scikit-learn not installed: {e}")
    sys.exit(1)

# Test 6: Check numpy
print("\n[6/10] Checking NumPy...")
try:
    import numpy as np
    print(f"  ✓ NumPy {np.__version__}")
except ImportError as e:
    print(f"  ✗ NumPy not installed: {e}")
    sys.exit(1)

# Test 7: Check AutoGPTQ
print("\n[7/10] Checking AutoGPTQ...")
try:
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    print(f"  ✓ AutoGPTQ installed")
except ImportError as e:
    print(f"  ✗ AutoGPTQ not installed: {e}")
    print("  Install with: pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/")
    sys.exit(1)

# Test 8: Check COLA-Zero package
print("\n[8/10] Checking COLA-Zero package...")
try:
    from cola_zero import COLAZeroSampler, RandomSampler, StratifiedSampler
    print(f"  ✓ COLA-Zero package imported successfully")
except ImportError as e:
    print(f"  ✗ COLA-Zero package not found: {e}")
    print("  Make sure you're in the cola_zero directory")
    sys.exit(1)

# Test 9: Smoke test - Feature extraction
print("\n[9/10] Running smoke test - Feature extraction...")
try:
    from cola_zero.features import extract_tfidf_features, compute_sequence_lengths
    from transformers import AutoTokenizer

    # Test data
    test_texts = [
        "This is a test document about machine learning.",
        "Another document discussing natural language processing.",
        "A third document with different content about AI."
    ]

    # Test TF-IDF
    tfidf = extract_tfidf_features(test_texts, max_features=100)
    assert tfidf.shape == (3, 100), f"Expected (3, 100), got {tfidf.shape}"
    print(f"  ✓ TF-IDF extraction works")

    # Test sequence length
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    lengths = compute_sequence_lengths(test_texts, tokenizer)
    assert len(lengths) == 3, f"Expected 3 lengths, got {len(lengths)}"
    print(f"  ✓ Sequence length computation works")

except Exception as e:
    print(f"  ✗ Smoke test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 10: Smoke test - Sampler
print("\n[10/10] Running smoke test - Sampler...")
try:
    from cola_zero import COLAZeroSampler, RandomSampler
    from transformers import AutoTokenizer

    # Test data
    test_texts = [
        "Sample document one with some content.",
        "Sample document two with different content.",
        "Sample document three with more varied content.",
        "Sample document four discussing another topic.",
        "Sample document five with unique information."
    ] * 10  # 50 documents

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Test COLA-Zero sampler (without perplexity for speed)
    sampler = COLAZeroSampler(
        tokenizer=tokenizer,
        device='cpu',
        include_perplexity=False  # Skip perplexity for faster test
    )

    samples = sampler.select_samples(
        texts=test_texts,
        n_samples=5,
        seq_len=128,
        min_length=10
    )

    assert len(samples) == 5, f"Expected 5 samples, got {len(samples)}"
    assert 'input_ids' in samples[0], "Sample missing 'input_ids'"
    assert 'attention_mask' in samples[0], "Sample missing 'attention_mask'"
    assert samples[0]['input_ids'].shape[0] == 128, f"Expected shape (128,), got {samples[0]['input_ids'].shape}"

    print(f"  ✓ COLA-Zero sampler works")

    # Test Random sampler
    random_sampler = RandomSampler(tokenizer=tokenizer)
    random_samples = random_sampler.select_samples(
        texts=test_texts,
        n_samples=5,
        seq_len=128
    )

    assert len(random_samples) == 5, f"Expected 5 samples, got {len(random_samples)}"
    print(f"  ✓ Random sampler works")

except Exception as e:
    print(f"  ✗ Smoke test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# All tests passed
print("\n" + "="*80)
print("✓ All tests passed! COLA-Zero is ready to use.")
print("="*80)

print("\nNext steps:")
print("  1. Run a quick test: python experiments/01_quantize_cola_zero.py --model facebook/opt-125m --n_samples 32 --no_perplexity")
print("  2. Run full comparison: python experiments/05_compare_methods.py --model facebook/opt-125m")
print("  3. Run on larger model: python experiments/05_compare_methods.py --model meta-llama/Meta-Llama-3-8B-Instruct")
