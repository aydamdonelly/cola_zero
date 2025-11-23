"""
Quick test script for COLA-Zero with minimal resources.

This runs a fast end-to-end test with:
- Small model (OPT-125M)
- Few samples (32 instead of 128)
- No perplexity feature (faster)
- Shorter sequences (512 instead of 2048)
"""

import torch
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import load_dataset
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from cola_zero.sampler import COLAZeroSampler
from cola_zero.baselines import RandomSampler
import time

print("="*80)
print("COLA-Zero Quick Test")
print("="*80)
print("Running minimal test with:")
print("  - Model: facebook/opt-125m")
print("  - Samples: 32")
print("  - Sequence length: 512")
print("  - Perplexity feature: DISABLED (faster)")
print("="*80)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nDevice: {device}")

# Configuration
MODEL_NAME = 'facebook/opt-125m'
N_SAMPLES = 32
SEQ_LEN = 512
BITS = 4

# Step 1: Load data
print("\n[1/6] Loading WikiText-2 dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
documents = [text for text in dataset['text'] if len(text.strip()) > 50]
print(f"  Loaded {len(documents)} documents")

# Step 2: Load tokenizer
print("\n[2/6] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"  Loaded: {tokenizer.__class__.__name__}")

# Step 3: Test COLA-Zero selection
print("\n[3/6] Testing COLA-Zero sampler (without perplexity)...")
start_time = time.time()

cola_sampler = COLAZeroSampler(
    tokenizer=tokenizer,
    device=device,
    include_perplexity=False  # Skip for speed
)

cola_samples = cola_sampler.select_samples(
    texts=documents,
    n_samples=N_SAMPLES,
    seq_len=SEQ_LEN,
    min_length=50
)

cola_time = time.time() - start_time
print(f"  ✓ Selected {len(cola_samples)} samples in {cola_time:.2f}s")

# Step 4: Test Random selection
print("\n[4/6] Testing Random sampler...")
start_time = time.time()

random_sampler = RandomSampler(tokenizer=tokenizer)
random_samples = random_sampler.select_samples(
    texts=documents,
    n_samples=N_SAMPLES,
    seq_len=SEQ_LEN
)

random_time = time.time() - start_time
print(f"  ✓ Selected {len(random_samples)} samples in {random_time:.2f}s")

# Step 5: Test quantization configuration
print("\n[5/6] Testing quantization configuration...")
try:
    quantize_config = BaseQuantizeConfig(
        bits=BITS,
        group_size=128,
        desc_act=False,
        sym=True,
        true_sequential=True
    )
    print(f"  ✓ Config created: {BITS}-bit quantization")

    # Try loading model (this validates AutoGPTQ works)
    print(f"  Loading model {MODEL_NAME}...")
    model = AutoGPTQForCausalLM.from_pretrained(
        MODEL_NAME,
        quantize_config=quantize_config,
        device_map='auto',
        use_safetensors=True
    )
    print(f"  ✓ Model loaded successfully")

    # Clean up to save memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: Summary
print("\n[6/6] Test Summary")
print("="*80)
print(f"✓ COLA-Zero selection:  {cola_time:.2f}s for {N_SAMPLES} samples")
print(f"✓ Random selection:     {random_time:.2f}s for {N_SAMPLES} samples")
print(f"✓ Model loading:        Works with safetensors")
print(f"✓ Sample shapes:        {cola_samples[0]['input_ids'].shape}")
print("="*80)

print("\n✓ Quick test PASSED! All components working correctly.")
print("\nTo run full experiment:")
print(f"  python experiments/05_compare_methods.py --model {MODEL_NAME}")
print("\nTo run with perplexity feature:")
print(f"  python experiments/01_quantize_cola_zero.py --model {MODEL_NAME} --n_samples 128")
