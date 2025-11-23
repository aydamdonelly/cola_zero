# COLA-Zero Troubleshooting Guide

This guide covers common issues and their solutions.

## Installation Issues

### Issue 1: `torch.load` Security Vulnerability Error

**Error Message:**
```
ValueError: Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`,
we now require users to upgrade torch to at least v2.6 in order to use the function.
This version restriction does not apply when loading files with safetensors.
```

**Solution:**
✅ **FIXED** - All scripts now use `use_safetensors=True` by default.

If you still encounter this issue:
```bash
# Option 1: Use safetensors (recommended)
# Already implemented in all scripts

# Option 2: Upgrade PyTorch (if compatible with your environment)
pip install torch>=2.6.0
```

### Issue 2: AutoGPTQ Not Found

**Error Message:**
```
ImportError: No module named 'auto_gptq'
```

**Solution:**
```bash
# For CUDA 11.8
pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/

# For CUDA 12.1
pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu121/

# For CPU only
pip install auto-gptq
```

### Issue 3: CUDA Out of Memory (OOM)

**Error Message:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Disable perplexity feature** (fastest fix):
   ```bash
   python experiments/01_quantize_cola_zero.py --model facebook/opt-125m --no_perplexity
   ```

2. **Reduce batch size** in `cola_zero/features.py`:
   ```python
   # Line 44: Change batch_size from 16 to 4 or 8
   def estimate_perplexity(
       texts: List[str],
       batch_size: int = 4,  # Reduce this
       max_length: int = 256,
       device: str = 'cuda'
   ):
   ```

3. **Use CPU for perplexity** (slower but no OOM):
   ```python
   # In features.py estimate_perplexity call, use device='cpu'
   perplexity_scores = estimate_perplexity(texts, device='cpu')
   ```

4. **Reduce number of calibration samples**:
   ```bash
   python experiments/01_quantize_cola_zero.py --n_samples 64
   ```

### Issue 4: K-Means Takes Too Long

**Symptom:** Clustering step hangs or takes >5 minutes

**Solutions:**

1. **Reduce max iterations** in `cola_zero/sampler.py` (line 71):
   ```python
   kmeans = KMeans(
       n_clusters=n_samples,
       random_state=self.random_state,
       n_init=5,      # Reduce from 10
       max_iter=100,  # Reduce from 300
       verbose=0
   )
   ```

2. **Use MiniBatchKMeans** for large datasets:
   ```python
   from sklearn.cluster import MiniBatchKMeans

   kmeans = MiniBatchKMeans(
       n_clusters=n_samples,
       random_state=self.random_state,
       batch_size=1000
   )
   ```

## Runtime Issues

### Issue 5: Empty Clusters Warning

**Warning Message:**
```
Warning: Cluster 42 is empty!
After backfill: 128 samples
```

**Explanation:** This is **NORMAL** and **handled automatically**. K-means occasionally creates empty clusters. COLA-Zero detects this and backfills with random samples to ensure you get exactly the requested number of samples.

**No action needed** - this is part of the robust design.

### Issue 6: Perplexity Calculation is Slow

**Symptom:** Perplexity computation takes >10 minutes for 15k documents

**Solutions:**

1. **Fastest: Disable perplexity** (for Iteration 1 / quick testing):
   ```bash
   python experiments/01_quantize_cola_zero.py --no_perplexity
   ```

2. **Increase batch size** (if you have enough GPU memory):
   ```python
   # In features.py, line 44
   batch_size: int = 32,  # Increase from 16
   ```

3. **Use CPU + multiprocessing** (for systems with limited GPU memory):
   ```python
   # Run perplexity on CPU in parallel
   device='cpu'
   # Add multiprocessing in future version
   ```

### Issue 7: Model Loading Fails

**Error Message:**
```
OSError: facebook/opt-125m does not appear to be a valid git repo or folder
```

**Solution:**
```bash
# Check internet connection
ping huggingface.co

# Set HuggingFace cache directory
export HF_HOME=/path/to/cache

# Download model manually first
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('facebook/opt-125m')"
```

### Issue 8: Quantization Crashes

**Error Message:**
```
RuntimeError: Error in quantization
```

**Solutions:**

1. **Disable Triton** (already done by default):
   ```python
   model.quantize(
       examples=calibration_data,
       batch_size=1,
       use_triton=False  # Already set
   )
   ```

2. **Reduce quantization batch size**:
   ```python
   model.quantize(
       examples=calibration_data,
       batch_size=1,  # Keep at 1 for stability
       cache_examples_on_gpu=False  # Disable GPU caching
   )
   ```

3. **Check calibration data format**:
   ```python
   # Each sample must be a dict with 'input_ids' and 'attention_mask'
   assert 'input_ids' in calibration_data[0]
   assert 'attention_mask' in calibration_data[0]
   assert calibration_data[0]['input_ids'].dim() == 1  # Must be 1D
   ```

## Testing Issues

### Issue 9: Test Script Fails

**Solution:**
```bash
# Run installation test first
python test_installation.py

# If passes, run quick test
python quick_test.py

# If still fails, check logs and report issue with full traceback
```

### Issue 10: Import Errors

**Error Message:**
```
ImportError: cannot import name 'COLAZeroSampler'
```

**Solution:**
```bash
# Make sure you're in the project root directory
cd /path/to/cola_zero

# Verify package structure
ls cola_zero/
# Should show: __init__.py, features.py, sampler.py, baselines.py

# Test import
python -c "from cola_zero import COLAZeroSampler; print('OK')"
```

## Performance Optimization

### Faster Testing

```bash
# Minimal test (fastest)
python quick_test.py

# Small model, no perplexity (fast)
python experiments/01_quantize_cola_zero.py \
    --model facebook/opt-125m \
    --n_samples 32 \
    --no_perplexity

# Small model, with perplexity (slower)
python experiments/01_quantize_cola_zero.py \
    --model facebook/opt-125m \
    --n_samples 128

# Full pipeline comparison (slowest)
python experiments/05_compare_methods.py \
    --model facebook/opt-125m
```

### Recommended Settings by Hardware

**Low-end GPU (4-8GB VRAM):**
```bash
python experiments/05_compare_methods.py \
    --model facebook/opt-125m \
    --n_samples 64 \
    --no_perplexity_feature
```

**Mid-range GPU (16-24GB VRAM):**
```bash
python experiments/05_compare_methods.py \
    --model facebook/opt-125m \
    --n_samples 128
```

**High-end GPU (40GB+ VRAM):**
```bash
python experiments/05_compare_methods.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --n_samples 256
```

## Expected Timing

### OPT-125M on A100 (40GB):
- COLA-Zero selection (with perplexity): ~3-4 minutes
- COLA-Zero selection (without perplexity): ~30-60 seconds
- Random selection: ~1-2 seconds
- Quantization: ~5-10 minutes
- **Total: ~8-15 minutes**

### Llama-3-8B on A100 (40GB):
- COLA-Zero selection (with perplexity): ~5-7 minutes
- Random selection: ~1-2 seconds
- Quantization: ~30-60 minutes
- **Total: ~35-70 minutes**

## Debugging Tips

### Enable Verbose Logging

```python
# In sampler.py, enable verbose K-means
kmeans = KMeans(
    n_clusters=n_samples,
    random_state=self.random_state,
    verbose=1  # Change from 0 to 1
)
```

### Check GPU Memory Usage

```bash
# Monitor GPU during execution
watch -n 1 nvidia-smi

# Or in Python
import torch
print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
```

### Save Intermediate Results

```python
# After feature extraction, save features
import numpy as np
np.save('features.npy', features)

# After clustering, save labels
np.save('cluster_labels.npy', cluster_labels)
```

## Getting Help

If you encounter an issue not listed here:

1. Check the error message carefully
2. Run `python test_installation.py` to verify setup
3. Try `python quick_test.py` for a minimal test
4. Check GitHub issues (if repository available)
5. Include full error traceback when reporting issues

## Common Error Patterns

| Error Pattern | Likely Cause | Solution |
|--------------|--------------|----------|
| `CUDA out of memory` | GPU too small | Use `--no_perplexity` or reduce batch size |
| `torch.load` error | Security vulnerability | Use safetensors (already implemented) |
| `Empty cluster` warning | Normal K-means behavior | No action needed (auto-handled) |
| Import errors | Wrong directory | Run from project root |
| Slow perplexity | Large dataset | Use `--no_perplexity` for testing |
| Model download fails | Network/cache issue | Check internet, set HF_HOME |

## Contact

For persistent issues, please provide:
- Full error traceback
- Output of `python test_installation.py`
- System info (GPU, CUDA version, Python version)
- Command you ran
