# COLA-Zero Quick Start Guide

Get up and running with COLA-Zero in 5 minutes.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 16GB+ RAM
- Internet connection (for downloading models/datasets)

## Installation (3 steps)

### 1. Install Dependencies

```bash
cd cola_zero
pip install -r requirements.txt
```

### 2. Install AutoGPTQ

```bash
# Check your CUDA version first
nvcc --version

# For CUDA 11.8
pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/

# For CUDA 12.1
pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu121/

# For CPU only
pip install auto-gptq
```

### 3. Verify Installation

```bash
python test_installation.py
```

You should see:
```
✓ All tests passed! COLA-Zero is ready to use.
```

## Quick Test (Recommended First Step)

Before running full experiments, test with minimal resources:

```bash
python quick_test.py
```

This runs a 2-3 minute smoke test and confirms everything works.

## Your First Experiment

### Option A: Fast Test (5-10 minutes)

Test COLA-Zero without perplexity feature for speed:

```bash
python experiments/01_quantize_cola_zero.py \
    --model facebook/opt-125m \
    --n_samples 32 \
    --no_perplexity
```

### Option B: Full Test on Small Model (15-20 minutes)

Run complete COLA-Zero with all features on OPT-125M:

```bash
python experiments/05_compare_methods.py --model facebook/opt-125m
```

This will:
1. Quantize with COLA-Zero (~4 min)
2. Quantize with Random baseline (~10 min)
3. Evaluate perplexity (~2 min)
4. Generate comparison report

Results saved to: `./results/metrics/comparison_report.json`

### Option C: Production Run on Llama-3-8B (1-2 hours)

For thesis/paper results:

```bash
python experiments/05_compare_methods.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct
```

## Understanding the Output

After running, you'll see:

```
================================================================================
FINAL SUMMARY
================================================================================

Timing:
  COLA-Zero selection:    225.30s
  Random selection:          1.50s
  Selection overhead:      223.80s

Perplexity:
  COLA-Zero:    15.23
  Random:       16.87
  ↓ Improvement:  1.64 (9.7%)

Report saved to: ./results/metrics/comparison_report.json
================================================================================
```

**Key Metrics:**
- **Perplexity** (lower is better): How well the quantized model predicts text
- **Improvement**: % better than random baseline
- **Selection overhead**: Extra time for intelligent selection (one-time cost)

## Common Scenarios

### Scenario 1: I have limited GPU memory (8GB)

```bash
python experiments/01_quantize_cola_zero.py \
    --model facebook/opt-125m \
    --n_samples 32 \
    --no_perplexity
```

### Scenario 2: I want fastest possible test

```bash
python quick_test.py
```

### Scenario 3: I need results for my thesis

```bash
# Run on both OPT-125M and Llama-3-8B
python experiments/05_compare_methods.py --model facebook/opt-125m
python experiments/05_compare_methods.py --model meta-llama/Meta-Llama-3-8B-Instruct

# Results will be in ./results/metrics/
```

### Scenario 4: I want to test "Iteration 1" vs "Iteration 2" (DSR)

```bash
# Iteration 1: Without perplexity feature
python experiments/01_quantize_cola_zero.py \
    --model facebook/opt-125m \
    --no_perplexity \
    --output ./results/quantized_models/opt-125m-iter1-4bit

# Iteration 2: With perplexity feature
python experiments/01_quantize_cola_zero.py \
    --model facebook/opt-125m \
    --output ./results/quantized_models/opt-125m-iter2-4bit

# Compare
python experiments/03_evaluate_perplexity.py \
    --model_paths \
        iteration1:./results/quantized_models/opt-125m-iter1-4bit \
        iteration2:./results/quantized_models/opt-125m-iter2-4bit
```

## Troubleshooting

### ❌ Error: `torch.load` security vulnerability

**Solution:** ✅ Already fixed! All scripts use `use_safetensors=True`.

If you still see this, make sure you're using the latest version of the scripts.

### ❌ Error: CUDA out of memory

**Solution:** Use `--no_perplexity` flag:

```bash
python experiments/01_quantize_cola_zero.py \
    --model facebook/opt-125m \
    --no_perplexity
```

### ❌ Error: AutoGPTQ not found

**Solution:** Install AutoGPTQ with correct CUDA version:

```bash
pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/
```

### More Issues?

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed solutions.

## File Structure

After running experiments, your directory will look like:

```
cola_zero/
├── results/
│   ├── quantized_models/
│   │   ├── opt-125m-cola-zero-4bit/     # Quantized with COLA-Zero
│   │   └── opt-125m-random-4bit/        # Quantized with Random
│   └── metrics/
│       ├── comparison_report.json       # Main results
│       ├── perplexity.json              # Perplexity scores
│       └── task_scores.json             # Zero-shot task scores (if run)
```

## Next Steps

1. ✅ Run `python test_installation.py`
2. ✅ Run `python quick_test.py`
3. ✅ Run first experiment on OPT-125M
4. 📊 Analyze results in `./results/metrics/comparison_report.json`
5. 🚀 Run on larger model (Llama-3-8B)
6. 📝 Use results in thesis/paper

## Getting Help

1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Run `python test_installation.py` to diagnose issues
3. Check error messages carefully
4. Report issues with full error traceback

## Expected Performance

| Model | COLA-Zero Selection | Random Selection | Quantization | Total Time |
|-------|---------------------|------------------|--------------|------------|
| OPT-125M | 3-4 min | 1-2 sec | 5-10 min | ~8-15 min |
| Llama-3-8B | 5-7 min | 1-2 sec | 30-60 min | ~35-70 min |

*Times on A100 40GB. Add 50-100% on slower GPUs.*

## Success Checklist

- [ ] `python test_installation.py` passes
- [ ] `python quick_test.py` passes
- [ ] Can run COLA-Zero quantization
- [ ] Can run Random baseline
- [ ] Can compare results
- [ ] Results saved to `./results/`

## Ready to Publish?

For thesis/paper quality results:

1. Run on multiple models (OPT-125M, OPT-1.3B, Llama-3-8B)
2. Run with multiple random seeds (for stability analysis)
3. Compare Iteration 1 (no perplexity) vs Iteration 2 (with perplexity)
4. Include stratified baseline comparison
5. Report all metrics in comparison_report.json

---

**Questions?** Check [README.md](README.md) for full documentation.
