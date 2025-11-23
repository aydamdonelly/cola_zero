# 10-Minute Validation Workflow

## Quick Validation (Run This First!)

```bash
python test_pipeline.py
```

**Expected output:**
```
================================================================================
PIPELINE VALIDATION TEST
================================================================================
This runs 1 model × 1 method × 1 seed to catch bugs
Expected runtime: ~10 minutes
================================================================================

[TEST] Configuration:
  Model: facebook/opt-125m
  Method: random
  Seed: 42
  Samples: 32

[TEST] Starting experiment...

[RUNNER] EXPERIMENT: model=facebook/opt-125m, method=random, seed=42
[RUNNER] Building calibration data: method=random, seed=42

============================================================
Random Baseline Sampling
============================================================
Selecting 32 samples of length 2048...
  Concatenated corpus length: ... characters
  Total tokens in corpus: ...
  Random sampling complete: 32 samples
============================================================

[RUNNER] Calibration data ready: 32 samples, 8.52s
[RUNNER] Quantizing model: facebook/opt-125m
[RUNNER] Output directory: ./results/quantized_models/opt-125m__random__seed42
[RUNNER] Loading model for quantization...
[RUNNER] Starting quantization (this may take 30-60 minutes)...
INFO - Start quantizing layer 1/12
INFO - Quantizing self_attn.k_proj in layer 1/12...
...
INFO - Model packed.
[RUNNER] Quantization complete: 42.18s
[RUNNER] Saving quantized model...
[RUNNER] Evaluating perplexity: ./results/quantized_models/opt-125m__random__seed42
Loading WikiText-2 test set...
  Model device: cuda:0
Computing perplexity...
  Number of samples: 166
  Processed 10/166 samples
  ...
Perplexity: 29.87
[RUNNER] Perplexity: 29.87

[RUNNER] Results saved to: ./results/metrics/raw_runs/opt-125m__random__seed42.json
[RUNNER] PPL=29.87, Selection=8.5s, Quant=42.2s

================================================================================
[TEST] ✓ EXPERIMENT COMPLETED SUCCESSFULLY
================================================================================
Runtime: 8.3 minutes

Results:
  PPL: 29.87
  Selection time: 8.5s
  Quant time: 42.2s

[TEST] Validating JSON output...
[TEST] ✓ JSON valid: ./results/metrics/raw_runs/opt-125m__random__seed42.json

[TEST] Testing aggregation...
[AGG] Loading results from: ./results/metrics/raw_runs
[AGG] Found 1 result files
[AGG] Grouping results by (model, method)
[AGG]   opt-125m:
[AGG]     random: 1 runs

[AGG] Computing statistics and statistical tests

[AGG] Analyzing: opt-125m
[AGG]   random:
[AGG]     PPL: 29.87 ± 0.00 (CV=0.0%)
[AGG]     Selection time: 8.5s
[AGG]     Quant time: 42.2s
[TEST] ✓ Aggregation successful

================================================================================
[TEST] ✓✓✓ ALL VALIDATION PASSED ✓✓✓
================================================================================

Pipeline is bug-free and ready for full suite!

Next steps:
  1. Edit experiments/runner.py config for your needs
  2. Run: bash run_full_suite.sh
================================================================================
```

**If test passes:** Pipeline is bug-free ✓

**If test fails:** Fix the error shown before running full suite ✗

---

## Full Workflow Stages

### Stage 1: Run Experiments

```bash
python -m experiments.runner
```

**Expected output per experiment:**
```
[RUNNER] Progress: 1/45 experiments

================================================================================
[RUNNER] EXPERIMENT: model=meta-llama/Llama-2-7b-hf, method=cola_zero, seed=42
================================================================================

[RUNNER] Building calibration data: method=cola_zero, seed=42

============================================================
COLA-Zero Sample Selection
============================================================
Include perplexity: True

Step 1: Filtering documents (min_length=500 tokens)...
  Valid documents: 15283 / 23767

Step 2: Extracting features from 15283 documents...
Extracting TF-IDF features...
  TF-IDF shape: (15283, 5000)
Computing perplexity scores (using GPT-2 small as proxy)...
  Processed 160/15283 documents
  ...
  Perplexity shape: (15283,)
Computing sequence lengths...
  Lengths shape: (15283,)
Computing vocabulary diversity...
  Diversity shape: (15283,)
  Feature matrix shape: (15283, 5003)

Step 3: Clustering into 128 groups...
  Clustering complete. Inertia: 71633840.03

Step 4: Selecting representative documents from each cluster...
  Selected 384 representative documents (avg 3.0 per cluster)

  Total tokens in selected documents: 450,000
  Target tokens for 128 chunks: 262,144
  Coverage: 171.6%

Step 5: Tokenizing to AutoGPTQ format (seq_len=2048)...
  Tokenization complete. Sample shape: torch.Size([2048])

============================================================
Selection complete: 128 samples ready
============================================================

[RUNNER] Calibration data ready: 128 samples, 165.14s
[RUNNER] Quantizing model: meta-llama/Llama-2-7b-hf
[RUNNER] Loading model for quantization...
[RUNNER] Starting quantization (this may take 30-60 minutes)...
INFO - Start quantizing layer 1/32
...
[RUNNER] Quantization complete: 755.78s
[RUNNER] Evaluating perplexity: ./results/quantized_models/Llama-2-7b-hf__cola_zero__seed42
Perplexity: 5.70
[RUNNER] Results saved to: ./results/metrics/raw_runs/Llama-2-7b-hf__cola_zero__seed42.json
[RUNNER] PPL=5.70, Selection=165.1s, Quant=755.8s

[RUNNER] Progress: 2/45 experiments
...

================================================================================
[RUNNER] EXPERIMENTAL SUITE COMPLETE
[RUNNER] Completed: 45/45 experiments
[RUNNER] Results directory: ./results/metrics/raw_runs/
================================================================================
```

**Expected files after Stage 1:**
- `results/metrics/raw_runs/Llama-2-7b-hf__cola_zero__seed42.json`
- `results/metrics/raw_runs/Llama-2-7b-hf__cola_zero__seed43.json`
- ... (45 files total)

---

### Stage 2: Aggregate Results

```bash
python -m experiments.aggregate_results
```

**Expected output:**
```
================================================================================
[AGG] RESULTS AGGREGATION AND ANALYSIS
================================================================================

[AGG] Loading results from: ./results/metrics/raw_runs
[AGG] Found 45 result files
[AGG] Loaded 45 total experiments
[AGG] Grouping results by (model, method)
[AGG]   Llama-2-7b-hf:
[AGG]     cola_zero: 5 runs
[AGG]     random: 5 runs
[AGG]     gptq_default: 5 runs
[AGG]   Llama-2-13b-hf:
[AGG]     cola_zero: 5 runs
[AGG]     random: 5 runs
[AGG]     gptq_default: 5 runs
[AGG]   opt-6.7b:
[AGG]     cola_zero: 5 runs
[AGG]     random: 5 runs
[AGG]     gptq_default: 5 runs

[AGG] Computing statistics and statistical tests

[AGG] Analyzing: Llama-2-7b-hf

[AGG]   cola_zero:
[AGG]     PPL: 5.72 ± 0.08 (CV=1.4%)
[AGG]     Selection time: 167.3s
[AGG]     Quant time: 758.2s

[AGG]   random:
[AGG]     PPL: 5.79 ± 0.12 (CV=2.1%)
[AGG]     Selection time: 11.8s
[AGG]     Quant time: 742.1s

[AGG]   gptq_default:
[AGG]     PPL: 5.81 ± 0.10 (CV=1.7%)
[AGG]     Selection time: 0.8s
[AGG]     Quant time: 740.5s

[AGG]   T-test (COLA-Zero vs Random):
[AGG]     t=-1.285, p=0.2431
[AGG]     Cohen's d=-0.645
[AGG]     ✗ Not significant (p >= 0.05)

[AGG]   T-test (COLA-Zero vs GPTQ Default):
[AGG]     t=-1.891, p=0.0982
[AGG]     Cohen's d=-0.948
[AGG]     ✗ Not significant (p >= 0.05)

...

================================================================================
[AGG] SUMMARY TABLE
================================================================================

LLAMA-2-7B-HF
--------------------------------------------------------------------------------

  cola_zero:
    PPL:       5.72 ±  0.08  (CV:   1.4%)
    Range:     [ 5.65,  5.82]
    Selection:  167.3s
    Quant:      758.2s

  random:
    PPL:       5.79 ±  0.12  (CV:   2.1%)
    Range:     [ 5.68,  5.94]
    Selection:   11.8s
    Quant:      742.1s

  gptq_default:
    PPL:       5.81 ±  0.10  (CV:   1.7%)
    Range:     [ 5.70,  5.91]
    Selection:    0.8s
    Quant:      740.5s

  Statistical Tests:
    cola zero vs random:
      p=0.2431 ns, d=-0.645
    cola zero vs gptq default:
      p=0.0982 ns, d=-0.948
    random vs gptq default:
      p=0.6782 ns, d=-0.201

...

================================================================================
[AGG] Legend: ns=not significant, *p<0.05, **p<0.01, ***p<0.001
================================================================================

[AGG] Summary statistics saved to: ./results/metrics/summary_stats.json
[AGG] This file can be directly referenced in thesis Section 5.2

================================================================================
[AGG] AGGREGATION COMPLETE
================================================================================
```

**Expected files after Stage 2:**
- `results/metrics/summary_stats.json` ← **Your thesis results**

---

## Quick Checklist

**Before full suite:**
- [ ] Run `python test_pipeline.py` (10 min)
- [ ] See `✓✓✓ ALL VALIDATION PASSED ✓✓✓`
- [ ] Check disk space: `df -h` (need ~100GB)
- [ ] Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

**If validation passes:**
```bash
bash run_full_suite.sh  # 150-180 hours
```

**If validation fails:**
- Read error message
- Fix the bug
- Re-run `python test_pipeline.py`
- Only proceed when validation passes

---

## Error Meanings

| Error | Meaning | Fix |
|-------|---------|-----|
| `ModuleNotFoundError: gptqmodel` | GPTQModel not installed | `pip install gptqmodel==4.0.0` |
| `CUDA out of memory` | GPU memory full | Reduce `n_calibration_samples` in config |
| `FileNotFoundError` | Missing directory | `mkdir -p results/metrics/raw_runs` |
| `ValueError: numpy.dtype size changed` | Numpy version mismatch | `pip install numpy==2.2.6` |
| `RuntimeError: Expected all tensors` | Device mismatch | Already fixed in code |

---

## Time Estimates

| Task | Time |
|------|------|
| Validation test | 10 min |
| Single Llama-2-7B experiment | 23 min |
| Full suite (45 experiments) | 150-180 hours |
| Aggregation | 1 min |

**Total:** ~10 min validation + 150-180 hours full suite
