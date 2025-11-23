# Multi-Seed Experimental Framework

This document describes the multi-seed experimental orchestration added to the COLA-Zero quantization evaluation system.

## Overview

The experimental framework systematically evaluates three calibration data selection methods across multiple models and random seeds to establish statistical significance:

**Methods:**
- `gptq_default`: Sequential (deterministic) sampling - first N chunks
- `random`: Random sampling with fixed seed
- `cola_zero`: COLA-Zero intelligent sampling with fixed seed

**Models:**
- `meta-llama/Llama-2-7b-hf`
- `meta-llama/Llama-2-13b-hf`
- `facebook/opt-6.7b`

**Seeds:** 42, 43, 44, 45, 46 (5 runs per configuration)

**Total experiments:** 3 models × 3 methods × 5 seeds = **45 experiments**

## File Structure

```
cola_zero/
├── cola_zero/
│   └── baselines.py                    [UPDATED]
│       └── DefaultSequentialSampler    [NEW]
│
├── experiments/
│   ├── runner.py                       [NEW]
│   └── aggregate_results.py            [NEW]
│
├── run_full_suite.sh                   [NEW]
│
└── results/
    ├── quantized_models/
    │   └── {model}__{method}__seed{seed}/
    └── metrics/
        ├── raw_runs/
        │   └── {model}__{method}__seed{seed}.json
        └── summary_stats.json
```

## Usage

### Quick Start

```bash
# Run full experimental suite
bash run_full_suite.sh
```

This runs all 45 experiments sequentially and generates statistical summaries.

**Estimated runtime:** 150-180 hours on A100 GPU

### Step-by-Step

If you want more control:

```bash
# 1. Run all experiments
python -m experiments.runner

# 2. Aggregate results
python -m experiments.aggregate_results

# 3. Check results
cat results/metrics/summary_stats.json
```

### Running Subset of Experiments

Edit `experiments/runner.py` at the bottom:

```python
config = {
    "models": ["meta-llama/Llama-2-7b-hf"],  # Run only Llama-2-7B
    "methods": ["random", "cola_zero"],       # Skip gptq_default
    "seeds": [42, 43],                         # Only 2 seeds for testing
    ...
}
```

Then run:
```bash
python -m experiments.runner
```

## Output Format

### Individual Experiment Results

Each experiment produces a JSON file: `results/metrics/raw_runs/{model}__{method}__seed{seed}.json`

**Example:** `llama-2-7b-hf__cola_zero__seed42.json`

```json
{
  "model": "meta-llama/Llama-2-7b-hf",
  "method": "cola_zero",
  "seed": 42,
  "n_calibration_samples": 128,
  "seq_len": 2048,
  "quant_bits": 4,
  "group_size": 128,
  "selection_time_sec": 165.14,
  "quant_time_sec": 755.78,
  "total_time_sec": 920.92,
  "perplexity": 5.70,
  "downstream": {
    "arc_easy": null,
    "hellaswag": null,
    "mmlu": null,
    "winogrande": null,
    "average": null
  }
}
```

### Aggregated Statistics

`results/metrics/summary_stats.json` contains:

```json
{
  "llama-2-7b-hf": {
    "cola_zero": {
      "ppl": {
        "mean": 5.72,
        "std": 0.08,
        "cv_percent": 1.4,
        "min": 5.65,
        "max": 5.82
      },
      "times": {
        "selection_time_sec_mean": 167.3,
        "quant_time_sec_mean": 758.2
      }
    },
    "random": {
      "ppl": {
        "mean": 5.79,
        "std": 0.12,
        "cv_percent": 2.1,
        "min": 5.68,
        "max": 5.94
      },
      "times": {
        "selection_time_sec_mean": 11.8,
        "quant_time_sec_mean": 742.1
      }
    },
    "ttest_cola_zero_vs_random": {
      "t_stat": -1.285,
      "p_value": 0.2431,
      "cohens_d": -0.645
    }
  }
}
```

## Statistical Interpretation

The aggregate results provide:

1. **Descriptive Statistics**
   - Mean PPL: Average across 5 seeds
   - Std: Standard deviation
   - CV%: Coefficient of variation (std/mean × 100)
   - Min/Max: Range across seeds

2. **Inferential Statistics**
   - t-statistic: Magnitude of difference
   - p-value: Statistical significance (p < 0.05 = significant)
   - Cohen's d: Effect size
     - Small: |d| = 0.2
     - Medium: |d| = 0.5
     - Large: |d| = 0.8

3. **Interpretation Guide**

| p-value | Cohen's d | Interpretation |
|---------|-----------|----------------|
| < 0.001 | > 0.8 | Strong evidence, large effect |
| < 0.01  | 0.5-0.8 | Strong evidence, medium-large effect |
| < 0.05  | 0.2-0.5 | Moderate evidence, small-medium effect |
| ≥ 0.05  | any | No significant difference (null result) |

## Thesis Integration

### Section 5.2: Experimental Results

**Use `summary_stats.json` directly:**

```latex
\begin{table}
\caption{Perplexity comparison across calibration methods (mean ± std, N=5)}
\begin{tabular}{lrrr}
\toprule
Model & COLA-Zero & Random & p-value \\
\midrule
Llama-2-7B & 5.72 ± 0.08 & 5.79 ± 0.12 & 0.243 \\
Llama-2-13B & ... & ... & ... \\
OPT-6.7B & ... & ... & ... \\
\bottomrule
\end{tabular}
\end{table}
```

### Reporting Null Results

If p-values are not significant (p ≥ 0.05), report honestly:

> "Multi-seed evaluation (N=5) on Llama-2-7B showed no statistically significant difference between COLA-Zero (PPL=5.72±0.08) and random sampling (PPL=5.79±0.12, p=0.243, d=-0.645). The coefficient of variation (CV) for both methods was low (1.4% and 2.1% respectively), indicating stable quantization quality across different calibration seeds. These results suggest that for 4-bit GPTQ quantization with 128 calibration samples, intelligent sample selection provides minimal benefit over random sampling."

This is **valid science** - negative results are publishable and valuable.

## Configuration Options

### In `experiments/runner.py`

```python
config = {
    # Models to evaluate
    "models": [
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-13b-hf",
        "facebook/opt-6.7b"
    ],

    # Calibration methods
    "methods": ["gptq_default", "random", "cola_zero"],

    # Random seeds for statistical significance
    "seeds": [42, 43, 44, 45, 46],

    # Quantization parameters
    "n_calibration_samples": 128,
    "seq_len": 2048,
    "quant_bits": 4,
    "group_size": 128,

    # Downstream evaluation (optional, only runs on seed=42)
    "do_downstream": False  # Set to True to enable lm-eval integration
}
```

### Adding New Models

Simply add to `config['models']`:

```python
"models": [
    "meta-llama/Llama-2-7b-hf",
    "mistralai/Mistral-7B-v0.1",  # New model
    ...
]
```

### Adding New Methods

1. Implement sampler in `cola_zero/baselines.py` or `cola_zero/sampler.py`
2. Add method name to config
3. Update `build_calibration_examples()` in `experiments/runner.py`:

```python
elif method == "your_new_method":
    sampler = YourNewSampler(tokenizer, random_state=seed)
    examples = sampler.select_samples(documents, n_samples, seq_len)
```

## Troubleshooting

### Out of Memory (OOM)

If you get CUDA OOM errors:

1. **Reduce batch size** in quantization (already set to 1)
2. **Reduce n_calibration_samples** from 128 to 64
3. **Skip larger models** temporarily

### Missing Results

If some experiments fail:

```bash
# Check which experiments completed
ls results/metrics/raw_runs/ | wc -l  # Should be 45 for full suite

# Find missing experiments
python -c "
import os
expected = set()
for model in ['llama-2-7b-hf', 'llama-2-13b-hf', 'opt-6.7b']:
    for method in ['gptq_default', 'random', 'cola_zero']:
        for seed in [42, 43, 44, 45, 46]:
            expected.add(f'{model}__{method}__seed{seed}.json')

existing = set(os.listdir('results/metrics/raw_runs'))
missing = expected - existing
print('\n'.join(sorted(missing)))
"
```

Then re-run missing experiments individually by editing `config` in `runner.py`.

### Aggregation Errors

If `aggregate_results.py` fails:

```bash
# Validate JSON files
python -c "
import json
from pathlib import Path

for f in Path('results/metrics/raw_runs').glob('*.json'):
    try:
        json.load(open(f))
    except:
        print(f'Invalid JSON: {f}')
"
```

## Performance Notes

**Per-experiment timing (A100 GPU):**
- OPT-6.7B: ~15-20 min
- Llama-2-7B: ~20-25 min
- Llama-2-13B: ~40-50 min

**Selection overhead:**
- `gptq_default`: <1s (deterministic)
- `random`: ~10-15s
- `cola_zero`: ~3-4 min (perplexity calculation)

**Full suite total:**
- With all 3 models: 150-180 hours
- Llama-2-7B only: 50-60 hours (recommended for initial testing)

## Citation

When using this framework in publications:

```bibtex
@misc{cola_zero_multiseed,
  title={Multi-Seed Experimental Framework for COLA-Zero Quantization Evaluation},
  author={Your Name},
  year={2025},
  note={Master's Thesis}
}
```

## Support

For issues or questions:
1. Check `logs/full_suite_*.log`
2. Verify CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
3. Check disk space: `df -h`
4. Review individual experiment JSONs in `results/metrics/raw_runs/`
