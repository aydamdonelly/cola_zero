"""
Proxy Perplexity Pilot Experiment

Quick test (3-5 seeds) to see if proxy perplexity helps.

Runtime: ~6-9 hours for 5 seeds (longer than balanced due to proxy inference)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.runner import run_experiment_suite

if __name__ == "__main__":
    # Pilot configuration
    config = {
        "models": [
            "meta-llama/Meta-Llama-3-8B-Instruct"
        ],
        "methods": ["cola_zero_proxy"],  # Only proxy for pilot
        "seeds": list(range(1, 11)),  # 10 seeds: 1-10
        "n_calibration_samples": 128,
        "seq_len": 2048,
        "quant_bits": 4,
        "group_size": 128,
        "do_downstream": True,
        "eval_batch_size": 8,
        # Proxy-specific settings
        "proxy_model": "gpt2",  # GPT-2 (124M) - FASTEST & excellent quality!
        # Alternatives:
        # "gpt2-medium"  # 355M - good balance
        # "gpt2-large"   # 774M - better quality
        # "facebook/opt-350m"  # 350M
        # "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # 1.1B
        "proxy_max_length": 512  # Truncates long docs to 512 tokens for PPL computation (doesn't filter them out!)
    }

    print("="*80)
    print("COLA-ZERO PROXY EVALUATION")
    print("="*80)
    print("Seeds: 1-10")
    print(f"Proxy model: {config['proxy_model']}")
    print("Expected runtime: ~10-14 hours (GPT-2 inference adds ~15-25 min per seed)")
    print("="*80)
    print()

    # Run pilot
    run_experiment_suite(config)

    print()
    print("="*80)
    print("PROXY EVALUATION COMPLETE")
    print("="*80)
    print("Compare results:")
    print("  python analyse.py results/metrics/raw_runs")
    print()
    print("Expected outcome (n=10 seeds):")
    print("  - PPL: Similar/better than cola_zero (~9.29)")
    print("  - Downstream: Closer to random (~0.714) than vanille")
    print("  - Stability: CV < 2% (better than vanille's 2.92%)")
    print()
    print("This tests: TF-IDF + Length + Diversity + Proxy Perplexity (GPT-2)")
    print("="*80)
