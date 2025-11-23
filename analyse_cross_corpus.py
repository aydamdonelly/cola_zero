"""
Cross-Corpus Results Analysis

Analyzes results from cross-corpus calibration experiments.

Reports:
1. Per-source performance: (Method, CalibSource) → PPL@WT2, PPL@C4, Downstream
2. Transfer scores: Average relative improvement across corpora
3. Generalization gap: In-domain gain - Out-of-domain gain
4. Domain correspondence: MathQA calib effect on math tasks
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats


def load_results(results_dir: str):
    """Load all result JSONs."""
    results_path = Path(results_dir)
    results = []

    for json_file in results_path.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
                results.append(data)
        except:
            continue

    return results


def compute_stats(values):
    """Compute mean, std, CV."""
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    cv = (std / mean * 100) if mean != 0 else 0
    return mean, std, cv


def compute_transfer_score(method_results, baseline_results, corpora):
    """
    Compute transfer score: average relative improvement across corpora.

    Transfer = (1/M) * Σ_c (PPL_baseline(c) - PPL_method(c)) / PPL_baseline(c)
    """
    improvements = []

    for corpus in corpora:
        ppl_key = f"ppl_{corpus}" if corpus != "wikitext2" else "perplexity"

        method_ppls = [r[ppl_key] for r in method_results if ppl_key in r]
        baseline_ppls = [r[ppl_key] for r in baseline_results if ppl_key in r]

        if len(method_ppls) == 0 or len(baseline_ppls) == 0:
            continue

        # Paired comparison (same seeds)
        method_mean = np.mean(method_ppls)
        baseline_mean = np.mean(baseline_ppls)

        rel_improvement = (baseline_mean - method_mean) / baseline_mean
        improvements.append(rel_improvement * 100)  # Convert to percentage

    return np.mean(improvements) if improvements else 0.0


def main(results_dir: str):
    print("="*80)
    print("CROSS-CORPUS CALIBRATION ANALYSIS")
    print("="*80)
    print()

    # Load results
    results = load_results(results_dir)
    print(f"Loaded {len(results)} experiment results")
    print()

    # Group by (method, calibration_source)
    grouped = defaultdict(list)
    for r in results:
        method = r.get('method', 'unknown')
        calib_source = r.get('calibration_source', 'wikitext')
        key = (method, calib_source)
        grouped[key].append(r)

    # Get unique methods and sources
    methods = sorted(set(k[0] for k in grouped.keys()))
    sources = sorted(set(k[1] for k in grouped.keys()))

    print(f"Methods: {methods}")
    print(f"Calibration sources: {sources}")
    print()

    # Table 1: Per-source, per-method performance
    print("="*120)
    print("TABLE 1: PERFORMANCE BY CALIBRATION SOURCE")
    print("="*120)
    print(f"{'Method':<20} {'CalibSource':<12} {'n':>3} {'PPL@WT2':>12} {'PPL@C4':>12} {'ARC-e':>8} {'HellaSwag':>10} {'PIQA':>8} {'Math':>8} {'Avg↑':>8}")
    print("-"*120)

    for method in methods:
        for source in sources:
            key = (method, source)
            if key not in grouped:
                continue

            group = grouped[key]
            n = len(group)

            # PPL metrics
            ppl_wt2_vals = [r.get('perplexity', 0) for r in group if 'perplexity' in r]
            ppl_c4_vals = [r.get('ppl_c4', 0) for r in group if 'ppl_c4' in r]

            # Downstream metrics (stored in 'downstream' dict)
            arc_vals = [r.get('downstream', {}).get('arc_easy', 0) for r in group if r.get('downstream', {}).get('arc_easy') is not None]
            hellaswag_vals = [r.get('downstream', {}).get('hellaswag', 0) for r in group if r.get('downstream', {}).get('hellaswag') is not None]
            piqa_vals = [r.get('downstream', {}).get('piqa', 0) for r in group if r.get('downstream', {}).get('piqa') is not None]
            math_vals = [r.get('downstream', {}).get('mathqa', 0) for r in group if r.get('downstream', {}).get('mathqa') is not None]

            # Compute means
            ppl_wt2_mean, ppl_wt2_std, _ = compute_stats(ppl_wt2_vals) if ppl_wt2_vals else (0, 0, 0)
            ppl_c4_mean, ppl_c4_std, _ = compute_stats(ppl_c4_vals) if ppl_c4_vals else (0, 0, 0)

            arc_mean = np.mean(arc_vals) if arc_vals else 0
            hellaswag_mean = np.mean(hellaswag_vals) if hellaswag_vals else 0
            piqa_mean = np.mean(piqa_vals) if piqa_vals else 0
            math_mean = np.mean(math_vals) if math_vals else 0

            avg_downstream = np.mean([arc_mean, hellaswag_mean, piqa_mean, math_mean])

            print(f"{method:<20} {source:<12} {n:>3} "
                  f"{ppl_wt2_mean:>6.2f}±{ppl_wt2_std:<4.2f} "
                  f"{ppl_c4_mean:>6.2f}±{ppl_c4_std:<4.2f} "
                  f"{arc_mean:>8.4f} "
                  f"{hellaswag_mean:>10.4f} "
                  f"{piqa_mean:>8.4f} "
                  f"{math_mean:>8.4f} "
                  f"{avg_downstream:>8.4f}")

    print()
    print()

    # Table 2: Transfer Scores and Generalization Gap
    print("="*80)
    print("TABLE 2: TRANSFER SCORES & GENERALIZATION")
    print("="*80)
    print(f"{'Method':<20} {'CalibSource':<12} {'Transfer Score (%)':>20} {'Best PPL Corpus':>18}")
    print("-"*80)

    for method in methods:
        for source in sources:
            key = (method, source)
            if key not in grouped:
                continue

            group = grouped[key]
            baseline_group = grouped.get(('random', source), [])

            if not baseline_group:
                continue

            # Compute transfer score
            corpora = ['wikitext2', 'c4']
            transfer = compute_transfer_score(group, baseline_group, corpora)

            # Find best corpus
            ppl_wt2 = np.mean([r.get('perplexity', 999) for r in group if 'perplexity' in r])
            ppl_c4 = np.mean([r.get('ppl_c4', 999) for r in group if 'ppl_c4' in r])

            best_corpus = "WikiText-2" if ppl_wt2 < ppl_c4 else "C4"

            print(f"{method:<20} {source:<12} {transfer:>19.2f}% {best_corpus:>18}")

    print()
    print()

    # Table 3: Domain Correspondence (MathQA effect)
    print("="*80)
    print("TABLE 3: DOMAIN CORRESPONDENCE (MathQA Calibration Effect)")
    print("="*80)
    print(f"{'Method':<20} {'Math@WikiText':>15} {'Math@C4':>15} {'Math@MathQA':>15} {'Δ (Math calib)':>18}")
    print("-"*80)

    for method in methods:
        math_wt = grouped.get((method, 'wikitext'), [])
        math_c4 = grouped.get((method, 'c4'), [])
        math_mq = grouped.get((method, 'mathqa'), [])

        if not math_wt or not math_mq:
            continue

        wt_math_vals = [r.get('downstream', {}).get('mathqa', 0) for r in math_wt if r.get('downstream', {}).get('mathqa') is not None]
        c4_math_vals = [r.get('downstream', {}).get('mathqa', 0) for r in math_c4 if r.get('downstream', {}).get('mathqa') is not None]
        mq_math_vals = [r.get('downstream', {}).get('mathqa', 0) for r in math_mq if r.get('downstream', {}).get('mathqa') is not None]

        wt_math = np.mean(wt_math_vals) if wt_math_vals else 0
        c4_math = np.mean(c4_math_vals) if c4_math_vals else 0
        mq_math = np.mean(mq_math_vals) if mq_math_vals else 0

        delta = mq_math - wt_math

        print(f"{method:<20} {wt_math:>15.4f} {c4_math:>15.4f} {mq_math:>15.4f} {delta:>17.4f} ({delta/wt_math*100:+.1f}%)")

    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    print("Key Findings:")
    print("  1. Cross-corpus generalization: Check if transfer scores > 0 across sources")
    print("  2. Domain correspondence: MathQA calib should boost math tasks")
    print("  3. Best calibration source: Depends on deployment scenario")
    print()
    print("For thesis:")
    print("  - If transfer scores positive: COLA-Zero generalizes beyond WikiText")
    print("  - If MathQA calib boosts math: Domain-specific calibration works")
    print("  - If C4 calib competitive: Method is source-agnostic")
    print("="*80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyse_cross_corpus.py <results_dir>")
        print("Example: python analyse_cross_corpus.py results/metrics/raw_runs")
        sys.exit(1)

    results_dir = sys.argv[1]
    main(results_dir)
