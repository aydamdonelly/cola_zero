"""
Aggregate and analyze experimental results across all runs.

This script:
1. Loads all JSON files from results/metrics/raw_runs/
2. Groups by (model, method)
3. Computes statistics: mean, std, CV, min, max
4. Performs t-tests between methods
5. Calculates Cohen's d effect sizes
6. Generates summary_stats.json

Usage:
    python -m experiments.aggregate_results

Output:
    results/metrics/summary_stats.json
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
from scipy import stats


def load_all_results(results_dir: str = "./results/metrics/raw_runs") -> List[Dict]:
    """
    Load all JSON result files.

    Args:
        results_dir: Directory containing raw run JSONs

    Returns:
        List of result dictionaries
    """
    print(f"[AGG] Loading results from: {results_dir}")

    results = []
    json_files = list(Path(results_dir).glob("*.json"))

    print(f"[AGG] Found {len(json_files)} result files")

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            results.append(data)

    return results


def group_results(results: List[Dict]) -> Dict:
    """
    Group results by (model, method).

    Args:
        results: List of result dictionaries

    Returns:
        Nested dict: {model: {method: [results]}}
    """
    print(f"[AGG] Grouping results by (model, method)")

    grouped = {}

    for result in results:
        model = result['model']
        method = result['method']

        if model not in grouped:
            grouped[model] = {}

        if method not in grouped[model]:
            grouped[model][method] = []

        grouped[model][method].append(result)

    # Print summary
    for model in grouped:
        model_clean = model.replace("/", "-")
        print(f"[AGG]   {model_clean}:")
        for method in grouped[model]:
            n_runs = len(grouped[model][method])
            print(f"[AGG]     {method}: {n_runs} runs")

    return grouped


def compute_statistics(values: List[float]) -> Dict:
    """
    Compute descriptive statistics.

    Args:
        values: List of numeric values

    Returns:
        Dict with mean, std, cv, min, max
    """
    values_arr = np.array(values)

    mean_val = float(np.mean(values_arr))
    std_val = float(np.std(values_arr, ddof=1))  # Sample std
    cv_val = (std_val / mean_val * 100) if mean_val != 0 else 0
    min_val = float(np.min(values_arr))
    max_val = float(np.max(values_arr))

    return {
        "mean": mean_val,
        "std": std_val,
        "cv_percent": cv_val,
        "min": min_val,
        "max": max_val
    }


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Calculate Cohen's d effect size.

    Args:
        group1: First group values
        group2: Second group values

    Returns:
        Cohen's d (float)
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std

    return float(d)


def perform_ttest(group1: List[float], group2: List[float]) -> Dict:
    """
    Perform independent t-test.

    Args:
        group1: First group values
        group2: Second group values

    Returns:
        Dict with t_stat, p_value, cohens_d
    """
    if len(group1) < 2 or len(group2) < 2:
        return {
            "t_stat": None,
            "p_value": None,
            "cohens_d": None,
            "note": "Insufficient samples for t-test"
        }

    t_stat, p_value = stats.ttest_ind(group1, group2)
    d = cohens_d(group1, group2)

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": d
    }


def aggregate_and_analyze(grouped: Dict) -> Dict:
    """
    Compute statistics and perform statistical tests.

    Args:
        grouped: Grouped results from group_results()

    Returns:
        Summary statistics dictionary
    """
    print(f"\n[AGG] Computing statistics and statistical tests")

    summary = {}

    for model in grouped:
        model_clean = model.replace("/", "-")
        print(f"\n[AGG] Analyzing: {model_clean}")

        summary[model_clean] = {}

        # Compute statistics for each method
        for method in grouped[model]:
            results_list = grouped[model][method]

            # Extract perplexity values
            ppl_values = [r['perplexity'] for r in results_list]

            # Extract timing values
            selection_times = [r['selection_time_sec'] for r in results_list]
            quant_times = [r['quant_time_sec'] for r in results_list]

            # Compute stats
            ppl_stats = compute_statistics(ppl_values)
            selection_time_mean = float(np.mean(selection_times))
            quant_time_mean = float(np.mean(quant_times))

            summary[model_clean][method] = {
                "ppl": ppl_stats,
                "times": {
                    "selection_time_sec_mean": selection_time_mean,
                    "quant_time_sec_mean": quant_time_mean
                }
            }

            print(f"[AGG]   {method}:")
            print(f"[AGG]     PPL: {ppl_stats['mean']:.2f} ± {ppl_stats['std']:.2f} (CV={ppl_stats['cv_percent']:.1f}%)")
            print(f"[AGG]     Selection time: {selection_time_mean:.1f}s")
            print(f"[AGG]     Quant time: {quant_time_mean:.1f}s")

        # Perform t-tests between methods
        methods = list(grouped[model].keys())

        # cola_zero vs random
        if 'cola_zero' in methods and 'random' in methods:
            cola_ppl = [r['perplexity'] for r in grouped[model]['cola_zero']]
            random_ppl = [r['perplexity'] for r in grouped[model]['random']]

            ttest_result = perform_ttest(cola_ppl, random_ppl)
            summary[model_clean]['ttest_cola_zero_vs_random'] = ttest_result

            if ttest_result['p_value'] is not None:
                print(f"\n[AGG]   T-test (COLA-Zero vs Random):")
                print(f"[AGG]     t={ttest_result['t_stat']:.3f}, p={ttest_result['p_value']:.4f}")
                print(f"[AGG]     Cohen's d={ttest_result['cohens_d']:.3f}")

                # Interpretation
                if ttest_result['p_value'] < 0.05:
                    print(f"[AGG]     ✓ Statistically significant (p < 0.05)")
                else:
                    print(f"[AGG]     ✗ Not significant (p >= 0.05)")

    return summary


def print_summary_table(summary: Dict):
    """
    Print formatted summary table to console.

    Args:
        summary: Summary statistics dictionary
    """
    print(f"\n{'='*80}")
    print(f"[AGG] SUMMARY TABLE")
    print(f"{'='*80}\n")

    for model in summary:
        print(f"\n{model.upper()}")
        print(f"{'-'*80}")

        # Print method statistics
        for key in summary[model]:
            if not key.startswith('ttest_'):
                method = key
                stats_data = summary[model][method]

                ppl = stats_data['ppl']
                times = stats_data['times']

                print(f"\n  {method}:")
                print(f"    PPL:       {ppl['mean']:6.2f} ± {ppl['std']:5.2f}  (CV: {ppl['cv_percent']:5.1f}%)")
                print(f"    Range:     [{ppl['min']:6.2f}, {ppl['max']:6.2f}]")
                print(f"    Selection: {times['selection_time_sec_mean']:6.1f}s")
                print(f"    Quant:     {times['quant_time_sec_mean']:6.1f}s")

        # Print t-test results
        print(f"\n  Statistical Tests:")

        for key in summary[model]:
            if key.startswith('ttest_'):
                comparison = key.replace('ttest_', '').replace('_', ' ')
                test_result = summary[model][key]

                if test_result.get('p_value') is not None:
                    sig_marker = "***" if test_result['p_value'] < 0.001 else \
                                 "**" if test_result['p_value'] < 0.01 else \
                                 "*" if test_result['p_value'] < 0.05 else "ns"

                    print(f"    {comparison}:")
                    print(f"      p={test_result['p_value']:.4f} {sig_marker}, d={test_result['cohens_d']:.3f}")

    print(f"\n{'='*80}")
    print(f"[AGG] Legend: ns=not significant, *p<0.05, **p<0.01, ***p<0.001")
    print(f"{'='*80}\n")


def main():
    """Main aggregation pipeline."""
    print(f"\n{'='*80}")
    print(f"[AGG] RESULTS AGGREGATION AND ANALYSIS")
    print(f"{'='*80}\n")

    # Load results
    results = load_all_results()

    if len(results) == 0:
        print(f"[AGG] ERROR: No results found in ./results/metrics/raw_runs/")
        print(f"[AGG] Please run experiments/runner.py first")
        return

    print(f"[AGG] Loaded {len(results)} total experiments")

    # Group by (model, method)
    grouped = group_results(results)

    # Compute statistics and tests
    summary = aggregate_and_analyze(grouped)

    # Print summary table
    print_summary_table(summary)

    # Save to JSON
    output_path = "./results/metrics/summary_stats.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"[AGG] Summary statistics saved to: {output_path}")
    print(f"[AGG] This file can be directly referenced in thesis Section 5.2")

    print(f"\n{'='*80}")
    print(f"[AGG] AGGREGATION COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
