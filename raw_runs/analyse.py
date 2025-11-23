#!/usr/bin/env python3
import argparse
import json
import math
import os
import glob
from collections import defaultdict, OrderedDict
from statistics import mean, pstdev, stdev
import csv
from typing import Dict, List, Tuple

def collect_metrics(paths: List[str]) -> Dict[str, Dict[str, List[float]]]:
    """
    Load all json files in paths and collect metrics by method.
    Returns: { method: { metric_name: [values...] } }
    """
    by_method: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for path in paths:
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Skipping {path}: {e}")
            continue

        method = data.get("method")
        model = data.get("model")
        if method == "gptq_default" and model == "meta-llama/Meta-Llama-3-8B":
            continue
        if not method:
            print(f"[WARN] {path} has no 'method'; skipping.")
            continue

        # Perplexity
        ppl = data.get("perplexity")
        if isinstance(ppl, (int, float)):
            by_method[method]["perplexity"].append(float(ppl))

        # Downstream metrics (arc_easy, hellaswag, piqa, winogrande, average)
        downstream = data.get("downstream", {})
        for k, v in downstream.items():
            if isinstance(v, (int, float)):
                by_method[method][k].append(float(v))

    return by_method

def safe_stdev(values: List[float]) -> float:
    """Sample standard deviation with ddof=1; returns 0.0 if not enough samples."""
    if len(values) <= 1:
        return 0.0
    try:
        return stdev(values)
    except Exception:
        # Fallback to population stdev if something odd happens
        return pstdev(values)

def summarize(by_method: Dict[str, Dict[str, List[float]]]
             ) -> List[Tuple[str, str, int, float, float, float]]:
    """
    Produce rows of (metric, method, n, mean, std, cv).
    cv is std/mean; 0 if mean==0 or n<2.
    """
    # Collect the complete set of metric names to keep ordering stable
    all_metrics = set()
    for method in by_method:
        all_metrics.update(by_method[method].keys())
    ordered_metrics = sorted(all_metrics, key=lambda s: ["perplexity","arc_easy","hellaswag","piqa","winogrande","average"].index(s) if s in ["perplexity","arc_easy","hellaswag","piqa","winogrande","average"] else 999)

    rows = []
    for metric in ordered_metrics:
        for method in sorted(by_method.keys()):
            vals = by_method[method].get(metric, [])
            if not vals:
                continue
            n = len(vals)
            m = mean(vals)
            sd = safe_stdev(vals)
            cv = (sd / m) if (n >= 2 and m != 0) else 0.0
            rows.append((metric, method, n, m, sd, cv))
    return rows

def print_table(rows: List[Tuple[str, str, int, float, float, float]], cv_pct=True):
    # Pretty print to console
    headers = ["metric", "method", "n", "mean", "std", "cv" + ("(%)" if cv_pct else "")]
    col_widths = [max(len(h), 12) for h in headers]

    def fmt_row(r):
        metric, method, n, m, sd, cv = r
        cv_disp = cv * 100 if cv_pct else cv
        return [
            f"{metric:<{col_widths[0]}}",
            f"{method:<{col_widths[1]}}",
            f"{n:<{col_widths[2]}}",
            f"{m:<{col_widths[3]}.6f}",
            f"{sd:<{col_widths[4]}.6f}",
            f"{cv_disp:<{col_widths[5]}.2f}",
        ]

    # Adjust widths using data
    for metric, method, n, m, sd, cv in rows:
        col_widths[0] = max(col_widths[0], len(str(metric)))
        col_widths[1] = max(col_widths[1], len(str(method)))

    header_line = "  ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))
    for r in rows:
        print("  ".join(fmt_row(r)))

def write_csv(rows: List[Tuple[str, str, int, float, float, float]], out_path: str):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "method", "n", "mean", "std", "cv"])
        for metric, method, n, m, sd, cv in rows:
            writer.writerow([metric, method, n, f"{m:.8f}", f"{sd:.8f}", f"{cv:.8f}"])
    print(f"\n[INFO] CSV written to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Summarize mean and CV for perplexity and zero-shot tasks by sampling method.")
    parser.add_argument(
        "path",
        help="Directory containing JSON run files, or a glob pattern (e.g., '/path/*.json')."
    )
    parser.add_argument(
        "--csv",
        help="Optional path to write a CSV with results (e.g., results_summary.csv)."
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=[],
        help="Optional metric names to include (default: all found). Examples: perplexity arc_easy hellaswag piqa winogrande average"
    )
    args = parser.parse_args()

    # Expand files
    if os.path.isdir(args.path):
        pattern = os.path.join(args.path, "*.json")
        files = sorted(glob.glob(pattern))
    else:
        files = sorted(glob.glob(args.path))

    if not files:
        print(f"[ERROR] No JSON files found for '{args.path}'.")
        return

    by_method = collect_metrics(files)
    rows = summarize(by_method)

    # Filter by --include if provided
    if args.include:
        inc = set(args.include)
        rows = [r for r in rows if r[0] in inc]

    print_table(rows, cv_pct=True)

    if args.csv:
        write_csv(rows, args.csv)

if __name__ == "__main__":
    main()

