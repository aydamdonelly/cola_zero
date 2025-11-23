"""
Aggregate raw run metrics for COLA-Zero experiments.

Usage:
    python -m experiments.aggregate --input ./results/metrics/raw_runs --output ./results/metrics/summary.json

Computes per-(model, method) statistics (mean, std, CV, 95% CI) for perplexity and
selection times, plus COLA vs Random significance tests (t-test & Levene).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy import stats  # type: ignore
except ImportError:
    stats = None


@dataclass
class MetricSummary:
    mean: float
    std: float
    cv: Optional[float]
    ci95: Optional[Tuple[float, float]]
    n: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "mean": self.mean,
            "std": self.std,
            "cv_percent": self.cv,
            "ci95": self.ci95,
            "n": self.n,
        }


def compute_summary(values: List[float]) -> MetricSummary:
    arr = np.array(values, dtype=float)
    n = arr.size
    if n == 0:
        return MetricSummary(mean=float("nan"), std=float("nan"), cv=None, ci95=None, n=0)

    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    cv = float((std / mean) * 100) if n > 1 and mean != 0.0 else None

    if n > 1:
        if stats is not None:
            t_multiplier = float(stats.t.ppf(0.975, df=n - 1))
        else:
            # Normal approximation fallback
            t_multiplier = 1.96
        margin = t_multiplier * std / math.sqrt(n)
        ci95 = (mean - margin, mean + margin)
    else:
        ci95 = None

    return MetricSummary(mean=mean, std=std, cv=cv, ci95=ci95, n=n)


def load_runs(input_dir: Path) -> List[Dict[str, object]]:
    runs: List[Dict[str, object]] = []
    for path in sorted(input_dir.glob("*.json")):
        with path.open("r") as fh:
            data = json.load(fh)
            data["_path"] = str(path)
            runs.append(data)
    return runs


def aggregate_runs(runs: List[Dict[str, object]]) -> Dict[str, Dict[str, Dict[str, object]]]:
    grouped: Dict[str, Dict[str, List[Dict[str, object]]]] = defaultdict(lambda: defaultdict(list))

    for run in runs:
        model = run.get("model")
        method = run.get("method")
        if not model or not method:
            continue
        grouped[model][method].append(run)

    summary: Dict[str, Dict[str, Dict[str, object]]] = {}

    for model, method_runs in grouped.items():
        model_key = model.replace("/", "-")
        summary[model_key] = {}

        for method, entries in method_runs.items():
            ppl_values = [entry.get("perplexity") for entry in entries if entry.get("perplexity") is not None]
            selection_times = [
                entry.get("selection_time_sec") for entry in entries if entry.get("selection_time_sec") is not None
            ]
            overhead_values = [
                entry.get("selection_overhead_vs_random_sec")
                for entry in entries
                if entry.get("selection_overhead_vs_random_sec") is not None
            ]

            summary[model_key][method] = {
                "perplexity": compute_summary(ppl_values).as_dict(),
                "selection_time_sec": compute_summary(selection_times).as_dict() if selection_times else None,
                "selection_overhead_vs_random_sec": compute_summary(overhead_values).as_dict()
                if overhead_values
                else None,
                "runs": len(entries),
            }

        # Statistical tests between COLA-Zero and Random
        if "cola_zero" in method_runs and "random" in method_runs:
            cola_ppl = [
                entry.get("perplexity")
                for entry in method_runs["cola_zero"]
                if entry.get("perplexity") is not None
            ]
            random_ppl = [
                entry.get("perplexity")
                for entry in method_runs["random"]
                if entry.get("perplexity") is not None
            ]

            if cola_ppl and random_ppl and len(cola_ppl) > 1 and len(random_ppl) > 1:
                if stats is not None:
                    ttest = stats.ttest_ind(cola_ppl, random_ppl, equal_var=False)
                    levene = stats.levene(cola_ppl, random_ppl)
                    summary[model_key]["stats"] = {
                        "t_test_cola_vs_random": {
                            "statistic": float(ttest.statistic),
                            "p_value": float(ttest.pvalue),
                            "test": "Welch t-test",
                        },
                        "levene_variance_test": {
                            "statistic": float(levene.statistic),
                            "p_value": float(levene.pvalue),
                        },
                    }
                else:
                    summary[model_key]["stats"] = {
                        "t_test_cola_vs_random": {
                            "statistic": None,
                            "p_value": None,
                            "test": "Welch t-test (scipy required)",
                        },
                        "levene_variance_test": {
                            "statistic": None,
                            "p_value": None,
                        },
                    }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate COLA-Zero raw run metrics.")
    parser.add_argument(
        "--input",
        type=str,
        default="./results/metrics/raw_runs",
        help="Directory containing raw run JSON files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/metrics/aggregate_summary.json",
        help="Path to save aggregated summary JSON.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print summary to stdout.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    runs = load_runs(input_dir)
    if not runs:
        raise RuntimeError(f"No JSON files found in {input_dir}")

    summary = aggregate_runs(runs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"[AGGREGATE] Summary written to {output_path}")

    if args.pretty:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
