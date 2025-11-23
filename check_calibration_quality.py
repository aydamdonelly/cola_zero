"""
Quick check: Was ist der Unterschied zwischen Random und COLA-Zero Calibration?
"""

import json
import os
from pathlib import Path

def check_quantization_logs():
    """Check GPTQ quantization logs for loss differences."""

    print("="*80)
    print("CHECKING QUANTIZATION LOGS")
    print("="*80)

    # Find log files
    log_files = list(Path(".").glob("gptq_log_*.log"))

    if not log_files:
        print("❌ No GPTQ log files found!")
        return

    print(f"\nFound {len(log_files)} log files:")
    for log in log_files:
        print(f"  - {log}")

    # Parse last 2 logs (random + cola_zero)
    if len(log_files) >= 2:
        log_files_sorted = sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True)

        print("\n" + "="*80)
        print("COMPARING LAST TWO RUNS")
        print("="*80)

        for i, log_file in enumerate(log_files_sorted[:2]):
            print(f"\n--- Log {i+1}: {log_file.name} ---")

            with open(log_file, 'r') as f:
                lines = f.readlines()

            # Extract first layer loss (indicator of calibration quality)
            first_layer_losses = []
            for line in lines[:50]:  # Check first 50 lines
                if '| gptq' in line and '| 0 |' in line:
                    parts = line.split('|')
                    if len(parts) >= 4:
                        try:
                            loss = float(parts[3].strip())
                            module = parts[2].strip()
                            first_layer_losses.append((module, loss))
                        except:
                            pass

            if first_layer_losses:
                print("First layer quantization losses:")
                for module, loss in first_layer_losses:
                    print(f"  {module}: {loss:.10f}")

                avg_loss = sum(l for _, l in first_layer_losses) / len(first_layer_losses)
                print(f"Average first layer loss: {avg_loss:.10f}")
            else:
                print("⚠️  Could not parse losses from log")


def check_model_sizes():
    """Check if quantized models have the same size."""

    print("\n" + "="*80)
    print("CHECKING MODEL SIZES")
    print("="*80)

    models_dir = Path("./results/quantized_models")

    if not models_dir.exists():
        print("❌ Models directory not found!")
        return

    model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and "seed1" in d.name and "c4" not in d.name]

    if not model_dirs:
        print("❌ No seed1 models found!")
        return

    print(f"\nFound {len(model_dirs)} model directories:")

    for model_dir in model_dirs:
        safetensors_files = list(model_dir.glob("*.safetensors"))

        if safetensors_files:
            total_size = sum(f.stat().st_size for f in safetensors_files)
            total_size_mb = total_size / (1024 * 1024)

            # Extract method from dirname
            method = "unknown"
            if "__random__" in model_dir.name:
                method = "random"
            elif "__cola_zero__" in model_dir.name:
                method = "cola_zero"

            print(f"\n{method:15} {model_dir.name}")
            print(f"  Total size: {total_size_mb:.2f} MB")
            print(f"  Files: {len(safetensors_files)}")
        else:
            print(f"\n⚠️  {model_dir.name}: No safetensors files found")


def check_result_json():
    """Check if result JSON files exist and show perplexity."""

    print("\n" + "="*80)
    print("CHECKING RESULT JSON FILES")
    print("="*80)

    results_dir = Path("./results/metrics/raw_runs")

    if not results_dir.exists():
        print("❌ Results directory not found!")
        return

    json_files = list(results_dir.glob("*__seed1.json"))

    if not json_files:
        print("❌ No seed1 JSON results found!")
        return

    print(f"\nFound {len(json_files)} result files:")

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        method = data.get('method', 'unknown')
        calib_source = data.get('calibration_source', 'unknown')
        ppl = data.get('perplexity', -1)
        ppl_c4 = data.get('ppl_c4', -1)

        print(f"\n{method:15} (calib: {calib_source})")
        print(f"  WikiText-2 PPL: {ppl:.2f}")
        print(f"  C4 PPL: {ppl_c4:.2f}")

        # Check calibration metadata
        calib_meta = data.get('calib_meta', {})
        if 'n_samples' in calib_meta:
            print(f"  Calibration samples: {calib_meta.get('n_samples', '?')}")
        if 'backfilled' in calib_meta:
            backfilled = calib_meta.get('backfilled', 0)
            if backfilled > 0:
                print(f"  ⚠️  Backfilled samples: {backfilled}")
        if 'token_coverage_ratio' in calib_meta:
            coverage = calib_meta.get('token_coverage_ratio', 0)
            print(f"  Token coverage: {coverage:.2f}×")
            if coverage < 1.0:
                print(f"  ⚠️  INSUFFICIENT TOKEN COVERAGE!")


if __name__ == "__main__":
    check_quantization_logs()
    check_model_sizes()
    check_result_json()

    print("\n" + "="*80)
    print("DONE")
    print("="*80)
