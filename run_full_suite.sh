#!/bin/bash
#
# Full Experimental Suite Runner
#
# This script runs the complete COLA-Zero evaluation pipeline:
# 1. Runs all experiments (models × methods × seeds)
# 2. Aggregates results and computes statistics
#
# Usage:
#   bash run_full_suite.sh
#
# Requirements:
#   - CUDA-enabled GPU
#   - All dependencies installed (see requirements.txt)
#   - HuggingFace models downloadable
#
# Estimated Runtime:
#   - OPT-6.7B: ~8-10 hours per method (5 seeds)
#   - Llama-2-7B: ~10-12 hours per method (5 seeds)
#   - Llama-2-13B: ~15-20 hours per method (5 seeds)
#   - Total: ~150-180 hours for full suite (3 models × 3 methods × 5 seeds)
#
# Output:
#   - results/metrics/raw_runs/*.json (individual experiments)
#   - results/metrics/summary_stats.json (aggregated statistics)
#
# After Completion:
#   Check results/metrics/summary_stats.json
#   This is what goes into thesis Section 5.2

set -e  # Exit on error

# Configuration
LOG_DIR="./logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_DIR}/full_suite_${TIMESTAMP}.log"

# Create log directory
mkdir -p ${LOG_DIR}

echo "=================================================================="
echo "COLA-Zero Full Experimental Suite"
echo "=================================================================="
echo "Started at: $(date)"
echo "Main log: ${MAIN_LOG}"
echo "=================================================================="
echo ""

# Function to log with timestamp
log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a ${MAIN_LOG}
}

# Check CUDA availability
log "Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')" | tee -a ${MAIN_LOG}

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    log "ERROR: CUDA not available. This suite requires GPU."
    exit 1
fi

log "✓ CUDA available"
echo ""

# Step 1: Run all experiments
log "=================================================================="
log "STEP 1: Running all experiments"
log "=================================================================="
log "This will take 150-180 hours for full suite (3 models × 3 methods × 5 seeds)"
log "Progress will be logged to: ${MAIN_LOG}"
echo ""

python -m experiments.runner 2>&1 | tee -a ${MAIN_LOG}

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "ERROR: Experiment runner failed"
    exit 1
fi

log "✓ All experiments complete"
echo ""

# Step 2: Aggregate results
log "=================================================================="
log "STEP 2: Aggregating results and computing statistics"
log "=================================================================="
echo ""

python -m experiments.aggregate_results 2>&1 | tee -a ${MAIN_LOG}

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "ERROR: Result aggregation failed"
    exit 1
fi

log "✓ Results aggregated"
echo ""

# Final summary
log "=================================================================="
log "FULL SUITE COMPLETE"
log "=================================================================="
log "Ended at: $(date)"
log ""
log "Results:"
log "  - Individual runs: ./results/metrics/raw_runs/*.json"
log "  - Summary stats:   ./results/metrics/summary_stats.json"
log ""
log "Next steps:"
log "  1. Review summary_stats.json for statistical significance"
log "  2. Check p-values and Cohen's d effect sizes"
log "  3. Incorporate results into thesis Section 5.2"
log ""
log "Main log saved to: ${MAIN_LOG}"
log "=================================================================="

exit 0
