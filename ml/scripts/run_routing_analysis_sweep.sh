#!/usr/bin/env bash
# Run routing analysis (knockout + co_activation) on all MoE jobs from the
# pes2o and starcoder data-repetition sweeps. Single-GPU, sequential.
#
# Usage (from /mmfs1/gscratch/zlab/atindra/OLMo-core, inside a salloc'd
# compute node with one GPU):
#
#     bash ml/scripts/run_routing_analysis_sweep.sh
#
# The script:
#   - Loops over every "*moe*" job dir in each sweep
#   - Skips jobs that already have BOTH knockout.json + co_activation.json
#     (so it's safely re-runnable / resumable)
#   - Writes per-job stdout+stderr to a log file under analysis_results/logs/
#   - Prints a one-line START/OK/FAIL/SKIP summary per job

set -euo pipefail

# --- Sweep dirs to analyze ------------------------------------------------
PES2O_SWEEP=/gscratch/zlab/atindra/models/2026_05_08-06_46_47_data_rep_AC_pes2o_olmo2_ml_80M
STARCODER_SWEEP=/gscratch/zlab/atindra/models/2026_05_08-06_42_40_data_rep_AC_starcoder_olmo2_ml_80M

# --- Validation files (used as routing-analysis input tokens) -------------
PES2O_VAL=/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/ml/data/preprocessed/eval-data/perplexity/v3_small_dolma2-tokenizer/dolma_pes2o/val/part-0-00000.npy
STACK_VAL=/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/ml/data/preprocessed/eval-data/perplexity/v3_small_dolma2-tokenizer/dolma_stack/val/part-0-00000.npy

RESULTS_ROOT=/gscratch/zlab/atindra/analysis_results
LOG_ROOT=$RESULTS_ROOT/logs
mkdir -p "$RESULTS_ROOT" "$LOG_ROOT"

# Counters for end-of-run summary
N_OK=0
N_FAIL=0
N_SKIP=0

run_one() {
    local job=$1
    local data=$2
    local label=$3
    local name sweep out_dir log_file
    name=$(basename "$job")
    sweep=$(basename "$(dirname "$job")")
    out_dir="$RESULTS_ROOT/$sweep/$name"
    log_file="$LOG_ROOT/${sweep}__${name}.log"

    # Determine which analyses are missing (smart resume).
    local missing=()
    [[ ! -f "$out_dir/knockout.json"      ]] && missing+=("knockout")
    [[ ! -f "$out_dir/co_activation.json" ]] && missing+=("co_activation")
    [[ ! -f "$out_dir/ossification.json"  ]] && missing+=("ossification")

    if [[ ${#missing[@]} -eq 0 ]]; then
        echo "[$label] SKIP (all done): $name"
        N_SKIP=$((N_SKIP + 1))
        return
    fi

    mkdir -p "$out_dir"
    echo "[$label] START (analyses: ${missing[*]}): $name"
    local t0=$SECONDS
    if python ml/scripts/routing_analysis.py \
            --checkpoint_dir "$job" \
            --data_path "$data" \
            --analysis "${missing[@]}" \
            --device cuda \
            --output_dir "$out_dir" \
            > "$log_file" 2>&1; then
        local dt=$((SECONDS - t0))
        echo "[$label] OK (${dt}s): $name"
        N_OK=$((N_OK + 1))
    else
        echo "[$label] FAIL: $name  --  see $log_file"
        N_FAIL=$((N_FAIL + 1))
    fi
}

echo "============================="
echo "PES2O MoE jobs"
echo "============================="
for job in "$PES2O_SWEEP"/*moe*/; do
    run_one "${job%/}" "$PES2O_VAL" pes2o
done

echo
echo "============================="
echo "STARCODER MoE jobs"
echo "============================="
for job in "$STARCODER_SWEEP"/*moe*/; do
    run_one "${job%/}" "$STACK_VAL" starcoder
done

echo
echo "============================="
echo "Summary: $N_OK ok, $N_FAIL failed, $N_SKIP skipped"
echo "Results under: $RESULTS_ROOT/"
echo "Per-job logs:  $LOG_ROOT/"
echo "============================="
