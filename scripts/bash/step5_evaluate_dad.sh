#!/bin/bash
# Step 5: Evaluate DAD method using same initial MOCU as baselines

set -e

CONFIG_FILE=$1
if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)

# Get baseline results folder
BASELINE_RESULTS=$(cat /tmp/baseline_results_folder_${CONFIG_NAME}.txt 2>/dev/null || echo "")
DAD_POLICY_PATH=$(cat /tmp/dad_policy_path_${CONFIG_NAME}.txt 2>/dev/null || echo "")

if [ -z "$BASELINE_RESULTS" ] || [ ! -d "$BASELINE_RESULTS" ]; then
    echo "Error: Baseline results not found. Run step3_evaluate_baselines.sh first."
    exit 1
fi

if [ -z "$DAD_POLICY_PATH" ] || [ ! -f "$DAD_POLICY_PATH" ]; then
    echo "Error: DAD policy not found. Run step4_train_dad.sh first."
    exit 1
fi

# Parse config parameters
N=$(grep "^N:" $CONFIG_FILE | awk '{print $2}')
UPDATE_CNT=$(grep -A 10 "^experiment:" $CONFIG_FILE | grep "update_count:" | awk '{print $2}')
IT_IDX=$(grep -A 10 "^experiment:" $CONFIG_FILE | grep "it_idx:" | awk '{print $2}')
K_MAX=$(grep -A 10 "^experiment:" $CONFIG_FILE | grep "K_max:" | awk '{print $2}')
NUM_SIMULATIONS=$(grep -A 10 "^experiment:" $CONFIG_FILE | grep "num_simulations:" | awk '{print $2}')

# Validate and set defaults
[ -z "$N" ] && N=5
[ -z "$UPDATE_CNT" ] && UPDATE_CNT=10
[ -z "$IT_IDX" ] && IT_IDX=10
[ -z "$K_MAX" ] && K_MAX=20480
[ -z "$NUM_SIMULATIONS" ] && NUM_SIMULATIONS=10

# Use same result folder as baselines
RESULT_FOLDER="$BASELINE_RESULTS"

export RESULT_FOLDER="$RESULT_FOLDER"
export EVAL_N="$N"
export EVAL_UPDATE_CNT="$UPDATE_CNT"
export EVAL_IT_IDX="$IT_IDX"
export EVAL_K_MAX="$K_MAX"
export EVAL_NUM_SIMULATIONS="$NUM_SIMULATIONS"
export DAD_POLICY_PATH="$DAD_POLICY_PATH"

echo "Running DAD evaluation (Step 5/6)..."
echo "  Using baseline results: $BASELINE_RESULTS"
echo "  DAD policy: $DAD_POLICY_PATH"
echo "  N=$N, update_cnt=$UPDATE_CNT, it_idx=$IT_IDX, K_max=$K_MAX, num_simulations=$NUM_SIMULATIONS"
echo "  Results: $RESULT_FOLDER"

cd "${PROJECT_ROOT}/scripts"
python3 dad_eval.py --baseline_results "$BASELINE_RESULTS" --result_folder "$RESULT_FOLDER"

echo "✓ DAD evaluation complete: $RESULT_FOLDER"

