#!/bin/bash
# Step 3: Evaluate baseline methods first
# This runs baseline methods BEFORE DAD training, regardless of config
# Uses PyCUDA for MOCU computation (matches original paper 2023 workflow)

set -e

CONFIG_FILE=$1
if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
TIMESTAMP=$(date +"%m%d%Y_%H%M%S")

# Use timestamp for results folder to allow multiple runs

# Get info from previous steps
MOCU_MODEL_NAME=$(cat /tmp/mocu_model_name_${CONFIG_NAME}.txt 2>/dev/null || echo "")

# Parse config parameters
N=$(grep "^N:" $CONFIG_FILE | awk '{print $2}')
UPDATE_CNT=$(grep -A 10 "^experiment:" $CONFIG_FILE | grep "update_count:" | awk '{print $2}')
IT_IDX=$(grep -A 10 "^experiment:" $CONFIG_FILE | grep "it_idx:" | awk '{print $2}')
K_MAX=$(grep -A 10 "^experiment:" $CONFIG_FILE | grep "K_max:" | awk '{print $2}')
NUM_SIMULATIONS=$(grep -A 10 "^experiment:" $CONFIG_FILE | grep "num_simulations:" | awk '{print $2}')

# Validate and set defaults if empty
[ -z "$N" ] && N=5
[ -z "$UPDATE_CNT" ] && UPDATE_CNT=10
[ -z "$IT_IDX" ] && IT_IDX=10
[ -z "$K_MAX" ] && K_MAX=20480
[ -z "$NUM_SIMULATIONS" ] && NUM_SIMULATIONS=10

# Run baseline methods (matching original paper + regression_scorer)
BASELINE_METHODS="iNN,NN,ODE,ENTROPY,RANDOM,REGRESSION_SCORER"

RESULT_RUN_FOLDER="${PROJECT_ROOT}/results/${CONFIG_NAME}/${TIMESTAMP}/"
mkdir -p "$RESULT_RUN_FOLDER"

export MOCU_MODEL_NAME="$MOCU_MODEL_NAME"
export RESULT_FOLDER="$RESULT_RUN_FOLDER"
export EVAL_N="$N"
export EVAL_UPDATE_CNT="$UPDATE_CNT"
export EVAL_IT_IDX="$IT_IDX"
export EVAL_K_MAX="$K_MAX"
export EVAL_NUM_SIMULATIONS="$NUM_SIMULATIONS"

echo "Running baseline evaluation (Step 3/6)..."
echo "  Methods: $BASELINE_METHODS"
echo "  N=$N, update_cnt=$UPDATE_CNT, it_idx=$IT_IDX, K_max=$K_MAX, num_simulations=$NUM_SIMULATIONS"
echo "  Results: $RESULT_RUN_FOLDER"

cd "${PROJECT_ROOT}/scripts"
python3 evaluate.py --methods "$BASELINE_METHODS"

echo "✓ Baseline evaluation complete: $RESULT_RUN_FOLDER"
echo "$RESULT_RUN_FOLDER" > /tmp/baseline_results_folder_${CONFIG_NAME}.txt

# Step 3.5: Visualize baseline-only results
echo ""
echo "Generating baseline-only visualizations..."
ABS_RESULT_FOLDER=$(cd "$RESULT_RUN_FOLDER" && pwd)
if [ "${ABS_RESULT_FOLDER: -1}" != "/" ]; then
    ABS_RESULT_FOLDER="${ABS_RESULT_FOLDER}/"
fi

if [ -n "$UPDATE_CNT" ]; then
    python3 visualize.py --N $N --update_cnt $UPDATE_CNT --result_folder "$ABS_RESULT_FOLDER" --baseline_only
else
    python3 visualize.py --N $N --result_folder "$ABS_RESULT_FOLDER" --baseline_only
fi

echo "✓ Baseline-only visualizations generated"

