#!/bin/bash
# Step 4: Evaluate methods (may use PyCUDA for evaluation metrics)
# This script runs in isolation

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

# Get info from previous steps
MOCU_MODEL_NAME=$(cat /tmp/mocu_model_name_${CONFIG_NAME}.txt 2>/dev/null || echo "")
DAD_POLICY_PATH=$(cat /tmp/dad_policy_path_${CONFIG_NAME}.txt 2>/dev/null || echo "")

# Parse config parameters
N=$(grep "^N:" $CONFIG_FILE | awk '{print $2}')
UPDATE_CNT=$(grep -A 3 "^experiment:" $CONFIG_FILE | grep "update_count:" | awk '{print $2}')
IT_IDX=$(grep -A 3 "^experiment:" $CONFIG_FILE | grep "it_idx:" | awk '{print $2}')
K_MAX=$(grep -A 3 "^experiment:" $CONFIG_FILE | grep "K_max:" | awk '{print $2}')
NUM_SIMULATIONS=$(grep -A 3 "^experiment:" $CONFIG_FILE | grep "num_simulations:" | awk '{print $2}')

# Parse methods from config
METHODS=$(grep -A 20 "^  methods:" $CONFIG_FILE | grep '    - "' | sed 's/.*"\(.*\)".*/\1/' | grep -v '^#' | tr '\n' ',' | sed 's/,$//')

RESULT_RUN_FOLDER="${PROJECT_ROOT}/results/${CONFIG_NAME}/${TIMESTAMP}/"
mkdir -p "$RESULT_RUN_FOLDER"

export MOCU_MODEL_NAME="$MOCU_MODEL_NAME"
export RESULT_FOLDER="$RESULT_RUN_FOLDER"
export EVAL_N="$N"
export EVAL_UPDATE_CNT="$UPDATE_CNT"
export EVAL_IT_IDX="$IT_IDX"
export EVAL_K_MAX="$K_MAX"
export EVAL_NUM_SIMULATIONS="$NUM_SIMULATIONS"
if [ -n "$DAD_POLICY_PATH" ]; then
    export DAD_POLICY_PATH="$DAD_POLICY_PATH"
fi

echo "Running evaluation (Step 4/5)..."
echo "  Methods: $METHODS"
echo "  N=$N, update_cnt=$UPDATE_CNT, it_idx=$IT_IDX, K_max=$K_MAX, num_simulations=$NUM_SIMULATIONS"
echo "  Results: $RESULT_RUN_FOLDER"

cd "${PROJECT_ROOT}/scripts"
# Python scripts remain in scripts/ directory
python3 evaluate.py --methods "$METHODS"

echo "✓ Evaluation complete: $RESULT_RUN_FOLDER"

