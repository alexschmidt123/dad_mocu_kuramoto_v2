#!/bin/bash
# Step 6: Generate visualizations
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
N=$(grep "^N:" $CONFIG_FILE | awk '{print $2}')

# Get update_cnt from config (auto-detect from data if not provided)
UPDATE_CNT=$(grep -A 10 "^experiment:" $CONFIG_FILE | grep "update_count:" | awk '{print $2}')
[ -z "$UPDATE_CNT" ] && UPDATE_CNT=""

# Get result folder (most recent)
RESULT_RUN_FOLDER=$(ls -td ${PROJECT_ROOT}/results/${CONFIG_NAME}/*/ 2>/dev/null | head -1 || echo "")

if [ -z "$RESULT_RUN_FOLDER" ]; then
    echo "Error: No results folder found. Run step3_evaluate_baselines.sh first."
    exit 1
fi

echo "Generating visualizations (Step 6/6)..."
echo "  Results: $RESULT_RUN_FOLDER"

cd "${PROJECT_ROOT}/scripts"
# Python scripts remain in scripts/ directory
ABS_RESULT_FOLDER=$(cd "$RESULT_RUN_FOLDER" && pwd)

# Ensure path ends with / for proper path joining in visualize.py
if [ "${ABS_RESULT_FOLDER: -1}" != "/" ]; then
    ABS_RESULT_FOLDER="${ABS_RESULT_FOLDER}/"
fi

# Pass update_cnt if found in config, otherwise let visualize.py auto-detect
if [ -n "$UPDATE_CNT" ]; then
    python3 visualize.py --N $N --update_cnt $UPDATE_CNT --result_folder "$ABS_RESULT_FOLDER"
else
    python3 visualize.py --N $N --result_folder "$ABS_RESULT_FOLDER"
fi

echo "✓ Visualizations generated: ${RESULT_RUN_FOLDER}MOCU_${N}.png"

