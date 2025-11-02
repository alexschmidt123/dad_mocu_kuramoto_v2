#!/bin/bash
# Step 5: Generate visualizations
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

# Get result folder (most recent)
RESULT_RUN_FOLDER=$(ls -td ${PROJECT_ROOT}/results/${CONFIG_NAME}/*/ 2>/dev/null | head -1 || echo "")

if [ -z "$RESULT_RUN_FOLDER" ]; then
    echo "Error: No results folder found. Run step4_evaluate.sh first."
    exit 1
fi

echo "Generating visualizations (Step 5/5)..."
echo "  Results: $RESULT_RUN_FOLDER"

cd "${PROJECT_ROOT}/scripts"
# Python scripts remain in scripts/ directory
ABS_RESULT_FOLDER=$(cd "$RESULT_RUN_FOLDER" && pwd)

python3 visualize.py --N $N --update_cnt 10 --result_folder "$ABS_RESULT_FOLDER"

echo "✓ Visualizations generated: ${RESULT_RUN_FOLDER}MOCU_${N}.png"

