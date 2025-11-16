#!/bin/bash
# Sweep K values: Run entire experiment pipeline with different K values
# This script simply calls run.sh multiple times with different K values
# Usage: bash run_sweepK.sh <config_file> [K_values...]
# Example: bash run_sweepK.sh configs/N5_config.yaml
#          bash run_sweepK.sh configs/N5_config.yaml 4 6 8 10

set -e

# Get script directory and determine project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# If script is in scripts/bash/, go up 2 levels; if in root, use current dir
if [[ "$SCRIPT_DIR" == *"/scripts/bash" ]]; then
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
    PROJECT_ROOT="$SCRIPT_DIR"
fi
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

CONFIG_FILE=$1
if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: $0 <config_file> [K_values...]"
    echo "Example: $0 configs/N5_config.yaml"
    echo "         $0 configs/N5_config.yaml 4 6 8 10"
    exit 1
fi

# Resolve config file path
if [[ "$CONFIG_FILE" != /* ]]; then
    CONFIG_FILE="${PROJECT_ROOT}/${CONFIG_FILE}"
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
# Extract base config name (remove _K* suffix if present, since results folder doesn't use _K suffix)
BASE_CONFIG_NAME=$(echo "$CONFIG_NAME" | sed 's/_K[0-9]*$//')
N=$(grep "^N:" "$CONFIG_FILE" | awk '{print $2}')

# K values to sweep (use command line args if provided, otherwise default)
if [ $# -gt 1 ]; then
    shift  # Remove config_file from args
    K_VALUES=("$@")
else
    K_VALUES=(4 6 8 10)
fi

echo "=========================================="
echo "K Sweep Experiment"
echo "=========================================="
echo "Config: $CONFIG_NAME"
echo "N: $N"
echo "K values: ${K_VALUES[@]}"
echo "=========================================="
echo ""
echo "This will run: bash run.sh $CONFIG_FILE <K> for each K value"
echo ""

# Run run.sh for each K value
for K in "${K_VALUES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running with K=$K"
    echo "=========================================="
    echo ""
    
    # Simply call run.sh with K as second argument
    bash "${PROJECT_ROOT}/run.sh" "$CONFIG_FILE" "$K" || {
        echo "Error: Failed for K=$K"
        echo "Continuing with next K value..."
        continue
    }
    
    echo ""
    echo "✓ Completed K=$K experiment"
    echo ""
done

echo ""
echo "=========================================="
echo "K Sweep Complete!"
echo "=========================================="
echo ""
echo "Results for each K are in:"
# All K values share the same results folder (BASE_CONFIG_NAME)
RESULT_DIR="${PROJECT_ROOT}/results/${BASE_CONFIG_NAME}/"
if [ -d "$RESULT_DIR" ]; then
    LATEST=$(ls -td "${RESULT_DIR}"/*/ 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        echo "  All K values: $LATEST"
        echo "  (All K values share the same results folder)"
    else
        echo "  Results folder: $RESULT_DIR (no timestamp folder found)"
    fi
else
    echo "  Results folder: $RESULT_DIR (not found)"
fi
echo ""
echo "DAD models for each K are in:"
for K in "${K_VALUES[@]}"; do
    # Use BASE_CONFIG_NAME for model folder (all K values share same folder)
    MODEL_DIR="${PROJECT_ROOT}/models/${BASE_CONFIG_NAME}/"
    MODEL_FILE="${MODEL_DIR}dad_policy_N${N}_K${K}.pth"
    if [ -f "$MODEL_FILE" ]; then
        echo "  K=$K: $MODEL_FILE"
    else
        echo "  K=$K: (not found)"
    fi
done
echo ""
