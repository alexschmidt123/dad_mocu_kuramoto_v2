#!/bin/bash
# Sweep K values: Run entire experiment pipeline with different K values
# This script simply calls run.sh multiple times with different K values
# Usage: bash run_sweepK.sh <config_file> [K_values...]
# Example: bash run_sweepK.sh configs/N5_config.yaml
#          bash run_sweepK.sh configs/N5_config.yaml 4 6 8 10

# Don't use set -e here - we want to continue even if one K fails
# set -e

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
    echo "         (default: sweeps K=1,2,3,4,5,6,7)"
    echo "         $0 configs/N5_config.yaml 2 4 6"
    echo "         (custom: sweeps K=2,4,6)"
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

# Track experiment directories created during sweep
declare -a SWEEP_EXPERIMENTS=()

# K values to sweep (use command line args if provided, otherwise default)
# Default: sweep K from 1 to 7 (limited budgets where DAD might beat greedy)
if [ $# -gt 1 ]; then
    shift  # Remove config_file from args
    K_VALUES=("$@")
else
    K_VALUES=(1 2 3 4 5 6 7)
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
EXPERIMENTS_DIR="${PROJECT_ROOT}/experiments"
mkdir -p "$EXPERIMENTS_DIR"

for K in "${K_VALUES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running with K=$K"
    echo "=========================================="
    echo ""
    
    LATEST_BEFORE=$(ls -td "${EXPERIMENTS_DIR}/${CONFIG_NAME}_"*/ 2>/dev/null | head -1)
    
    # Simply call run.sh with K as second argument
    if bash "${PROJECT_ROOT}/run.sh" "$CONFIG_FILE" "$K"; then
        echo ""
        echo "✓ Completed K=$K experiment"
        echo ""
        LATEST_AFTER=$(ls -td "${EXPERIMENTS_DIR}/${CONFIG_NAME}_"*/ 2>/dev/null | head -1)
        if [ -n "$LATEST_AFTER" ] && [ "$LATEST_AFTER" != "$LATEST_BEFORE" ]; then
            SWEEP_EXPERIMENTS+=("K=$K -> $LATEST_AFTER")
        else
            SWEEP_EXPERIMENTS+=("K=$K -> (experiment folder not detected)")
        fi
    else
        echo ""
        echo "⚠ Error: Failed for K=$K"
        echo "Continuing with next K value..."
        echo ""
        SWEEP_EXPERIMENTS+=("K=$K -> (failed)")
    fi
done

echo ""
echo "=========================================="
echo "K Sweep Complete!"
echo "=========================================="
echo ""
echo "Experiment folders created:"
if [ ${#SWEEP_EXPERIMENTS[@]} -gt 0 ]; then
    for entry in "${SWEEP_EXPERIMENTS[@]}"; do
        echo "  $entry"
    done
else
    echo "  (none detected)"
fi
echo ""
echo "Each folder contains:"
echo "  - config.yaml (resolved config with K override)"
echo "  - dad_data/ (trajectories)"
echo "  - dad_models/ (policy checkpoints + metrics)"
echo "  - logs/workflow.log"
echo "  - eval/ (baseline + DAD evaluation outputs)"
echo ""
