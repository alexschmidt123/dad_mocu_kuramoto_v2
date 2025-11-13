#!/bin/bash
# Sweep K values: Run entire experiment pipeline with different K values
# Usage: bash scripts/bash/run_sweepK.sh <config_file>
# Example: bash scripts/bash/run_sweepK.sh configs/N5_config.yaml

set -e

CONFIG_FILE=$1
if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: $0 <config_file>"
    echo "Example: $0 configs/N5_config.yaml"
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Resolve config file path
if [[ "$CONFIG_FILE" != /* ]]; then
    CONFIG_FILE="${PROJECT_ROOT}/${CONFIG_FILE}"
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
N=$(grep "^N:" "$CONFIG_FILE" | awk '{print $2}')

# K values to sweep
K_VALUES=(4 6 8 10)

echo "=========================================="
echo "K Sweep Experiment"
echo "=========================================="
echo "Config: $CONFIG_NAME"
echo "N: $N"
echo "K values: ${K_VALUES[@]}"
echo "=========================================="
echo ""

# Create temporary directory for modified configs
TMP_DIR="${PROJECT_ROOT}/tmp_sweepK_${CONFIG_NAME}"
mkdir -p "$TMP_DIR"

# Function to update K in config file
update_config_K() {
    local config_file=$1
    local k_value=$2
    local output_file=$3
    
    # Use Python to properly parse and update YAML
    python3 << PYEOF
import yaml
import sys

with open('$config_file', 'r') as f:
    config = yaml.safe_load(f)

# Update experiment.update_count (this is K)
if 'experiment' in config:
    config['experiment']['update_count'] = $k_value

# Update dad_data.K (should match update_count)
if 'dad_data' in config:
    config['dad_data']['K'] = $k_value

# Write updated config (preserve order and formatting)
with open('$output_file', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

print(f"Updated config: update_count={config['experiment']['update_count']}, dad_data.K={config['dad_data']['K']}")
PYEOF
}

# Run experiment for each K value
for K in "${K_VALUES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running with K=$K"
    echo "=========================================="
    
    # Create temporary config with updated K
    # Use a unique config name that includes K to avoid conflicts
    TMP_CONFIG_NAME="${CONFIG_NAME}_K${K}"
    TMP_CONFIG="${TMP_DIR}/${TMP_CONFIG_NAME}.yaml"
    update_config_K "$CONFIG_FILE" "$K" "$TMP_CONFIG"
    
    echo "Temporary config: $TMP_CONFIG"
    echo "Config name: $TMP_CONFIG_NAME"
    echo ""
    
    # Run full pipeline for this K value
    echo "Step 1: Generate MOCU data..."
    bash "${PROJECT_ROOT}/scripts/bash/step1_generate_mocu_data.sh" "$TMP_CONFIG" || {
        echo "Error in step1 for K=$K"
        continue
    }
    
    echo ""
    echo "Step 2: Train MPNN predictor..."
    bash "${PROJECT_ROOT}/scripts/bash/step2_train_mpnn.sh" "$TMP_CONFIG" || {
        echo "Error in step2 for K=$K"
        continue
    }
    
    echo ""
    echo "Step 3: Evaluate baselines..."
    bash "${PROJECT_ROOT}/scripts/bash/step3_evaluate_baselines.sh" "$TMP_CONFIG" || {
        echo "Error in step3 for K=$K"
        continue
    }
    
    echo ""
    echo "Step 4: Train DAD policy (K=$K will be in model name)..."
    bash "${PROJECT_ROOT}/scripts/bash/step4_train_dad.sh" "$TMP_CONFIG" || {
        echo "Error in step4 for K=$K"
        continue
    }
    
    echo ""
    echo "Step 5: Evaluate DAD..."
    bash "${PROJECT_ROOT}/scripts/bash/step5_evaluate_dad.sh" "$TMP_CONFIG" || {
        echo "Error in step5 for K=$K"
        continue
    }
    
    echo ""
    echo "Step 6: Visualize results..."
    bash "${PROJECT_ROOT}/scripts/bash/step6_visualize.sh" "$TMP_CONFIG" || {
        echo "Error in step6 for K=$K"
        continue
    }
    
    # Find the results folder with timestamp (created in step3)
    # Results are stored in results/${CONFIG_NAME}_K${K}/${TIMESTAMP}/
    # We need to find the most recent timestamp folder
    RESULTS_BASE_DIR="${PROJECT_ROOT}/results/${TMP_CONFIG_NAME}"
    if [ -d "$RESULTS_BASE_DIR" ]; then
        # Get the most recent timestamp folder
        LATEST_RESULT_DIR=$(ls -td "${RESULTS_BASE_DIR}"/*/ 2>/dev/null | head -1)
        
        if [ -n "$LATEST_RESULT_DIR" ] && [ -d "$LATEST_RESULT_DIR" ]; then
            # Extract timestamp from folder name
            TIMESTAMP=$(basename "$LATEST_RESULT_DIR")
            
            # Copy config file to results folder with new name
            CONFIG_COPY_NAME="N${N}_K${K}_config.yaml"
            CONFIG_COPY_PATH="${LATEST_RESULT_DIR}${CONFIG_COPY_NAME}"
            
            cp "$TMP_CONFIG" "$CONFIG_COPY_PATH"
            echo "✓ Copied config to: $CONFIG_COPY_PATH"
        else
            echo "⚠ Could not find results folder for K=$K"
        fi
    else
        echo "⚠ Results directory not found: $RESULTS_BASE_DIR"
    fi
    
    echo ""
    echo "✓ Completed K=$K experiment"
    echo ""
done

# Cleanup temporary configs
echo "Cleaning up temporary configs..."
rm -rf "$TMP_DIR"

echo ""
echo "=========================================="
echo "K Sweep Complete!"
echo "=========================================="
echo "Results for each K are in:"
for K in "${K_VALUES[@]}"; do
    echo "  K=$K: results/${CONFIG_NAME}_K${K}/"
done
echo ""
echo "DAD models for each K are in:"
for K in "${K_VALUES[@]}"; do
    MODEL_DIR=$(cat /tmp/mocu_model_folder_${CONFIG_NAME}_K${K}.txt 2>/dev/null || echo "models/${CONFIG_NAME}_K${K}/")
    echo "  K=$K: ${MODEL_DIR}dad_policy_N${N}_K${K}.pth"
done
echo ""

