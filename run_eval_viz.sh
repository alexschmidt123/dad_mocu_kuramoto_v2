#!/bin/bash
# Run only evaluation and visualization (skip training steps)
# Usage: bash run_eval_viz.sh configs/fast_config.yaml

set -e

CONFIG_FILE=$1
if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: bash run_eval_viz.sh <config_file>"
    echo "Example: bash run_eval_viz.sh configs/fast_config.yaml"
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)

# Check if model files exist (optional - scripts will handle missing models)
MOCU_MODEL_NAME=$(cat /tmp/mocu_model_name_${CONFIG_NAME}.txt 2>/dev/null || echo "")
DAD_POLICY_PATH=$(cat /tmp/dad_policy_path_${CONFIG_NAME}.txt 2>/dev/null || echo "")

if [ -z "$MOCU_MODEL_NAME" ]; then
    echo "Warning: MOCU_MODEL_NAME not found in /tmp/mocu_model_name_${CONFIG_NAME}.txt"
    echo "  Evaluation will try to find model automatically or use defaults"
fi

if [ -z "$DAD_POLICY_PATH" ]; then
    echo "Note: DAD_POLICY_PATH not found - DAD method will use default path if needed"
fi

echo "========================================"
echo "Running Evaluation and Visualization"
echo "========================================"
echo "Config: $CONFIG_FILE"
echo ""

# Step 4: Run evaluation
echo "[Step 1/2] Running evaluation..."
bash "${PROJECT_ROOT}/scripts/bash/step4_evaluate.sh" "$CONFIG_FILE"

# Step 5: Generate visualizations
echo ""
echo "[Step 2/2] Generating visualizations..."
bash "${PROJECT_ROOT}/scripts/bash/step5_visualize.sh" "$CONFIG_FILE"

echo ""
echo "========================================"
echo "✓ Complete!"
echo "========================================"
RESULT_RUN_FOLDER=$(ls -td ${PROJECT_ROOT}/results/${CONFIG_NAME}/*/ 2>/dev/null | head -1)
if [ -n "$RESULT_RUN_FOLDER" ]; then
    echo "Results: $RESULT_RUN_FOLDER"
fi
