#!/bin/bash
# Step 2: Train MPNN predictor (uses PyTorch only - clean CUDA state)
# This script runs in isolation to ensure clean PyTorch CUDA context

set -e

CONFIG_FILE=$1
if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: $0 <config_file> [train_file]"
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
# Extract base config name (remove _K* suffix if present, since MPNN model doesn't depend on K)
BASE_CONFIG_NAME=$(echo "$CONFIG_NAME" | sed 's/_K[0-9]*$//')
EPOCHS=$(grep "epochs:" $CONFIG_FILE | awk '{print $2}')
CONSTRAIN_WEIGHT=$(grep "constrain_weight:" $CONFIG_FILE | awk '{print $2}')

# Use base config name for model folder (MPNN model doesn't depend on K)
MODEL_FOLDER="${PROJECT_ROOT}/models/${BASE_CONFIG_NAME}/"
MODEL_FILE="${MODEL_FOLDER}model.pth"
STATS_FILE="${MODEL_FOLDER}statistics.pth"

# Get train file from argument or temp file
if [ -n "$2" ]; then
    TRAIN_FILE="$2"
else
    TRAIN_FILE=$(cat /tmp/mocu_train_file_${CONFIG_NAME}.txt 2>/dev/null || echo "")
fi

if [ -z "$TRAIN_FILE" ] || [ ! -f "$TRAIN_FILE" ]; then
    echo "Error: Training file not found. Run step1_generate_mocu_data.sh first."
    exit 1
fi

# Check if MPNN model and first MOCU data both exist - skip training if so
if [ -f "$MODEL_FILE" ] && [ -f "$STATS_FILE" ] && [ -f "$TRAIN_FILE" ]; then
    echo "✓ MPNN model already exists: $MODEL_FILE"
    echo "✓ Skipping MPNN training (model and data detected)"
    echo "  Note: MPNN model doesn't depend on K, so same model is used for all K values"
    # Use CONFIG_NAME for tmp files to match other scripts
    echo "${BASE_CONFIG_NAME}" > /tmp/mocu_model_name_${CONFIG_NAME}.txt
    echo "$MODEL_FOLDER" > /tmp/mocu_model_folder_${CONFIG_NAME}.txt
    exit 0
fi

echo "Training MPNN predictor (Step 2/6)..."
echo "  Epochs=$EPOCHS, Output: $MODEL_FOLDER"
mkdir -p "$MODEL_FOLDER"

cd "${PROJECT_ROOT}/scripts"
# Python scripts remain in scripts/ directory
ABS_TRAIN_FILE=$(cd "$(dirname "$TRAIN_FILE")" && pwd)/$(basename "$TRAIN_FILE")
ABS_MODEL_FOLDER=$(cd "$MODEL_FOLDER" && pwd)

python3 train_predictor.py \
    --name "__USE_OUTPUT_DIR__" \
    --data_path "$ABS_TRAIN_FILE" \
    --EPOCH $EPOCHS \
    --Constrain_weight $CONSTRAIN_WEIGHT \
    --output_dir "$ABS_MODEL_FOLDER"

echo "✓ MPNN predictor trained: ${MODEL_FILE}"
# Use BASE_CONFIG_NAME for model name, but CONFIG_NAME for tmp file to match other scripts
echo "${BASE_CONFIG_NAME}" > /tmp/mocu_model_name_${CONFIG_NAME}.txt
echo "$MODEL_FOLDER" > /tmp/mocu_model_folder_${CONFIG_NAME}.txt

