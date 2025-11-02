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
TIMESTAMP=$(date +"%m%d%Y_%H%M%S")
EPOCHS=$(grep "epochs:" $CONFIG_FILE | awk '{print $2}')
CONSTRAIN_WEIGHT=$(grep "constrain_weight:" $CONFIG_FILE | awk '{print $2}')

MODEL_RUN_FOLDER="${PROJECT_ROOT}/models/${CONFIG_NAME}/${TIMESTAMP}/"
mkdir -p "$MODEL_RUN_FOLDER"

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

echo "Training MPNN predictor (Step 2/5)..."
echo "  Epochs=$EPOCHS, Output: $MODEL_RUN_FOLDER"

cd "${PROJECT_ROOT}/scripts"
# Python scripts remain in scripts/ directory
ABS_TRAIN_FILE=$(cd "$(dirname "$TRAIN_FILE")" && pwd)/$(basename "$TRAIN_FILE")
ABS_MODEL_RUN_FOLDER=$(cd "$MODEL_RUN_FOLDER" && pwd)

python3 train_predictor.py \
    --name "__USE_OUTPUT_DIR__" \
    --data_path "$ABS_TRAIN_FILE" \
    --EPOCH $EPOCHS \
    --Constrain_weight $CONSTRAIN_WEIGHT \
    --output_dir "$ABS_MODEL_RUN_FOLDER"

echo "✓ MPNN predictor trained: ${MODEL_RUN_FOLDER}model.pth"
echo "${CONFIG_NAME}_${TIMESTAMP}" > /tmp/mocu_model_name_${CONFIG_NAME}.txt
echo "$MODEL_RUN_FOLDER" > /tmp/mocu_model_folder_${CONFIG_NAME}.txt

