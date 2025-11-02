#!/bin/bash
# Standalone script to run train_dad_policy.py separately
# This helps isolate DAD training from other workflow steps

set -e

# Configuration - UPDATE THESE VALUES:
CONFIG_FILE="${1:-configs/fast_config.yaml}"  # Config file path
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Usage: $0 [config_file]"
    echo "  Or set: EPOCHS=100 BATCH_SIZE=64 $0 configs/fast_config.yaml"
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Parse config
CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
N=$(grep "^N:" $CONFIG_FILE | awk '{print $2}')
DAD_METHOD=$(grep "dad_method:" $CONFIG_FILE | awk '{print $2}' | tr -d '"' || echo "reinforce")

echo "=========================================="
echo "Standalone DAD Training"
echo "=========================================="
echo "Config: $CONFIG_NAME"
echo "N: $N"
echo "Method: $DAD_METHOD"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "=========================================="
echo

# Check for required files
DATA_FILE="${PROJECT_ROOT}/data/${CONFIG_NAME}/dad/dad_trajectories_N${N}_K4_random.pth"
if [ ! -f "$DATA_FILE" ]; then
    echo "Error: DAD trajectory data not found: $DATA_FILE"
    echo "Please run data generation first:"
    echo "  python3 scripts/generate_dad_data.py --N $N --K 4 --num-episodes 100 --output-dir data/${CONFIG_NAME}/dad/"
    exit 1
fi

# Find MPNN model (look for most recent timestamped folder)
MODEL_BASE_DIR="${PROJECT_ROOT}/models/${CONFIG_NAME}"
if [ ! -d "$MODEL_BASE_DIR" ]; then
    echo "Error: Model directory not found: $MODEL_BASE_DIR"
    echo "Please run MPNN training first (step 2)"
    exit 1
fi

# Find most recent model folder (timestamped folders: MMDDYYYY_HHMMSS)
LATEST_MODEL_FOLDER=$(find "$MODEL_BASE_DIR" -maxdepth 1 -type d -name "*_*" | sort | tail -1)
if [ -z "$LATEST_MODEL_FOLDER" ]; then
    echo "Error: No MPNN model found in $MODEL_BASE_DIR"
    exit 1
fi

TIMESTAMP=$(basename "$LATEST_MODEL_FOLDER")
# Model name format: {config_name}_{timestamp}
MODEL_NAME="${CONFIG_NAME}_${TIMESTAMP}"
MODEL_PATH="${LATEST_MODEL_FOLDER}/model.pth"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: MPNN model not found: $MODEL_PATH"
    exit 1
fi

echo "Found MPNN model: $MODEL_NAME"
echo "  Path: $MODEL_PATH"
echo

# Set environment variable for MPNN model name
export MOCU_MODEL_NAME="$MODEL_NAME"


# Determine if we should use MPNN predictor
USE_PREDICTED_MOCU=""
if [ "$DAD_METHOD" = "reinforce" ] && [ -f "$MODEL_PATH" ]; then
    USE_PREDICTED_MOCU="--use-predicted-mocu"
    echo "Using MPNN predictor for fast MOCU estimation"
else
    echo "Using direct CUDA MOCU computation (slow)"
fi
echo

# Run training
cd "${PROJECT_ROOT}/scripts"

echo "Starting DAD training..."
echo "Command:"
echo "  python3 train_dad_policy.py \\"
echo "    --data-path \"$DATA_FILE\" \\"
echo "    --method \"$DAD_METHOD\" \\"
echo "    --name \"dad_policy_N${N}\" \\"
echo "    --epochs $EPOCHS \\"
echo "    --batch-size $BATCH_SIZE \\"
echo "    --output-dir \"$LATEST_MODEL_FOLDER\" \\"
echo "    $USE_PREDICTED_MOCU"
echo

python3 train_dad_policy.py \
    --data-path "$DATA_FILE" \
    --method "$DAD_METHOD" \
    --name "dad_policy_N${N}" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --output-dir "$LATEST_MODEL_FOLDER" \
    $USE_PREDICTED_MOCU

echo
echo "=========================================="
echo "✓ DAD training completed!"
echo "=========================================="
echo "Model saved to: ${LATEST_MODEL_FOLDER}/dad_policy_N${N}.pth"

