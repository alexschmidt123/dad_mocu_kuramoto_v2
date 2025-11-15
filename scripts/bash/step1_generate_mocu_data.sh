#!/bin/bash
# Step 1: Generate MPNN training data (uses PyCUDA - original paper workflow)

set -e

# Load config
CONFIG_FILE=$1
if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi

# Parse config
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
# Extract base config name (remove _K* suffix if present, since MOCU data doesn't depend on K)
BASE_CONFIG_NAME=$(echo "$CONFIG_NAME" | sed 's/_K[0-9]*$//')
N=$(grep "^N:" $CONFIG_FILE | awk '{print $2}')
SAMPLES=$(grep "samples_per_type:" $CONFIG_FILE | awk '{print $2}')
TRAIN_SIZE=$(grep "train_size:" $CONFIG_FILE | awk '{print $2}')
K_MAX=$(grep -A 3 "^dataset:" $CONFIG_FILE | grep "K_max:" | awk '{print $2}')
SAVE_JSON=$(grep "save_json:" $CONFIG_FILE | awk '{print $2}')

# Parse data generation optimization settings (optional, with defaults)
NUM_WORKERS=$(grep -A 3 "^data_generation:" $CONFIG_FILE | grep "  num_workers:" | awk '{print $2}')
CHUNK_SIZE=$(grep -A 3 "^data_generation:" $CONFIG_FILE | grep "  chunk_size:" | awk '{print $2}')

# Use base config name for data folder (MOCU data doesn't depend on K)
DATA_FOLDER="${PROJECT_ROOT}/data/${BASE_CONFIG_NAME}/"
mkdir -p "$DATA_FOLDER"

# Check if data already exists (MOCU data doesn't depend on K, so we can reuse it)
TRAIN_FILE=$(find "$DATA_FOLDER" -name "*_${N}o_train.pth" -type f 2>/dev/null | head -1)

if [ -n "$TRAIN_FILE" ]; then
    echo "✓ Found existing MPNN training data: $TRAIN_FILE"
    echo "✓ Skipping data generation (data already exists)"
    echo "  Note: MOCU data doesn't depend on K, so same data is used for all K values"
    # Save train file path for next steps (use CONFIG_NAME for tmp file to match other scripts)
    echo "$TRAIN_FILE" > /tmp/mocu_train_file_${CONFIG_NAME}.txt
    exit 0
fi

echo "Generating MPNN training data (Step 1/6)..."
echo "  N=$N, Samples per type=$SAMPLES, Train size=$TRAIN_SIZE, K_max=$K_MAX"
echo "  Note: This data will be reused for all K values (MOCU data doesn't depend on K)"

cd "${PROJECT_ROOT}/scripts"
# Python scripts remain in scripts/ directory
ABS_DATA_FOLDER=$(cd "$DATA_FOLDER" && pwd)

CMD="python3 generate_mocu_data.py --N $N --samples_per_type $SAMPLES --train_size $TRAIN_SIZE --K_max $K_MAX --output_dir $ABS_DATA_FOLDER"

# MOCU is always computed twice (always enabled for stability)

if [ "$SAVE_JSON" = "true" ]; then
    CMD="$CMD --save_json"
fi

# Add multiprocessing settings if specified in config
if [ -n "$NUM_WORKERS" ]; then
    CMD="$CMD --num_workers $NUM_WORKERS"
fi

if [ -n "$CHUNK_SIZE" ]; then
    CMD="$CMD --chunk_size $CHUNK_SIZE"
fi

eval $CMD

TRAIN_FILE=$(find "$DATA_FOLDER" -name "*_${N}o_train.pth" -type f 2>/dev/null | head -1)
if [ -z "$TRAIN_FILE" ]; then
    echo "Error: No training file found"
    exit 1
fi

echo "✓ MPNN data generated: $TRAIN_FILE"
echo "$TRAIN_FILE" > /tmp/mocu_train_file_${CONFIG_NAME}.txt

