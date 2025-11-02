#!/bin/bash
# Step 3: Train DAD policy (uses MPNN predictor - clean CUDA state, no PyCUDA)
# This script runs in isolation to ensure no PyCUDA context conflicts

set -e

CONFIG_FILE=$1
if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
N=$(grep "^N:" $CONFIG_FILE | awk '{print $2}')
DAD_METHOD=$(grep "dad_method:" $CONFIG_FILE | awk '{print $2}' | tr -d '"' || echo "reinforce")

# Get model folder and name from previous step
MODEL_RUN_FOLDER=$(cat /tmp/mocu_model_folder_${CONFIG_NAME}.txt 2>/dev/null || echo "")
MOCU_MODEL_NAME=$(cat /tmp/mocu_model_name_${CONFIG_NAME}.txt 2>/dev/null || echo "")

if [ -z "$MODEL_RUN_FOLDER" ] || [ ! -d "$MODEL_RUN_FOLDER" ]; then
    echo "Error: MPNN model folder not found. Run step2_train_mpnn.sh first."
    exit 1
fi

export MOCU_MODEL_NAME="$MOCU_MODEL_NAME"

DATA_FOLDER="${PROJECT_ROOT}/data/${CONFIG_NAME}/dad/"
DAD_TRAJECTORY_FILE="${DATA_FOLDER}dad_trajectories_N${N}_K4_random.pth"

# Generate DAD data if missing
if [ ! -f "$DAD_TRAJECTORY_FILE" ]; then
    echo "Generating DAD trajectory data..."
    mkdir -p "$DATA_FOLDER"
    ABS_DAD_DATA_FOLDER=$(cd "$DATA_FOLDER" && pwd)
    
    cd "${PROJECT_ROOT}/scripts"
    python3 generate_dad_data.py \
        --N $N \
        --num-episodes 100 \
        --K 4 \
        --output-dir "$ABS_DAD_DATA_FOLDER"
fi

echo "Training DAD policy (Step 3/5)..."
echo "  Method: $DAD_METHOD"
echo "  Using MPNN predictor: $MOCU_MODEL_NAME"

cd "${PROJECT_ROOT}/scripts"
ABS_DAD_TRAJ_FILE=$(cd "$(dirname "$DAD_TRAJECTORY_FILE")" && pwd)/$(basename "$DAD_TRAJECTORY_FILE")

USE_PREDICTED_MOCU=""
if [ "$DAD_METHOD" = "reinforce" ]; then
    if [ -f "${MODEL_RUN_FOLDER}model.pth" ]; then
        USE_PREDICTED_MOCU="--use-predicted-mocu"
    fi
fi

python3 train_dad_policy.py \
    --data-path "$ABS_DAD_TRAJ_FILE" \
    --method "$DAD_METHOD" \
    --name "dad_policy_N${N}" \
    --epochs 100 \
    --batch-size 64 \
    --output-dir "$MODEL_RUN_FOLDER" \
    $USE_PREDICTED_MOCU

echo "✓ DAD policy trained: ${MODEL_RUN_FOLDER}dad_policy_N${N}.pth"
echo "${MODEL_RUN_FOLDER}dad_policy_N${N}.pth" > /tmp/dad_policy_path_${CONFIG_NAME}.txt

