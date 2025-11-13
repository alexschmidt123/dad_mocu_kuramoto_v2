#!/bin/bash
# Step 4: Generate DAD training data and train DAD policy (uses MPNN predictor for MOCU estimation)
# This runs AFTER baseline evaluation so DAD can use the same initial MOCU

set -e

CONFIG_FILE=$1
if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Resolve config file path (handle relative paths)
if [[ "$CONFIG_FILE" != /* ]]; then
    # Relative path - resolve from PROJECT_ROOT
    CONFIG_FILE="${PROJECT_ROOT}/${CONFIG_FILE}"
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
N=$(grep "^N:" "$CONFIG_FILE" | awk '{print $2}')
DAD_METHOD=$(grep "dad_method:" "$CONFIG_FILE" | awk '{print $2}' | tr -d '"' || echo "reinforce")

# Get model folder and name from previous step
MODEL_FOLDER=$(cat /tmp/mocu_model_folder_${CONFIG_NAME}.txt 2>/dev/null || echo "")
MOCU_MODEL_NAME=$(cat /tmp/mocu_model_name_${CONFIG_NAME}.txt 2>/dev/null || echo "")

if [ -z "$MODEL_FOLDER" ] || [ ! -d "$MODEL_FOLDER" ]; then
    echo "Error: MPNN model folder not found. Run step2_train_mpnn.sh first."
    exit 1
fi

# Get K value for model naming
K=$(grep -A 3 "^dad_data:" "$ABS_CONFIG_FILE" | grep "  K:" | awk '{print $2}' || echo "4")
if [ -z "$K" ]; then
    K=4
fi

# Check if DAD model already exists - skip training if so
# Model name includes K: dad_policy_N${N}_K${K}.pth
DAD_MODEL_FILE="${MODEL_FOLDER}dad_policy_N${N}_K${K}.pth"
if [ -f "$DAD_MODEL_FILE" ]; then
    echo "✓ DAD model already exists: $DAD_MODEL_FILE"
    echo "✓ Skipping DAD training (model detected)"
    echo "$DAD_MODEL_FILE" > /tmp/dad_policy_path_${CONFIG_NAME}.txt
    exit 0
fi

export MOCU_MODEL_NAME="$MOCU_MODEL_NAME"

DATA_FOLDER="${PROJECT_ROOT}/data/${CONFIG_NAME}/"

# Read DAD data generation settings from config file
# Ensure config file path is absolute
ABS_CONFIG_FILE="$CONFIG_FILE"
if [[ "$ABS_CONFIG_FILE" != /* ]]; then
    ABS_CONFIG_FILE="${PROJECT_ROOT}/${ABS_CONFIG_FILE}"
fi

NUM_EPISODES=$(grep -A 3 "^dad_data:" "$ABS_CONFIG_FILE" | grep "  num_episodes:" | awk '{print $2}' || echo "1000")
# K already read above, but ensure it's set
if [ -z "$K" ]; then
    K=$(grep -A 3 "^dad_data:" "$ABS_CONFIG_FILE" | grep "  K:" | awk '{print $2}' || echo "4")
fi
USE_PRECOMPUTED_MOCU=$(grep -A 3 "^dad_data:" "$ABS_CONFIG_FILE" | grep "  use_precomputed_mocu:" | awk '{print $2}' || echo "true")

# Validate values are not empty
if [ -z "$NUM_EPISODES" ]; then
    NUM_EPISODES=1000
fi
if [ -z "$K" ]; then
    K=4
fi
if [ -z "$USE_PRECOMPUTED_MOCU" ]; then
    USE_PRECOMPUTED_MOCU=true
fi

# Use dynamic filename based on K from config
DAD_TRAJECTORY_FILE="${DATA_FOLDER}dad_trajectories_N${N}_K${K}_random.pth"

# Check if DAD model exists and verify it was trained with correct K
if [ -f "$DAD_MODEL_FILE" ]; then
    # Check if model was trained with correct K by loading checkpoint
    MODEL_K=$(python3 -c "
import torch
try:
    checkpoint = torch.load('$DAD_MODEL_FILE', map_location='cpu', weights_only=False)
    if 'config' in checkpoint and 'K' in checkpoint['config']:
        print(checkpoint['config']['K'])
    elif 'train_config' in checkpoint and 'K' in checkpoint['train_config']:
        print(checkpoint['train_config']['K'])
    else:
        print('unknown')
except:
    print('error')
" 2>/dev/null || echo "unknown")
    
    if [ "$MODEL_K" = "$K" ]; then
        echo "✓ DAD model already exists: $DAD_MODEL_FILE"
        echo "✓ Model was trained with K=$K design steps ($((K+1)) total steps: 0-$K) (matches config)"
        echo "✓ Skipping DAD training (model detected with correct K)"
        echo "$DAD_MODEL_FILE" > /tmp/dad_policy_path_${CONFIG_NAME}.txt
        exit 0
    elif [ "$MODEL_K" != "unknown" ] && [ "$MODEL_K" != "error" ]; then
        echo "⚠ DAD model exists but was trained with K=$MODEL_K (config requires K=$K)"
        echo "  Will regenerate data and retrain with K=$K"
        # Remove old model to force retraining
        rm -f "$DAD_MODEL_FILE"
    fi
fi

# Generate DAD data if missing or K mismatch
if [ ! -f "$DAD_TRAJECTORY_FILE" ]; then
    echo "Generating DAD trajectory data..."
    mkdir -p "$DATA_FOLDER"
    ABS_DAD_DATA_FOLDER=$(cd "$DATA_FOLDER" && pwd)
    
    echo "  DAD data settings: num_episodes=$NUM_EPISODES, K=$K, use_precomputed_mocu=$USE_PRECOMPUTED_MOCU"
    
    cd "${PROJECT_ROOT}/scripts"
    
    # Pre-compute MOCU using MPNN predictor if available and configured
    CMD="python3 generate_dad_data.py --N $N --num-episodes $NUM_EPISODES --K $K --output-dir $ABS_DAD_DATA_FOLDER"
    if [ "$USE_PRECOMPUTED_MOCU" = "true" ] && [ -n "$MOCU_MODEL_NAME" ] && [ -f "${MODEL_FOLDER}model.pth" ]; then
        CMD="$CMD --use-mpnn-predictor --mpnn-model-name $MOCU_MODEL_NAME"
        echo "  Using MPNN predictor to pre-compute MOCU values"
    else
        echo "  Not using MPNN predictor (MOCU will be computed during training if needed)"
    fi
    
    eval $CMD
else
    # Data file exists - verify it has correct K
    echo "✓ Found existing DAD data: $DAD_TRAJECTORY_FILE"
    echo "✓ DAD data exists for K=$K design steps ($((K+1)) total steps: 0-$K)"
    # Check K value in saved data
    DATA_K=$(python3 -c "
import torch
try:
    data = torch.load('$DAD_TRAJECTORY_FILE', map_location='cpu', weights_only=False)
    if 'config' in data and 'K' in data['config']:
        print(data['config']['K'])
    else:
        print('unknown')
except:
    print('error')
" 2>/dev/null || echo "unknown")
    
    if [ "$DATA_K" != "$K" ] && [ "$DATA_K" != "unknown" ] && [ "$DATA_K" != "error" ]; then
        echo "⚠ Existing DAD data has K=$DATA_K (config requires K=$K)"
        echo "  Regenerating data with K=$K..."
        rm -f "$DAD_TRAJECTORY_FILE"
        mkdir -p "$DATA_FOLDER"
        ABS_DAD_DATA_FOLDER=$(cd "$DATA_FOLDER" && pwd)
        echo "  DAD data settings: num_episodes=$NUM_EPISODES, K=$K, use_precomputed_mocu=$USE_PRECOMPUTED_MOCU"
        cd "${PROJECT_ROOT}/scripts"
        CMD="python3 generate_dad_data.py --N $N --num-episodes $NUM_EPISODES --K $K --output-dir $ABS_DAD_DATA_FOLDER"
        if [ "$USE_PRECOMPUTED_MOCU" = "true" ] && [ -n "$MOCU_MODEL_NAME" ] && [ -f "${MODEL_FOLDER}model.pth" ]; then
            CMD="$CMD --use-mpnn-predictor --mpnn-model-name $MOCU_MODEL_NAME"
            echo "  Using MPNN predictor to pre-compute MOCU values"
        fi
        eval $CMD
    else
        echo "  Data has correct K=$K (or K not stored in data)"
    fi
fi

echo "Generating DAD training data and training DAD policy (Step 4/6)..."
echo "  Method: $DAD_METHOD"
echo "  Using MPNN predictor: $MOCU_MODEL_NAME"
echo "  K value: $K design steps ($((K+1)) total steps: 0-$K)"

cd "${PROJECT_ROOT}/scripts"
ABS_DAD_TRAJ_FILE=$(cd "$(dirname "$DAD_TRAJECTORY_FILE")" && pwd)/$(basename "$DAD_TRAJECTORY_FILE")

USE_PREDICTED_MOCU=""
if [ "$DAD_METHOD" = "reinforce" ]; then
    if [ -f "${MODEL_FOLDER}model.pth" ]; then
        USE_PREDICTED_MOCU="--use-predicted-mocu"
    fi
fi

python3 train_dad_policy.py \
    --data-path "$ABS_DAD_TRAJ_FILE" \
    --method "$DAD_METHOD" \
    --name "dad_policy_N${N}_K${K}" \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.0001 \
    --hidden-dim 256 \
    --encoding-dim 16 \
    --use-critic \
    --output-dir "$MODEL_FOLDER" \
    $USE_PREDICTED_MOCU

echo "✓ DAD policy trained: ${DAD_MODEL_FILE}"
echo "✓ DAD model trained for K=$K design steps ($((K+1)) total steps: 0-$K)"

# Prefer best checkpoint if it exists (better performance)
DAD_BEST_MODEL_FILE="${MODEL_FOLDER}dad_policy_N${N}_K${K}_best.pth"
if [ -f "$DAD_BEST_MODEL_FILE" ]; then
    echo "✓ Using best checkpoint: $DAD_BEST_MODEL_FILE"
    echo "$DAD_BEST_MODEL_FILE" > /tmp/dad_policy_path_${CONFIG_NAME}.txt
else
    echo "✓ Using final checkpoint: $DAD_MODEL_FILE"
    echo "$DAD_MODEL_FILE" > /tmp/dad_policy_path_${CONFIG_NAME}.txt
fi

