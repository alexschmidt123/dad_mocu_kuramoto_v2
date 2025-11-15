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
# Remove _K* suffix if present to get base config name (for folder structure)
BASE_CONFIG_NAME=$(echo "$CONFIG_NAME" | sed 's/_K[0-9]*$//')
N=$(grep "^N:" "$CONFIG_FILE" | awk '{print $2}')
DAD_METHOD=$(grep "dad_method:" "$CONFIG_FILE" | awk '{print $2}' | tr -d '"' || echo "reinforce")

# Get model folder and name from previous step (use CONFIG_NAME for tmp file lookup)
MODEL_FOLDER=$(cat /tmp/mocu_model_folder_${CONFIG_NAME}.txt 2>/dev/null || echo "")
MOCU_MODEL_NAME=$(cat /tmp/mocu_model_name_${CONFIG_NAME}.txt 2>/dev/null || echo "")

if [ -z "$MODEL_FOLDER" ] || [ ! -d "$MODEL_FOLDER" ]; then
    echo "Error: MPNN model folder not found. Run step2_train_mpnn.sh first."
    exit 1
fi

# Ensure config file path is absolute
ABS_CONFIG_FILE="$CONFIG_FILE"
if [[ "$ABS_CONFIG_FILE" != /* ]]; then
    ABS_CONFIG_FILE="${PROJECT_ROOT}/${ABS_CONFIG_FILE}"
fi

# Get K value for model naming
K=$(grep -A 3 "^dad_data:" "$ABS_CONFIG_FILE" | grep "  K:" | awk '{print $2}' || echo "4")
if [ -z "$K" ]; then
    K=4
fi

# Use BASE_CONFIG_NAME for data folder (all K values share same folder)
DATA_FOLDER="${PROJECT_ROOT}/data/${BASE_CONFIG_NAME}/"

export MOCU_MODEL_NAME="$MOCU_MODEL_NAME"

# Read DAD data generation settings from config file
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

# Define file paths
DAD_MODEL_FILE="${MODEL_FOLDER}dad_policy_N${N}_K${K}.pth"
DAD_BEST_MODEL_FILE="${MODEL_FOLDER}dad_policy_N${N}_K${K}_best.pth"
DAD_TRAJECTORY_FILE="${DATA_FOLDER}dad_trajectories_N${N}_K${K}_random.pth"

# Helper function to check if model has correct K and N
check_model_k_n() {
    local model_file="$1"
    local model_k=""
    local model_n=""
    
    # Extract from filename
    local filename_n=$(echo "$model_file" | sed -n 's/.*dad_policy_N\([0-9]*\)_K\([0-9]*\).*/\1/p')
    local filename_k=$(echo "$model_file" | sed -n 's/.*dad_policy_N\([0-9]*\)_K\([0-9]*\).*/\2/p')
    
    # Try to read from checkpoint
    local checkpoint_k=$(python3 -c "
import torch
try:
    checkpoint = torch.load('$model_file', map_location='cpu', weights_only=False)
    if 'config' in checkpoint and 'K' in checkpoint['config']:
        print(checkpoint['config']['K'])
    elif 'train_config' in checkpoint and 'K' in checkpoint['train_config']:
        print(checkpoint['train_config']['K'])
    else:
        print('')
except:
    print('')
" 2>/dev/null || echo "")
    
    local checkpoint_n=$(python3 -c "
import torch
try:
    checkpoint = torch.load('$model_file', map_location='cpu', weights_only=False)
    if 'config' in checkpoint and 'N' in checkpoint['config']:
        print(checkpoint['config']['N'])
    elif 'train_config' in checkpoint and 'N' in checkpoint['train_config']:
        print(checkpoint['train_config']['N'])
    else:
        print('')
except:
    print('')
" 2>/dev/null || echo "")
    
    # Determine K and N (prefer checkpoint, fallback to filename)
    if [ -n "$checkpoint_k" ]; then
        model_k="$checkpoint_k"
    elif [ -n "$filename_k" ]; then
        model_k="$filename_k"
    fi
    
    if [ -n "$checkpoint_n" ]; then
        model_n="$checkpoint_n"
    elif [ -n "$filename_n" ]; then
        model_n="$filename_n"
    fi
    
    # Check if matches
    if [ "$model_k" = "$K" ] && [ "$model_n" = "$N" ]; then
        echo "true"
    else
        echo "false"
    fi
}

# Helper function to check if data has correct K and N
check_data_k_n() {
    local data_file="$1"
    local data_k=""
    local data_n=""
    
    # Extract from filename
    local filename_n=$(echo "$data_file" | sed -n 's/.*dad_trajectories_N\([0-9]*\)_K\([0-9]*\).*/\1/p')
    local filename_k=$(echo "$data_file" | sed -n 's/.*dad_trajectories_N\([0-9]*\)_K\([0-9]*\).*/\2/p')
    
    # Try to read from data file
    local file_k=$(python3 -c "
import torch
try:
    data = torch.load('$data_file', map_location='cpu', weights_only=False)
    if 'config' in data and 'K' in data['config']:
        print(data['config']['K'])
    else:
        print('')
except:
    print('')
" 2>/dev/null || echo "")
    
    local file_n=$(python3 -c "
import torch
try:
    data = torch.load('$data_file', map_location='cpu', weights_only=False)
    if 'config' in data and 'N' in data['config']:
        print(data['config']['N'])
    else:
        print('')
except:
    print('')
" 2>/dev/null || echo "")
    
    # Determine K and N (prefer file metadata, fallback to filename)
    if [ -n "$file_k" ]; then
        data_k="$file_k"
    elif [ -n "$filename_k" ]; then
        data_k="$filename_k"
    fi
    
    if [ -n "$file_n" ]; then
        data_n="$file_n"
    elif [ -n "$filename_n" ]; then
        data_n="$filename_n"
    fi
    
    # Check if matches
    if [ "$data_k" = "$K" ] && [ "$data_n" = "$N" ]; then
        echo "true"
    else
        echo "false"
    fi
}

# Check model and data status
MODEL_EXISTS=false
MODEL_MATCHES=false
DATA_EXISTS=false
DATA_MATCHES=false

# Check model
if [ -f "$DAD_MODEL_FILE" ] || [ -f "$DAD_BEST_MODEL_FILE" ]; then
    MODEL_EXISTS=true
    MODEL_TO_CHECK="$DAD_MODEL_FILE"
    if [ -f "$DAD_BEST_MODEL_FILE" ]; then
        MODEL_TO_CHECK="$DAD_BEST_MODEL_FILE"
    fi
    if [ "$(check_model_k_n "$MODEL_TO_CHECK")" = "true" ]; then
        MODEL_MATCHES=true
    fi
fi

# Check data
if [ -f "$DAD_TRAJECTORY_FILE" ]; then
    DATA_EXISTS=true
    if [ "$(check_data_k_n "$DAD_TRAJECTORY_FILE")" = "true" ]; then
        DATA_MATCHES=true
    fi
fi

# Decision logic based on user requirements:
# 1. If both model AND data exist with same N and K → skip everything
# 2. If model exists but data is missing → delete model, regenerate data and train model
# 3. If data exists but model is missing → keep data, skip generation, train model only

if [ "$MODEL_MATCHES" = true ] && [ "$DATA_MATCHES" = true ]; then
    # Case 1: Both exist and match → skip everything
    echo "✓ DAD model already exists: $MODEL_TO_CHECK"
    echo "✓ DAD data already exists: $DAD_TRAJECTORY_FILE"
    echo "✓ Both have correct K=$K and N=$N (matches config)"
    echo "✓ Skipping DAD data generation and training"
    echo "$MODEL_TO_CHECK" > /tmp/dad_policy_path_${CONFIG_NAME}.txt
    exit 0
elif [ "$MODEL_EXISTS" = true ] && [ "$MODEL_MATCHES" = false ]; then
    # Model exists but doesn't match → delete it
    echo "⚠ DAD model exists but K or N mismatch"
    echo "  Deleting existing model(s) to retrain with K=$K, N=$N"
    rm -f "$DAD_MODEL_FILE" "$DAD_BEST_MODEL_FILE"
    MODEL_EXISTS=false
    MODEL_MATCHES=false
elif [ "$MODEL_EXISTS" = true ] && [ "$MODEL_MATCHES" = true ] && [ "$DATA_EXISTS" = false ]; then
    # Case 2: Model exists and matches but data missing → delete model, regenerate everything
    echo "⚠ DAD model exists with correct K=$K, N=$N, but data is missing"
    echo "  Deleting model and regenerating data (model will be retrained)"
    rm -f "$DAD_MODEL_FILE" "$DAD_BEST_MODEL_FILE"
    MODEL_EXISTS=false
    MODEL_MATCHES=false
fi

# Handle data
if [ "$DATA_EXISTS" = true ] && [ "$DATA_MATCHES" = false ]; then
    # Data exists but doesn't match → delete it
    echo "⚠ DAD data exists but K or N mismatch"
    echo "  Deleting existing data to regenerate with K=$K, N=$N"
    rm -f "$DAD_TRAJECTORY_FILE"
    DATA_EXISTS=false
    DATA_MATCHES=false
elif [ "$DATA_EXISTS" = true ] && [ "$DATA_MATCHES" = true ] && [ "$MODEL_EXISTS" = false ]; then
    # Case 3: Data exists and matches but model missing → keep data, skip generation
    echo "✓ DAD data already exists: $DAD_TRAJECTORY_FILE"
    echo "✓ Data has correct K=$K and N=$N (matches config)"
    echo "  Keeping existing data, skipping data generation"
    echo "  Will train model only"
fi

# Generate DAD data if missing or K/N mismatch
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

