#!/bin/bash
# Step 4: Generate DAD training data and train DAD policy (uses MPNN predictor for MOCU estimation)
# This runs AFTER baseline evaluation so DAD can use the same initial MOCU
# 
# DAD Methods:
#   - dad_mocu: DAD-MOCU (no critic, simple baseline, uses per-step rewards)
#   - idad_mocu: iDAD-MOCU (with critic from scratch, uses per-step rewards)
# Both methods are trained automatically to enable comparison in final plots

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

# Extract DAD methods from methods list (DAD_MOCU and/or IDAD_MOCU)
# Use Python to parse YAML properly
METHODS_TO_TRAIN=$(python3 << PYEOF
import sys
import yaml

try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    
    methods = []
    if 'experiment' in config and 'methods' in config['experiment']:
        method_list = config['experiment']['methods']
        if isinstance(method_list, list):
            # Check for DAD methods
            if 'DAD_MOCU' in method_list:
                methods.append('dad_mocu')
            if 'IDAD_MOCU' in method_list:
                methods.append('idad_mocu')
    
    # Default to both if neither specified
    if not methods:
        methods = ['dad_mocu', 'idad_mocu']
    
    print(' '.join(methods))
except Exception as e:
    # Fallback: check both
    print('dad_mocu idad_mocu')
PYEOF
)

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

# Define file paths (base names, method-specific names will be created in training loop)
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

# Helper function to check if data has correct K, N, and num_episodes
check_data_k_n() {
    local data_file="$1"
    local data_k=""
    local data_n=""
    local data_episodes=""
    
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
    
    local file_episodes=$(python3 -c "
import torch
try:
    data = torch.load('$data_file', map_location='cpu', weights_only=False)
    if 'config' in data and 'num_episodes' in data['config']:
        print(data['config']['num_episodes'])
    else:
        print('')
except:
    print('')
" 2>/dev/null || echo "")
    
    # Determine K, N, and num_episodes (prefer file metadata, fallback to filename)
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
    
    if [ -n "$file_episodes" ]; then
        data_episodes="$file_episodes"
    fi
    
    # Check if matches (K, N, and num_episodes if available)
    local k_match=false
    local n_match=false
    local episodes_match=true  # Default to true if not stored
    
    if [ "$data_k" = "$K" ]; then
        k_match=true
    fi
    
    if [ "$data_n" = "$N" ]; then
        n_match=true
    fi
    
    # Check num_episodes if it's stored in the file
    if [ -n "$data_episodes" ]; then
        if [ "$data_episodes" = "$NUM_EPISODES" ]; then
            episodes_match=true
        else
            episodes_match=false
        fi
    fi
    
    # All must match
    if [ "$k_match" = true ] && [ "$n_match" = true ] && [ "$episodes_match" = true ]; then
        echo "true"
    else
        echo "false"
    fi
}

# Check model and data status
DATA_EXISTS=false
DATA_MATCHES=false

# Check data
if [ -f "$DAD_TRAJECTORY_FILE" ]; then
    DATA_EXISTS=true
    if [ "$(check_data_k_n "$DAD_TRAJECTORY_FILE")" = "true" ]; then
        DATA_MATCHES=true
    fi
fi

# Note: Model checking is now done per-method in the training loop below
# This allows us to check each method's model separately

# Handle data
if [ "$DATA_EXISTS" = true ] && [ "$DATA_MATCHES" = false ]; then
    # Data exists but doesn't match → delete it
    echo "⚠ DAD data exists but K, N, or num_episodes mismatch"
    echo "  Deleting existing data to regenerate with K=$K, N=$N, num_episodes=$NUM_EPISODES"
    rm -f "$DAD_TRAJECTORY_FILE"
    DATA_EXISTS=false
    DATA_MATCHES=false
elif [ "$DATA_EXISTS" = true ] && [ "$DATA_MATCHES" = true ]; then
    # Case: Data exists and matches → keep data, skip generation
    echo "✓ DAD data already exists: $DAD_TRAJECTORY_FILE"
    echo "✓ Data has correct K=$K and N=$N (matches config)"
    echo "  Keeping existing data, skipping data generation"
    echo "  Will train models only"
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

echo "Generating DAD training data and training DAD policies (Step 4/6)..."
echo "  Methods: dad_mocu and idad_mocu (both will be trained)"
echo "  Using MPNN predictor: $MOCU_MODEL_NAME"
echo "  K value: $K design steps ($((K+1)) total steps: 0-$K)"

cd "${PROJECT_ROOT}/scripts"
ABS_DAD_TRAJ_FILE=$(cd "$(dirname "$DAD_TRAJECTORY_FILE")" && pwd)/$(basename "$DAD_TRAJECTORY_FILE")

# Both dad_mocu and idad_mocu need MPNN predictor for per-step rewards
USE_PREDICTED_MOCU=""
if [ -f "${MODEL_FOLDER}model.pth" ]; then
    USE_PREDICTED_MOCU="--use-predicted-mocu"
fi

# Parse methods to train from config (DAD_MOCU and/or IDAD_MOCU in methods list)
# Convert to array
METHODS_TO_TRAIN_ARRAY=($METHODS_TO_TRAIN)

if [ ${#METHODS_TO_TRAIN_ARRAY[@]} -eq 0 ]; then
    echo "Warning: No DAD methods found in methods list. Defaulting to both."
    METHODS_TO_TRAIN_ARRAY=("dad_mocu" "idad_mocu")
fi

echo "Will train DAD methods: ${METHODS_TO_TRAIN_ARRAY[*]}"
echo "  (Based on methods list in config: DAD_MOCU and/or IDAD_MOCU)"

for METHOD in "${METHODS_TO_TRAIN_ARRAY[@]}"; do
    echo ""
    echo "=========================================="
    echo "Training $METHOD method"
    echo "=========================================="
    
    # Model name includes method for differentiation
    MODEL_NAME="dad_policy_N${N}_K${K}_${METHOD}"
    METHOD_MODEL_FILE="${MODEL_FOLDER}${MODEL_NAME}.pth"
    METHOD_BEST_MODEL_FILE="${MODEL_FOLDER}${MODEL_NAME}_best.pth"
    
    # Check if this method's model already exists
    METHOD_MODEL_EXISTS=false
    METHOD_MODEL_MATCHES=false
    if [ -f "$METHOD_MODEL_FILE" ] || [ -f "$METHOD_BEST_MODEL_FILE" ]; then
        METHOD_MODEL_EXISTS=true
        METHOD_TO_CHECK="$METHOD_BEST_MODEL_FILE"
        if [ ! -f "$METHOD_TO_CHECK" ]; then
            METHOD_TO_CHECK="$METHOD_MODEL_FILE"
        fi
        if [ "$(check_model_k_n "$METHOD_TO_CHECK")" = "true" ]; then
            METHOD_MODEL_MATCHES=true
        fi
    fi
    
    if [ "$METHOD_MODEL_MATCHES" = true ] && [ "$DATA_MATCHES" = true ]; then
        echo "✓ $METHOD model already exists and matches: $METHOD_TO_CHECK"
        echo "✓ Skipping training for $METHOD"
        if [ -f "$METHOD_BEST_MODEL_FILE" ]; then
            echo "$METHOD_BEST_MODEL_FILE" > /tmp/${METHOD}_policy_path_${CONFIG_NAME}.txt
        else
            echo "$METHOD_MODEL_FILE" > /tmp/${METHOD}_policy_path_${CONFIG_NAME}.txt
        fi
        continue
    fi
    
    # Build training command for this method
    TRAIN_CMD="python3 train_dad_policy.py \
        --data-path \"$ABS_DAD_TRAJ_FILE\" \
        --method \"$METHOD\" \
        --name \"$MODEL_NAME\" \
        --epochs 100 \
        --batch-size 64 \
        --lr 0.000005 \
        --hidden-dim 256 \
        --encoding-dim 16 \
        --output-dir \"$MODEL_FOLDER\""
    
    # Add --use-predicted-mocu if needed (for per-step rewards)
    if [ -n "$USE_PREDICTED_MOCU" ]; then
        TRAIN_CMD="$TRAIN_CMD $USE_PREDICTED_MOCU"
    fi
    
    # Execute training command
    eval $TRAIN_CMD
    
    echo "✓ $METHOD policy trained: ${METHOD_MODEL_FILE}"
    echo "✓ $METHOD model trained for K=$K design steps ($((K+1)) total steps: 0-$K)"
    
    # Save policy path for this method
    if [ -f "$METHOD_BEST_MODEL_FILE" ]; then
        echo "✓ Using best checkpoint: $METHOD_BEST_MODEL_FILE"
        echo "$METHOD_BEST_MODEL_FILE" > /tmp/${METHOD}_policy_path_${CONFIG_NAME}.txt
    else
        echo "✓ Using final checkpoint: $METHOD_MODEL_FILE"
        echo "$METHOD_MODEL_FILE" > /tmp/${METHOD}_policy_path_${CONFIG_NAME}.txt
    fi
done

echo ""
echo "✓ All DAD methods trained"

