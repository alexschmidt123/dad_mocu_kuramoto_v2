#!/bin/bash
# Step 4: Generate DAD training data (before any policy training)

set -e

CONFIG_FILE=$1
if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: $0 <config_file>"
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
BASE_CONFIG_NAME=$(echo "$CONFIG_NAME" | sed 's/_K[0-9]*$//')
N=$(grep "^N:" "$CONFIG_FILE" | awk '{print $2}')

MOCU_MODEL_NAME=$(cat /tmp/mocu_model_name_${CONFIG_NAME}.txt 2>/dev/null || echo "")
MPNN_MODEL_FOLDER=$(cat /tmp/mocu_model_folder_${CONFIG_NAME}.txt 2>/dev/null || echo "")

if [ -z "$MPNN_MODEL_FOLDER" ] || [ ! -d "$MPNN_MODEL_FOLDER" ]; then
    echo "Error: MPNN model folder not found. Run step2_train_mpnn.sh first."
    exit 1
fi

# Determine output directories (experiment-local or legacy)
if [ -n "$EXP_DAD_DATA_DIR" ]; then
    DATA_FOLDER="$EXP_DAD_DATA_DIR"
else
    DATA_FOLDER="${PROJECT_ROOT}/data/${BASE_CONFIG_NAME}"
fi
mkdir -p "$DATA_FOLDER"

NUM_EPISODES=$(grep -A 3 "^dad_data:" "$CONFIG_FILE" | grep "  num_episodes:" | awk '{print $2}' || echo "1000")
K=$(grep -A 3 "^dad_data:" "$CONFIG_FILE" | grep "  K:" | awk '{print $2}' || echo "4")
USE_PRECOMPUTED_MOCU=$(grep -A 3 "^dad_data:" "$CONFIG_FILE" | grep "  use_precomputed_mocu:" | awk '{print $2}' || echo "true")

[ -z "$NUM_EPISODES" ] && NUM_EPISODES=1000
[ -z "$K" ] && K=4
[ -z "$USE_PRECOMPUTED_MOCU" ] && USE_PRECOMPUTED_MOCU=true

DAD_TRAJECTORY_FILE="${DATA_FOLDER}/dad_trajectories_N${N}_K${K}_random.pth"

check_data_k_n() {
    local data_file="$1"
    local filename_n=$(echo "$data_file" | sed -n 's/.*dad_trajectories_N\([0-9]*\)_K\([0-9]*\).*/\1/p')
    local filename_k=$(echo "$data_file" | sed -n 's/.*dad_trajectories_N\([0-9]*\)_K\([0-9]*\).*/\2/p')
    local file_k=$(python3 -c "
import torch
try:
    data = torch.load('$data_file', map_location='cpu', weights_only=False)
    print(data.get('config', {}).get('K', ''))
except:
    print('')
" 2>/dev/null || echo "")
    local file_n=$(python3 -c "
import torch
try:
    data = torch.load('$data_file', map_location='cpu', weights_only=False)
    print(data.get('config', {}).get('N', ''))
except:
    print('')
" 2>/dev/null || echo "")
    local file_episodes=$(python3 -c "
import torch
try:
    data = torch.load('$data_file', map_location='cpu', weights_only=False)
    print(data.get('config', {}).get('num_episodes', ''))
except:
    print('')
" 2>/dev/null || echo "")

    local k_match=false
    local n_match=false
    local episodes_match=true

    local data_k="${file_k:-$filename_k}"
    local data_n="${file_n:-$filename_n}"

    if [ "$data_k" = "$K" ]; then k_match=true; fi
    if [ "$data_n" = "$N" ]; then n_match=true; fi

    if [ -n "$file_episodes" ]; then
        if [ "$file_episodes" = "$NUM_EPISODES" ]; then
            episodes_match=true
        else
            episodes_match=false
        fi
    fi

    if [ "$k_match" = true ] && [ "$n_match" = true ] && [ "$episodes_match" = true ]; then
        echo "true"
    else
        echo "false"
    fi
}

DATA_EXISTS=false
DATA_MATCHES=false
if [ -f "$DAD_TRAJECTORY_FILE" ]; then
    DATA_EXISTS=true
    if [ "$(check_data_k_n "$DAD_TRAJECTORY_FILE")" = "true" ]; then
        DATA_MATCHES=true
    fi
fi

if [ "$DATA_EXISTS" = true ] && [ "$DATA_MATCHES" = false ]; then
    echo "⚠ DAD data exists but doesn't match current config (N/K/episodes). Regenerating..."
    rm -f "$DAD_TRAJECTORY_FILE"
    DATA_EXISTS=false
    DATA_MATCHES=false
fi

if [ "$DATA_EXISTS" = true ] && [ "$DATA_MATCHES" = true ]; then
    echo "✓ DAD data already exists and matches config: $DAD_TRAJECTORY_FILE"
else
    echo "Generating DAD trajectory data..."
    echo "  Settings: num_episodes=$NUM_EPISODES, K=$K, use_precomputed_mocu=$USE_PRECOMPUTED_MOCU"
    cd "${PROJECT_ROOT}/scripts"
    ABS_DAD_DATA_FOLDER=$(cd "$DATA_FOLDER" && pwd)
    CMD="python3 generate_dad_data.py --N $N --num-episodes $NUM_EPISODES --K $K --output-dir $ABS_DAD_DATA_FOLDER"
    if [ "$USE_PRECOMPUTED_MOCU" = "true" ] && [ -n "$MOCU_MODEL_NAME" ] && [ -f "${MPNN_MODEL_FOLDER}model.pth" ]; then
        CMD="$CMD --use-mpnn-predictor --mpnn-model-name $MOCU_MODEL_NAME"
        echo "  Using MPNN predictor ($MOCU_MODEL_NAME) to pre-compute MOCU values"
    else
        echo "  Not using MPNN predictor (terminal MOCU computed during training if needed)"
    fi
    eval $CMD
fi

ABS_DAD_TRAJ_FILE=$(cd "$(dirname "$DAD_TRAJECTORY_FILE")" && pwd)/$(basename "$DAD_TRAJECTORY_FILE")
echo "$ABS_DAD_TRAJ_FILE" > /tmp/dad_traj_file_${CONFIG_NAME}.txt

echo "✓ DAD training data ready: $ABS_DAD_TRAJ_FILE"

