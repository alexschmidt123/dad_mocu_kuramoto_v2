#!/bin/bash
# Step 5: Train DAD/IDAD policies (uses existing trajectories)

set -e

CONFIG_FILE=$1
if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

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

DAD_TRAJ_FILE=$(cat /tmp/dad_traj_file_${CONFIG_NAME}.txt 2>/dev/null || echo "")
if [ -z "$DAD_TRAJ_FILE" ] || [ ! -f "$DAD_TRAJ_FILE" ]; then
    echo "Error: DAD trajectory file not found. Run step4_generate_dad_data.sh first."
    exit 1
fi

MPNN_MODEL_FOLDER=$(cat /tmp/mocu_model_folder_${CONFIG_NAME}.txt 2>/dev/null || echo "")
MOCU_MODEL_NAME=$(cat /tmp/mocu_model_name_${CONFIG_NAME}.txt 2>/dev/null || echo "")

if [ -z "$MPNN_MODEL_FOLDER" ] || [ ! -d "$MPNN_MODEL_FOLDER" ]; then
    echo "Error: MPNN model folder not found. Run step2_train_mpnn.sh first."
    exit 1
fi

if [ -n "$EXP_DAD_MODELS_DIR" ]; then
    POLICY_OUTPUT_DIR="$EXP_DAD_MODELS_DIR"
else
    POLICY_OUTPUT_DIR="${PROJECT_ROOT}/models/${BASE_CONFIG_NAME}"
fi
mkdir -p "$POLICY_OUTPUT_DIR"

export MOCU_MODEL_NAME="$MOCU_MODEL_NAME"

NUM_EPISODES=$(grep -A 3 "^dad_data:" "$CONFIG_FILE" | grep "  num_episodes:" | awk '{print $2}' || echo "1000")
K=$(grep -A 3 "^dad_data:" "$CONFIG_FILE" | grep "  K:" | awk '{print $2}' || echo "4")
[ -z "$NUM_EPISODES" ] && NUM_EPISODES=1000
[ -z "$K" ] && K=4

# Determine which methods to train
METHODS_TO_TRAIN=$(python3 << PYEOF
import yaml
try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    methods = []
    if 'experiment' in config and 'methods' in config['experiment']:
        method_list = config['experiment']['methods']
        if isinstance(method_list, list):
            if 'DAD_MOCU' in method_list:
                methods.append('dad_mocu')
            if 'IDAD_MOCU' in method_list:
                methods.append('idad_mocu')
    if not methods:
        methods = ['dad_mocu', 'idad_mocu']
    print(' '.join(methods))
except Exception:
    print('dad_mocu idad_mocu')
PYEOF
)

if [ -z "$METHODS_TO_TRAIN" ]; then
    METHODS_TO_TRAIN="dad_mocu idad_mocu"
fi

# Helper functions (copied from original script)
check_model_k_n() {
    local model_file="$1"
    local filename_n=$(echo "$model_file" | sed -n 's/.*dad_policy_N\([0-9]*\)_K\([0-9]*\).*/\1/p')
    local filename_k=$(echo "$model_file" | sed -n 's/.*dad_policy_N\([0-9]*\)_K\([0-9]*\).*/\2/p')
    local checkpoint_k=$(python3 -c "
import torch
try:
    checkpoint = torch.load('$model_file', map_location='cpu', weights_only=False)
    cfg = checkpoint.get('config', {})
    train_cfg = checkpoint.get('train_config', {})
    print(cfg.get('K') or train_cfg.get('K', ''))
except:
    print('')
" 2>/dev/null || echo "")
    local checkpoint_n=$(python3 -c "
import torch
try:
    checkpoint = torch.load('$model_file', map_location='cpu', weights_only=False)
    cfg = checkpoint.get('config', {})
    train_cfg = checkpoint.get('train_config', {})
    print(cfg.get('N') or train_cfg.get('N', ''))
except:
    print('')
" 2>/dev/null || echo "")

    local model_k="${checkpoint_k:-$filename_k}"
    local model_n="${checkpoint_n:-$filename_n}"

    if [ "$model_k" = "$K" ] && [ "$model_n" = "$N" ]; then
        echo "true"
    else
        echo "false"
    fi
}

DATA_MATCHES=$(python3 << PYEOF
import torch
try:
    data = torch.load('$DAD_TRAJ_FILE', map_location='cpu', weights_only=False)
    cfg = data.get('config', {})
    ok = (cfg.get('N') == $N) and (cfg.get('K') == $K)
    if 'num_episodes' in cfg:
        ok = ok and (cfg.get('num_episodes') == $NUM_EPISODES)
    print('true' if ok else 'false')
except:
    print('false')
PYEOF
)

USE_PREDICTED_MOCU=""
if [ -f "${MPNN_MODEL_FOLDER}model.pth" ]; then
    USE_PREDICTED_MOCU="--use-predicted-mocu"
fi

ABS_DAD_TRAJ_FILE="$DAD_TRAJ_FILE"

echo "Training DAD policies (Step 5/6)..."
echo "  Methods: $METHODS_TO_TRAIN"
echo "  Data: $ABS_DAD_TRAJ_FILE"
echo "  Output dir: $POLICY_OUTPUT_DIR"

METHODS_TO_TRAIN_ARRAY=($METHODS_TO_TRAIN)
if [ ${#METHODS_TO_TRAIN_ARRAY[@]} -eq 0 ]; then
    METHODS_TO_TRAIN_ARRAY=("dad_mocu" "idad_mocu")
fi

for METHOD in "${METHODS_TO_TRAIN_ARRAY[@]}"; do
    echo ""
    echo "=========================================="
    echo "Training $METHOD method"
    echo "=========================================="

    MODEL_NAME="dad_policy_N${N}_K${K}_${METHOD}"
    METHOD_MODEL_FILE="${POLICY_OUTPUT_DIR}/${MODEL_NAME}.pth"
    METHOD_BEST_MODEL_FILE="${POLICY_OUTPUT_DIR}/${MODEL_NAME}_best.pth"

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
        echo "✓ $METHOD model already exists and matches config: $METHOD_TO_CHECK"
        echo "✓ Skipping training for $METHOD"
        if [ "$METHOD" = "dad_mocu" ]; then
            echo "$METHOD_TO_CHECK" > /tmp/dad_mocu_policy_path_${CONFIG_NAME}.txt
        else
            echo "$METHOD_TO_CHECK" > /tmp/idad_mocu_policy_path_${CONFIG_NAME}.txt
        fi
        continue
    fi

    TRAIN_CMD="python3 train_dad_policy.py \
        --data-path \"$ABS_DAD_TRAJ_FILE\" \
        --method \"$METHOD\" \
        --name \"$MODEL_NAME\" \
        --epochs 100 \
        --batch-size 64 \
        --lr 0.0000005 \
        --hidden-dim 256 \
        --encoding-dim 16 \
        --output-dir \"$POLICY_OUTPUT_DIR\" \
        $USE_PREDICTED_MOCU"

    eval $TRAIN_CMD

    if [ "$METHOD" = "dad_mocu" ]; then
        if [ -f "$METHOD_BEST_MODEL_FILE" ]; then
            echo "$METHOD_BEST_MODEL_FILE" > /tmp/dad_mocu_policy_path_${CONFIG_NAME}.txt
        else
            echo "$METHOD_MODEL_FILE" > /tmp/dad_mocu_policy_path_${CONFIG_NAME}.txt
        fi
    else
        if [ -f "$METHOD_BEST_MODEL_FILE" ]; then
            echo "$METHOD_BEST_MODEL_FILE" > /tmp/idad_mocu_policy_path_${CONFIG_NAME}.txt
        else
            echo "$METHOD_MODEL_FILE" > /tmp/idad_mocu_policy_path_${CONFIG_NAME}.txt
        fi
    fi
done

echo ""
echo "✓ DAD policy training complete"

