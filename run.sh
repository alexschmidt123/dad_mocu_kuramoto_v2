#!/bin/bash
# Main script to run complete MOCU-OED experiment workflow
# Usage: bash run.sh configs/N5_config.yaml

set -e  # Exit on error

# Set Python path to project root for imports
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if config file is provided
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: No configuration file provided${NC}"
    echo "Usage: bash run.sh <config_file>"
    echo "Example: bash run.sh configs/N5_config.yaml"
    exit 1
fi

CONFIG_FILE=$1

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Configuration file '$CONFIG_FILE' not found${NC}"
    exit 1
fi

# Extract config name from path (e.g., "N5_config" from "configs/N5_config.yaml")
CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)

# Generate timestamp for this run (format: DDMMYYYY_HHMMSS)
TIMESTAMP=$(date +"%d%m%Y_%H%M%S")

# New folder structure:
# - data/{config_name}/          (reusable, no timestamp)
# - models/{config_name}/{timestamp}/  (timestamped runs)
# - results/{config_name}/{timestamp}/ (timestamped runs)

# Base folders (in project root)
DATA_FOLDER="${PROJECT_ROOT}/data/${CONFIG_NAME}/"
MODEL_RUN_FOLDER="${PROJECT_ROOT}/models/${CONFIG_NAME}/${TIMESTAMP}/"
RESULT_RUN_FOLDER="${PROJECT_ROOT}/results/${CONFIG_NAME}/${TIMESTAMP}/"

# Create folders
mkdir -p "$DATA_FOLDER"
mkdir -p "$MODEL_RUN_FOLDER"
mkdir -p "$RESULT_RUN_FOLDER"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}MOCU-OED Experiment Workflow${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Config file: ${YELLOW}$CONFIG_FILE${NC}"
echo -e "Config name: ${YELLOW}$CONFIG_NAME${NC}"
echo -e "Run timestamp: ${YELLOW}$TIMESTAMP${NC}"
echo -e "Data folder: ${YELLOW}$DATA_FOLDER${NC}"
echo -e "Models folder: ${YELLOW}$MODEL_RUN_FOLDER${NC}"
echo -e "Results folder: ${YELLOW}$RESULT_RUN_FOLDER${NC}"
if [ -n "$N_GLOBAL_UPDATED" ]; then
    echo -e "${BLUE}[Re-execution after N_global update]${NC}"
fi
echo ""

# Parse YAML config file
N=$(grep "^N:" $CONFIG_FILE | awk '{print $2}')
N_GLOBAL=$(grep "^N_global:" $CONFIG_FILE | awk '{print $2}')
TRAINED_MODEL_NAME=$(grep "model_name:" $CONFIG_FILE | awk '{print $2}' | tr -d '"')
SAMPLES=$(grep "samples_per_type:" $CONFIG_FILE | awk '{print $2}')
TRAIN_SIZE=$(grep "train_size:" $CONFIG_FILE | awk '{print $2}')
K_MAX=$(grep -A 3 "^dataset:" $CONFIG_FILE | grep "K_max:" | awk '{print $2}')
EPOCHS=$(grep "epochs:" $CONFIG_FILE | awk '{print $2}')
CONSTRAIN_WEIGHT=$(grep "constrain_weight:" $CONFIG_FILE | awk '{print $2}')
SAVE_JSON=$(grep "save_json:" $CONFIG_FILE | awk '{print $2}')

# Parse methods list from config
METHODS=$(grep -A 20 "^  methods:" $CONFIG_FILE | grep '    - "' | sed 's/.*"\(.*\)".*/\1/' | grep -v '^#' | tr '\n' ',' | sed 's/,$//')

echo -e "${BLUE}Experiment Configuration:${NC}"
echo "  System size (N): $N"
echo "  N_global: $N_GLOBAL"
echo "  Samples per type: $SAMPLES"
echo "  Training set size: $TRAIN_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Methods to evaluate: $METHODS"
echo ""

# Step 0: Check and update N_global in CUDA code
echo -e "${GREEN}[Step 0/5]${NC} Checking CUDA N_global configuration..."
CUDA_FILE="src/core/mocu_cuda.py"
CURRENT_N_GLOBAL=$(grep "#define N_global" $CUDA_FILE | awk '{print $3}')

if [ "$CURRENT_N_GLOBAL" != "$N_GLOBAL" ]; then
    if [ -z "$N_GLOBAL_UPDATED" ]; then
        echo -e "${YELLOW}Updating N_global from $CURRENT_N_GLOBAL to $N_GLOBAL...${NC}"
        sed -i.bak "s/#define N_global.*/#define N_global $N_GLOBAL/" $CUDA_FILE
        echo -e "${GREEN}✓${NC} N_global updated successfully"
        echo -e "${BLUE}Re-executing script to apply changes...${NC}"
        echo ""
        export N_GLOBAL_UPDATED=1
        exec bash "$0" "$@"
    else
        echo -e "${RED}Error: N_global mismatch after update. Please check $CUDA_FILE${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓${NC} N_global is correctly set to $N_GLOBAL"
fi

# Step 1: Generate dataset (reusable - check if exists)
echo ""
echo -e "${GREEN}[Step 1/5]${NC} Checking dataset..."

EXISTING_TRAIN_FILE=$(find "${DATA_FOLDER}" -name "*_${N}o_train.pth" -type f 2>/dev/null | head -1)

if [ -n "$EXISTING_TRAIN_FILE" ]; then
    echo -e "${GREEN}✓${NC} Dataset already exists: $EXISTING_TRAIN_FILE"
    echo "  Reusing existing data (same config = reusable data)"
    TRAIN_FILE="$EXISTING_TRAIN_FILE"
else
    echo "  Generating dataset (this may take time)..."
    echo -e "${YELLOW}  Note: Actual file size may differ from config due to sync filtering${NC}"
    
    cd scripts
    
    ABS_DATA_FOLDER=$(cd "${DATA_FOLDER}" && pwd)
    
    CMD="python generate_mocu_data.py --N $N --samples_per_type $SAMPLES --train_size $TRAIN_SIZE --K_max $K_MAX --output_dir $ABS_DATA_FOLDER"
    if [ "$SAVE_JSON" = "true" ]; then
        CMD="$CMD --save_json"
    fi
    
    eval $CMD
    
    cd "$PROJECT_ROOT"
    
    TRAIN_FILE=$(find "${DATA_FOLDER}" -name "*_${N}o_train.pth" -type f 2>/dev/null | head -1)
    
    if [ -z "$TRAIN_FILE" ]; then
        echo -e "${RED}Error: No training file found in ${DATA_FOLDER}${NC}"
        exit 1
    fi
    
    ACTUAL_SIZE=$(basename "$TRAIN_FILE" | sed "s/_${N}o_train.pth//" )
    
    echo -e "${GREEN}✓${NC} Dataset generated: $TRAIN_FILE"
    if [ "$ACTUAL_SIZE" != "$TRAIN_SIZE" ]; then
        echo -e "${YELLOW}  Note: Generated ${ACTUAL_SIZE} samples (config requested ${TRAIN_SIZE})${NC}"
    fi
fi

# Step 2: Train model (save to timestamped folder)
echo ""
echo -e "${GREEN}[Step 2/5]${NC} Training MPNN predictor..."
echo "  This may take 1-2 hours..."

ABS_TRAIN_FILE=$(cd "$(dirname "$TRAIN_FILE")" && pwd)/$(basename "$TRAIN_FILE")
ABS_MODEL_RUN_FOLDER=$(cd "$MODEL_RUN_FOLDER" && pwd)

cd scripts
python train_mocu_predictor.py \
    --name "${CONFIG_NAME}_${TIMESTAMP}" \
    --data_path "$ABS_TRAIN_FILE" \
    --EPOCH $EPOCHS \
    --Constrain_weight $CONSTRAIN_WEIGHT \
    --output_dir "$(dirname "$ABS_MODEL_RUN_FOLDER")/"
cd "$PROJECT_ROOT"

echo -e "${GREEN}✓${NC} Model trained: ${MODEL_RUN_FOLDER}model.pth"

# Export experiment ID (for DAD training and evaluation)
export MOCU_MODEL_NAME="${CONFIG_NAME}_${TIMESTAMP}"

# Step 2.5: Check if DAD is in methods and train if needed
echo ""
if echo "$METHODS" | grep -q "DAD"; then
    echo -e "${GREEN}[Step 2.5/5]${NC} Training DAD policy (DAD detected in methods)..."
    
    # Step 2.5a: Generate DAD trajectory data (AFTER MPNN is trained)
    echo "  [2.5a] Generating DAD trajectory data..."
    echo "  Note: DAD data generation uses random actions (REINFORCE doesn't need expert labels)"
    echo "        MOCU is computed during training using the trained MPNN predictor"
    
    mkdir -p "${DATA_FOLDER}dad/"
    ABS_DAD_DATA_FOLDER=$(cd "${DATA_FOLDER}dad/" && pwd)
    
    # Generate DAD training data (random expert - MOCU computed during training)
    cd scripts
    python generate_dad_data.py \
        --N $N \
        --num-episodes 100 \
        --K 4 \
        --output-dir "$ABS_DAD_DATA_FOLDER"
    cd "$PROJECT_ROOT"
    
    DAD_TRAJECTORY_FILE=$(find "${DATA_FOLDER}dad/" -name "dad_trajectories_N${N}_K4_random.pth" -type f 2>/dev/null | head -1)
    
    if [ ! -f "$DAD_TRAJECTORY_FILE" ]; then
        echo -e "${RED}Error: DAD trajectory file not found${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓${NC} DAD trajectories generated: $DAD_TRAJECTORY_FILE"
    
    # Step 2.5b: Train DAD policy
    echo ""
    echo "  [2.5b] Training DAD policy network..."
    
    DAD_METHOD=$(grep "dad_method:" $CONFIG_FILE | awk '{print $2}' | tr -d '"' || echo "reinforce")
    
    if [ "$DAD_METHOD" != "imitation" ] && [ "$DAD_METHOD" != "reinforce" ]; then
        echo -e "${YELLOW}Warning: Unknown DAD method '$DAD_METHOD', defaulting to 'reinforce'${NC}"
        DAD_METHOD="reinforce"
    fi
    
    echo "  Training method: $DAD_METHOD"
    if [ "$DAD_METHOD" = "reinforce" ]; then
        echo -e "${BLUE}  Using REINFORCE (direct MOCU optimization)${NC}"
    else
        echo -e "${BLUE}  Using Imitation Learning${NC}"
    fi
    
    ABS_DAD_TRAJ_FILE=$(cd "$(dirname "$DAD_TRAJECTORY_FILE")" && pwd)/$(basename "$DAD_TRAJECTORY_FILE")
    
    # Check if MPNN predictor is available
    USE_PREDICTED_MOCU=""
    if [ "$DAD_METHOD" = "reinforce" ]; then
        MPNN_MODEL_PATH="${MODEL_RUN_FOLDER}model.pth"
        if [ -f "$MPNN_MODEL_PATH" ]; then
            USE_PREDICTED_MOCU="--use-predicted-mocu"
            echo -e "${BLUE}  Using MPNN predictor for fast MOCU estimation${NC}"
        else
            echo -e "${YELLOW}  Warning: MPNN predictor not found. Using slow CUDA MOCU computation.${NC}"
        fi
    fi
    
    cd scripts
    python train_dad_policy.py \
        --data-path "$ABS_DAD_TRAJ_FILE" \
        --method "$DAD_METHOD" \
        --name "dad_policy_N${N}" \
        --epochs 100 \
        --batch-size 64 \
        --output-dir "$ABS_MODEL_RUN_FOLDER" \
        $USE_PREDICTED_MOCU
    cd "$PROJECT_ROOT"
    
    echo -e "${GREEN}✓${NC} DAD policy trained: ${MODEL_RUN_FOLDER}dad_policy_N${N}.pth"
    
    # Export DAD policy path for evaluation (set after training completes)
    export DAD_POLICY_PATH="${MODEL_RUN_FOLDER}dad_policy_N${N}.pth"
else
    echo -e "${BLUE}[Step 2.5/5]${NC} Skipping DAD training (not in methods list)"
    # Try to find existing DAD policy (from previous run)
    DAD_POLICY_PATH="${MODEL_RUN_FOLDER}dad_policy_N${N}.pth"
    if [ -f "$DAD_POLICY_PATH" ]; then
        export DAD_POLICY_PATH="$DAD_POLICY_PATH"
        echo -e "${BLUE}  Found existing DAD policy: $DAD_POLICY_PATH${NC}"
    fi
fi

# Step 3: Export configuration for evaluation scripts
echo ""
echo -e "${GREEN}[Step 3/5]${NC} Configuring experiment paths..."

export RESULT_FOLDER="$RESULT_RUN_FOLDER"

echo -e "${GREEN}✓${NC} MOCU model name: $MOCU_MODEL_NAME (via MOCU_MODEL_NAME)"
echo -e "${GREEN}✓${NC} Result folder: $RESULT_RUN_FOLDER (via RESULT_FOLDER)"

# Step 4: Run experiments
echo ""
echo -e "${GREEN}[Step 4/5]${NC} Running OED experiments..."

cd scripts
python evaluation.py --methods "$METHODS"
cd "$PROJECT_ROOT"

echo -e "${GREEN}✓${NC} Experiments complete: $RESULT_RUN_FOLDER"

# Step 5: Generate visualizations
echo ""
echo -e "${GREEN}[Step 5/5]${NC} Generating visualizations..."

ABS_RESULT_FOLDER=$(cd "$RESULT_RUN_FOLDER" && pwd)

cd scripts
python visualization.py --N $N --update_cnt 10 --result_folder "$ABS_RESULT_FOLDER"
cd "$PROJECT_ROOT"

echo -e "${GREEN}✓${NC} Plots generated: ${RESULT_RUN_FOLDER}MOCU_${N}.png, ${RESULT_RUN_FOLDER}timeComplexity_${N}.png"

# Summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Workflow Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Folder Structure:${NC}"
echo ""
echo -e "${BLUE}Data (reusable):${NC}"
echo "  ${DATA_FOLDER}"
echo "  ├── *_${N}o_train.pth  (MPNN training data)"
if echo "$METHODS" | grep -q "DAD"; then
    echo "  └── dad/"
    echo "      └── dad_trajectories_N${N}_K4_random.pth"
fi
echo ""
echo -e "${BLUE}Models (run-specific):${NC}"
echo "  ${MODEL_RUN_FOLDER}"
echo "  ├── model.pth                   (Trained MPNN predictor)"
echo "  ├── statistics.pth              (MPNN normalization stats)"
echo "  ├── curve.png                    (Training curve 1)"
echo "  ├── curve2.png                   (Training curve 2)"
echo "  └── Prediction.xlsx              (Prediction results)"
if echo "$METHODS" | grep -q "DAD"; then
    echo "  └── dad_policy_N${N}.pth        (Trained DAD policy) ⭐"
fi
echo ""
echo -e "${BLUE}Results (run-specific):${NC}"
echo "  ${RESULT_RUN_FOLDER}"
echo "  ├── *_MOCU.txt                  (MOCU curves)"
echo "  ├── *_timeComplexity.txt        (Time complexity)"
echo "  ├── *_sequence.txt              (Experiment sequences)"
echo "  ├── MOCU_${N}.png               (MOCU comparison plot)"
echo "  └── timeComplexity_${N}.png     (Time complexity plot)"
echo ""
echo -e "${GREEN}All done!${NC}"
