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

# Generate timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create unified experiment ID: config_name_timestamp
EXPERIMENT_ID="${CONFIG_NAME}_${TIMESTAMP}"

# Create experiment folder structure
EXPERIMENT_ROOT="../experiments/${EXPERIMENT_ID}/"
DATA_FOLDER="${EXPERIMENT_ROOT}data/"
MODEL_FOLDER="${EXPERIMENT_ROOT}models/"
RESULT_FOLDER="${EXPERIMENT_ROOT}results/"

# Create all folders
mkdir -p "$DATA_FOLDER"
mkdir -p "$MODEL_FOLDER"
mkdir -p "$RESULT_FOLDER"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}MOCU-OED Experiment Workflow${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Config file: ${YELLOW}$CONFIG_FILE${NC}"
echo -e "Experiment folder: ${YELLOW}$EXPERIMENT_ROOT${NC}"
if [ -n "$N_GLOBAL_UPDATED" ]; then
    echo -e "${BLUE}[Re-execution after N_global update]${NC}"
fi
echo ""

# Parse YAML config file
# Note: Config files (N5_config.yaml, N7_config.yaml, etc.) are experiment configurations
# The 'model_name' field specifies the identifier for the trained model (e.g., cons5, cons7)
N=$(grep "^N:" $CONFIG_FILE | awk '{print $2}')
N_GLOBAL=$(grep "^N_global:" $CONFIG_FILE | awk '{print $2}')
TRAINED_MODEL_NAME=$(grep "model_name:" $CONFIG_FILE | awk '{print $2}' | tr -d '"')
SAMPLES=$(grep "samples_per_type:" $CONFIG_FILE | awk '{print $2}')
TRAIN_SIZE=$(grep "train_size:" $CONFIG_FILE | awk '{print $2}')
K_MAX=$(grep -A 3 "^dataset:" $CONFIG_FILE | grep "K_max:" | awk '{print $2}')
EPOCHS=$(grep "epochs:" $CONFIG_FILE | awk '{print $2}')
CONSTRAIN_WEIGHT=$(grep "constrain_weight:" $CONFIG_FILE | awk '{print $2}')
SAVE_JSON=$(grep "save_json:" $CONFIG_FILE | awk '{print $2}')

# Parse methods list from config (convert YAML list to comma-separated string)
METHODS=$(grep -A 20 "^  methods:" $CONFIG_FILE | grep '    - "' | sed 's/.*"\(.*\)".*/\1/' | grep -v '^#' | tr '\n' ',' | sed 's/,$//')

echo -e "${BLUE}Experiment Configuration:${NC}"
echo "  System size (N): $N"
echo "  N_global: $N_GLOBAL"
echo "  Trained model identifier: $TRAINED_MODEL_NAME"
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
    # Only update if not already in a re-execution
    if [ -z "$N_GLOBAL_UPDATED" ]; then
        echo -e "${YELLOW}Updating N_global from $CURRENT_N_GLOBAL to $N_GLOBAL...${NC}"
        sed -i.bak "s/#define N_global.*/#define N_global $N_GLOBAL/" $CUDA_FILE
        echo -e "${GREEN}✓${NC} N_global updated successfully"
        echo -e "${BLUE}Re-executing script to apply changes...${NC}"
        echo ""
        # Re-execute the script with updated N_global
        export N_GLOBAL_UPDATED=1
        exec bash "$0" "$@"
    else
        echo -e "${RED}Error: N_global mismatch after update. Please check $CUDA_FILE${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓${NC} N_global is correctly set to $N_GLOBAL"
fi

# Step 1: Generate dataset
echo ""
echo -e "${GREEN}[Step 1/5]${NC} Checking dataset..."

# Check if any training file exists for this N (filename may differ from config due to filtering)
EXISTING_TRAIN_FILE=$(find "${DATA_FOLDER}" -name "*_${N}o_train.pth" -type f 2>/dev/null | head -1)

if [ -n "$EXISTING_TRAIN_FILE" ]; then
    echo -e "${GREEN}✓${NC} Dataset already exists: $EXISTING_TRAIN_FILE"
    echo "  Skipping data generation..."
    TRAIN_FILE="$EXISTING_TRAIN_FILE"
else
    echo "  Generating dataset (this may take time)..."
    echo -e "${YELLOW}  Note: Actual file size may differ from config due to sync filtering${NC}"
    
    # Save current directory
    ORIGINAL_DIR=$(pwd)
    
    cd scripts
    
    # Use absolute path for output
    ABS_DATA_FOLDER=$(cd "${ORIGINAL_DIR}/${DATA_FOLDER}" && pwd)
    
    CMD="python generate_mocu_data.py --N $N --samples_per_type $SAMPLES --train_size $TRAIN_SIZE --K_max $K_MAX --output_dir $ABS_DATA_FOLDER"
    if [ "$SAVE_JSON" = "true" ]; then
        CMD="$CMD --save_json"
    fi
    
    eval $CMD
    
    # Return to original directory
    cd "$ORIGINAL_DIR"
    
    # Find the actual generated file (may have different size than config)
    TRAIN_FILE=$(find "${DATA_FOLDER}" -name "*_${N}o_train.pth" -type f 2>/dev/null | head -1)
    
    if [ -z "$TRAIN_FILE" ]; then
        echo -e "${RED}Error: No training file found in ${DATA_FOLDER}${NC}"
        echo -e "${RED}Expected pattern: *_${N}o_train.pth${NC}"
        echo -e "${YELLOW}Debug: Listing DATA_FOLDER contents:${NC}"
        ls -la "${DATA_FOLDER}" 2>&1 || echo "  Directory not accessible"
        exit 1
    fi
    
    # Extract actual size from filename
    ACTUAL_SIZE=$(basename "$TRAIN_FILE" | sed "s/_${N}o_train.pth//" )
    
    echo -e "${GREEN}✓${NC} Dataset generated: $TRAIN_FILE"
    if [ "$ACTUAL_SIZE" != "$TRAIN_SIZE" ]; then
        echo -e "${YELLOW}  Note: Generated ${ACTUAL_SIZE} samples (config requested ${TRAIN_SIZE})${NC}"
        echo -e "${YELLOW}        This is normal - some samples filtered during sync detection${NC}"
    fi
fi

# Step 2: Train model
echo ""
echo -e "${GREEN}[Step 2/5]${NC} Training MPNN predictor..."
echo "  This may take 1-2 hours..."

# Convert paths to absolute for training script
ABS_TRAIN_FILE=$(cd "$(dirname "$TRAIN_FILE")" && pwd)/$(basename "$TRAIN_FILE")
ABS_MODEL_FOLDER=$(cd "$MODEL_FOLDER" && pwd)

cd scripts
python train_mocu_predictor.py \
    --name "$EXPERIMENT_ID" \
    --data_path "$ABS_TRAIN_FILE" \
    --EPOCH $EPOCHS \
    --Constrain_weight $CONSTRAIN_WEIGHT
cd ..

echo -e "${GREEN}✓${NC} Model trained: ${MODEL_FOLDER}model.pth"

# Step 2.5: Check if DAD is in methods and train if needed
echo ""
if echo "$METHODS" | grep -q "DAD"; then
    echo -e "${GREEN}[Step 2.5/5]${NC} Training DAD policy (DAD detected in methods)..."
    
    # Step 2.5a: Generate DAD trajectory data
    echo "  [2.5a] Generating DAD trajectory data..."
    echo "  Using random expert (fast) - REINFORCE doesn't need expert labels, only a_true for experiments..."
    
    # Create DAD data folder
    mkdir -p "${DATA_FOLDER}dad/"
    
    # Convert to absolute path for DAD data generation
    ABS_DAD_DATA_FOLDER=$(cd "${DATA_FOLDER}dad/" && pwd)
    
    # Generate DAD training data (random expert - fast, sufficient for REINFORCE)
    # REINFORCE doesn't use expert actions, only needs a_true for experiments
    cd scripts
    python generate_dad_data.py \
        --N $N \
        --num-episodes 100 \
        --K 4 \
        --output-dir "$ABS_DAD_DATA_FOLDER"
    cd ..
    
    # Find the generated trajectory file (always uses 'random' expert now)
    DAD_TRAJECTORY_FILE=$(find "${DATA_FOLDER}dad/" -name "dad_trajectories_N${N}_K4_random.pth" -type f 2>/dev/null | head -1)
    
    if [ ! -f "$DAD_TRAJECTORY_FILE" ]; then
        echo -e "${RED}Error: DAD trajectory file not found${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓${NC} DAD trajectories generated: $DAD_TRAJECTORY_FILE"
    
    # Step 2.5b: Train DAD policy
    echo ""
    echo "  [2.5b] Training DAD policy network..."
    echo "  This may take 30-60 minutes..."
    
    # Check for training method in config (default: reinforce - RL with direct MOCU optimization)
    DAD_METHOD=$(grep "dad_method:" $CONFIG_FILE | awk '{print $2}' | tr -d '"' || echo "reinforce")
    
    if [ "$DAD_METHOD" != "imitation" ] && [ "$DAD_METHOD" != "reinforce" ]; then
        echo -e "${YELLOW}Warning: Unknown DAD method '$DAD_METHOD', defaulting to 'reinforce' (RL with direct MOCU optimization)${NC}"
        DAD_METHOD="reinforce"
    fi
    
    echo "  Training method: $DAD_METHOD"
    if [ "$DAD_METHOD" = "reinforce" ]; then
        echo -e "${BLUE}  Using REINFORCE (direct MOCU optimization)${NC}"
    else
        echo -e "${BLUE}  Using Imitation Learning (behavior cloning)${NC}"
    fi
    
    # Convert paths to absolute for DAD training
    ABS_DAD_TRAJ_FILE=$(cd "$(dirname "$DAD_TRAJECTORY_FILE")" && pwd)/$(basename "$DAD_TRAJECTORY_FILE")
    
    # Check if MPNN predictor is available for fast MOCU prediction
    USE_PREDICTED_MOCU=""
    if [ "$DAD_METHOD" = "reinforce" ]; then
        # Check if MPNN model exists (same name as MPNN predictor)
        # Use PROJECT_ROOT from script beginning
        MPNN_MODEL_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/models/${TRAINED_MODEL_NAME}/model.pth"
        if [ -f "$MPNN_MODEL_PATH" ]; then
            USE_PREDICTED_MOCU="--use-predicted-mocu"
            echo -e "${BLUE}  Using MPNN predictor for fast MOCU estimation${NC}"
        else
            echo -e "${YELLOW}  Warning: MPNN predictor not found. Using slow CUDA MOCU computation.${NC}"
            echo -e "${YELLOW}  To speed up: Train MPNN first, then DAD training will use fast prediction${NC}"
        fi
    fi
    
    cd scripts
    python train_dad_policy.py \
        --data-path "$ABS_DAD_TRAJ_FILE" \
        --method "$DAD_METHOD" \
        --name "dad_policy_N${N}" \
        --epochs 100 \
        --batch-size 64 \
        --output-dir "$ABS_MODEL_FOLDER" \
        $USE_PREDICTED_MOCU
    cd ..
    
    # Copy DAD policy to project models directory for evaluation
    # (evaluation.py looks for models in PROJECT_ROOT/models/)
    mkdir -p ../models
    cp "${MODEL_FOLDER}dad_policy_N${N}.pth" ../models/
    
    echo -e "${GREEN}✓${NC} DAD policy trained and copied:"
    echo "     Experiment: ${MODEL_FOLDER}dad_policy_N${N}.pth"
    echo "     Project:    ../models/dad_policy_N${N}.pth"
else
    echo -e "${BLUE}[Step 2.5/5]${NC} Skipping DAD training (not in methods list)"
fi

# Step 3: Export configuration for evaluation scripts
echo ""
echo -e "${GREEN}[Step 3/5]${NC} Configuring experiment paths..."

# Export experiment ID so Python scripts can load the correct model
export MOCU_MODEL_NAME="$EXPERIMENT_ID"
# Export result folder path so evaluation scripts know where to save
export RESULT_FOLDER="$RESULT_FOLDER"

echo -e "${GREEN}✓${NC} Experiment ID: $EXPERIMENT_ID (via MOCU_MODEL_NAME)"
echo -e "${GREEN}✓${NC} Result folder: $RESULT_FOLDER (via RESULT_FOLDER)"

# Step 4: Run experiments
echo ""
echo -e "${GREEN}[Step 4/5]${NC} Running OED experiments..."
echo "  Methods: $METHODS"
echo ""

cd scripts
python evaluation.py --methods "$METHODS"
cd ..

echo -e "${GREEN}✓${NC} Experiments complete: $RESULT_FOLDER"

# Step 5: Visualize results
echo ""
echo -e "${GREEN}[Step 5/5]${NC} Generating visualizations..."

# Convert result folder to absolute path
ABS_RESULT_FOLDER=$(cd "$RESULT_FOLDER" && pwd)

cd scripts
python visualization.py --N $N --update_cnt 10 --result_folder "$ABS_RESULT_FOLDER"
cd ..

echo -e "${GREEN}✓${NC} Plots generated: ${RESULT_FOLDER}MOCU_${N}.png, ${RESULT_FOLDER}timeComplexity_${N}.png"

# Summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Workflow Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Experiment: ${EXPERIMENT_ROOT}${NC}"
echo ""
echo -e "${BLUE}Structure:${NC}"
echo "  ${EXPERIMENT_ROOT}"
echo "  ├── data/"
echo "  │   ├── $(basename $TRAIN_FILE)  (MPNN training data)"

if echo "$METHODS" | grep -q "DAD"; then
    echo "  │   └── dad/                        (DAD trajectory data)"
    echo "  │       └── dad_trajectories_N${N}_K4_*.pth"
else
    echo "  │   └── ..."
fi

echo "  ├── models/"
echo "  │   ├── model.pth                   (Trained MPNN predictor)"
echo "  │   ├── statistics.pth              (MPNN normalization stats)"

if echo "$METHODS" | grep -q "DAD"; then
    echo "  │   └── dad_policy_N${N}.pth        (Trained DAD policy) ⭐"
else
    echo "  │   └── ..."
fi

echo "  └── results/"
echo "      ├── *_MOCU.txt                  (MOCU curves for all methods)"
echo "      ├── *_timeComplexity.txt        (Time complexity per method)"
echo "      ├── *_sequence.txt              (Experiment sequences)"
echo "      ├── MOCU_${N}.png               (MOCU comparison plot)"
echo "      └── timeComplexity_${N}.png     (Time complexity plot)"
echo ""

if echo "$METHODS" | grep -q "DAD"; then
    echo -e "${BLUE}Note:${NC} DAD policy also copied to: ../models/dad_policy_N${N}.pth"
    echo ""
fi

echo -e "${GREEN}All done!${NC}"

