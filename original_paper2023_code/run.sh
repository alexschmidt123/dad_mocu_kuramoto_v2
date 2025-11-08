#!/bin/bash
# Main script to run complete MOCU-OED experiment workflow
# Usage: bash run.sh configs/N5_config.yaml

set -e  # Exit on error

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

# Generate timestamp for this run (format: MMDDYYYY_HHMMSS)
TIMESTAMP=$(date +"%m%d%Y_%H%M%S")

# Create unified experiment ID: config_name_timestamp
EXPERIMENT_ID="${CONFIG_NAME}_${TIMESTAMP}"

# Get absolute path to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create experiment folder structure (relative to project root)
EXPERIMENT_ROOT="${SCRIPT_DIR}/experiments/${EXPERIMENT_ID}/"
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

echo -e "${BLUE}Experiment Configuration:${NC}"
echo "  System size (N): $N"
echo "  N_global: $N_GLOBAL"
echo "  Trained model identifier: $TRAINED_MODEL_NAME"
echo "  Samples per type: $SAMPLES"
echo "  Training set size: $TRAIN_SIZE"
echo "  Epochs: $EPOCHS"
echo ""

# Step 0: Check and update N_global in CUDA code
echo -e "${GREEN}[Step 0/4]${NC} Checking CUDA N_global configuration..."
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
echo -e "${GREEN}[Step 1/4]${NC} Checking dataset..."

TRAIN_FILE="${DATA_FOLDER}${TRAIN_SIZE}_${N}o_train.pth"

if [ -f "$TRAIN_FILE" ]; then
    echo -e "${GREEN}✓${NC} Dataset already exists: $TRAIN_FILE"
    echo "  Skipping data generation..."
else
    echo "  Generating dataset (this may take time)..."
    
    cd scripts
    
    CMD="python data_generation.py --N $N --samples_per_type $SAMPLES --train_size $TRAIN_SIZE --K_max $K_MAX --output_dir $DATA_FOLDER"
    if [ "$SAVE_JSON" = "true" ]; then
        CMD="$CMD --save_json"
    fi
    
    eval $CMD
    cd ..
    
    if [ ! -f "$TRAIN_FILE" ]; then
        echo -e "${RED}Error: Training file not found: $TRAIN_FILE${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓${NC} Dataset generated: $TRAIN_FILE"
fi

# Step 2: Train model
echo ""
echo -e "${GREEN}[Step 2/4]${NC} Training model..."
echo "  This may take 1-2 hours..."

cd scripts
python training.py \
    --name "$EXPERIMENT_ID" \
    --data_path "$TRAIN_FILE" \
    --model_dir "$MODEL_FOLDER" \
    --EPOCH $EPOCHS \
    --Constrain_weight $CONSTRAIN_WEIGHT
cd ..

echo -e "${GREEN}✓${NC} Model trained: ${MODEL_FOLDER}model.pth"

# Step 3: Export configuration for evaluation scripts
echo ""
echo -e "${GREEN}[Step 3/4]${NC} Configuring experiment paths..."

# Export experiment ID so Python scripts can load the correct model
export MOCU_MODEL_NAME="$EXPERIMENT_ID"
# Export result folder path so evaluation scripts know where to save
export RESULT_FOLDER="$RESULT_FOLDER"

echo -e "${GREEN}✓${NC} Experiment ID: $EXPERIMENT_ID (via MOCU_MODEL_NAME)"
echo -e "${GREEN}✓${NC} Result folder: $RESULT_FOLDER (via RESULT_FOLDER)"

# Step 4: Run experiments
echo ""
echo -e "${GREEN}[Step 4/4]${NC} Running OED experiments..."
echo "  Note: Edit scripts/evaluation.py if needed to set:"
echo "    - N = $N (line 24)"
echo "    - Methods to run (line 43)"
echo ""

cd scripts
python evaluation.py
cd ..

echo -e "${GREEN}✓${NC} Experiments complete: $RESULT_FOLDER"

# Step 5: Visualize results
echo ""
echo -e "${GREEN}[Step 5/4]${NC} Generating visualizations..."

cd scripts
python visualization.py --N $N --update_cnt 10 --result_folder "$RESULT_FOLDER"
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
echo "  │   └── ${TRAIN_SIZE}_${N}o_train.pth"
echo "  ├── models/"
echo "  │   ├── model.pth"
echo "  │   └── statistics.pth"
echo "  └── results/"
echo "      ├── *_MOCU.txt"
echo "      ├── MOCU_${N}.png"
echo "      └── timeComplexity_${N}.png"
echo ""
echo -e "${GREEN}All done!${NC}"

