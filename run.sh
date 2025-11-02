#!/bin/bash
# Main workflow script - ORCHESTRATES separate steps
# Each step runs in its own process for clean CUDA context isolation
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

CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
TIMESTAMP=$(date +"%m%d%Y_%H%M%S")

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}MOCU-OED Experiment Workflow${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Config file: ${YELLOW}$CONFIG_FILE${NC}"
echo -e "Run timestamp: ${YELLOW}$TIMESTAMP${NC}"
echo ""

# Parse basic config
N=$(grep "^N:" $CONFIG_FILE | awk '{print $2}')
N_GLOBAL=$(grep "^N_global:" $CONFIG_FILE | awk '{print $2}')
METHODS=$(grep -A 20 "^  methods:" $CONFIG_FILE | grep '    - "' | sed 's/.*"\(.*\)".*/\1/' | grep -v '^#' | tr '\n' ',' | sed 's/,$//')

echo -e "${BLUE}Experiment Configuration:${NC}"
echo "  System size (N): $N"
echo "  N_global: $N_GLOBAL"
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

# Step 1: Generate MPNN data (runs in separate process - uses PyCUDA)
echo ""
echo -e "${GREEN}[Step 1/5]${NC} Generating MPNN training data..."
bash "${PROJECT_ROOT}/scripts/bash/step1_generate_mocu_data.sh" "$CONFIG_FILE"
TRAIN_FILE=$(cat /tmp/mocu_train_file_${CONFIG_NAME}.txt)

# Step 2: Train MPNN predictor (runs in separate process - uses PyTorch only)
echo ""
echo -e "${GREEN}[Step 2/5]${NC} Training MPNN predictor..."
bash "${PROJECT_ROOT}/scripts/bash/step2_train_mpnn.sh" "$CONFIG_FILE" "$TRAIN_FILE"

# Step 3: Train DAD policy (runs in separate process - uses MPNN predictor only, no PyCUDA)
if echo "$METHODS" | grep -q "DAD"; then
    echo ""
    echo -e "${GREEN}[Step 3/5]${NC} Training DAD policy..."
    bash "${PROJECT_ROOT}/scripts/bash/step3_train_dad.sh" "$CONFIG_FILE"
else
    echo ""
    echo -e "${BLUE}[Step 3/5]${NC} Skipping DAD training (not in methods list)"
fi

# Step 4: Evaluate methods (runs in separate process - may use PyCUDA for evaluation)
echo ""
echo -e "${GREEN}[Step 4/5]${NC} Running evaluation..."
bash "${PROJECT_ROOT}/scripts/bash/step4_evaluate.sh" "$CONFIG_FILE"

# Step 5: Generate visualizations (runs in separate process)
echo ""
echo -e "${GREEN}[Step 5/5]${NC} Generating visualizations..."
bash "${PROJECT_ROOT}/scripts/bash/step5_visualize.sh" "$CONFIG_FILE"

# Summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Workflow Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Results:${NC}"
RESULT_RUN_FOLDER=$(ls -td ${PROJECT_ROOT}/results/${CONFIG_NAME}/*/ 2>/dev/null | head -1)
if [ -n "$RESULT_RUN_FOLDER" ]; then
    echo "  $RESULT_RUN_FOLDER"
fi
echo ""
