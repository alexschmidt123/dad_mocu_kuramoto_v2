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

# Step 0: Verify configuration
echo -e "${GREEN}[Step 0/6]${NC} Verifying configuration..."
echo -e "${GREEN}✓${NC} Configuration loaded: N=$N_GLOBAL"

# Step 1: Generate MPNN data (uses PyCUDA - original paper workflow)
echo ""
echo -e "${GREEN}[Step 1/6]${NC} Generating MPNN training data..."

# Check if data already exists in config folder
DATA_FOLDER="${PROJECT_ROOT}/data/${CONFIG_NAME}/"
TRAIN_FILE=$(find "$DATA_FOLDER" -name "*_${N}o_train.pth" -type f 2>/dev/null | head -1)

if [ -n "$TRAIN_FILE" ]; then
    echo -e "${BLUE}✓${NC} Found existing data file: $TRAIN_FILE"
    echo -e "${BLUE}  Skipping data generation (data already exists)${NC}"
    # Save train file path for next steps
    echo "$TRAIN_FILE" > /tmp/mocu_train_file_${CONFIG_NAME}.txt
else
    echo -e "${YELLOW}  No existing data found, generating new data...${NC}"
    bash "${PROJECT_ROOT}/scripts/bash/step1_generate_mocu_data.sh" "$CONFIG_FILE"
    TRAIN_FILE=$(cat /tmp/mocu_train_file_${CONFIG_NAME}.txt)
fi

# Step 2: Train MPNN predictor (runs in separate process - uses PyTorch only)
echo ""
echo -e "${GREEN}[Step 2/6]${NC} Training MPNN predictor..."
bash "${PROJECT_ROOT}/scripts/bash/step2_train_mpnn.sh" "$CONFIG_FILE" "$TRAIN_FILE"

# Step 3: Evaluate ALL baseline methods first (uses PyCUDA for MOCU computation - original paper workflow)
echo ""
echo -e "${GREEN}[Step 3/6]${NC} Running baseline evaluation (ALL original methods: iNN, NN, ODE, ENTROPY, RANDOM)..."
bash "${PROJECT_ROOT}/scripts/bash/step3_evaluate_baselines.sh" "$CONFIG_FILE"

# Step 4: Generate DAD training data and train DAD policy (if DAD is in methods list)
# This runs AFTER baselines so DAD can use the same initial MOCU
if echo "$METHODS" | grep -q "DAD"; then
    echo ""
    echo -e "${GREEN}[Step 4/6]${NC} Generating DAD training data and training DAD policy..."
    bash "${PROJECT_ROOT}/scripts/bash/step4_train_dad.sh" "$CONFIG_FILE"
else
    echo ""
    echo -e "${BLUE}[Step 4/6]${NC} Skipping DAD (not in methods list)"
fi

# Step 5: Evaluate DAD method (uses same initial MOCU as baselines)
if echo "$METHODS" | grep -q "DAD"; then
    echo ""
    echo -e "${GREEN}[Step 5/6]${NC} Running DAD evaluation (using baseline initial MOCU)..."
    bash "${PROJECT_ROOT}/scripts/bash/step5_evaluate_dad.sh" "$CONFIG_FILE"
else
    echo ""
    echo -e "${BLUE}[Step 5/6]${NC} Skipping DAD evaluation (not in methods list)"
fi

# Step 6: Generate visualizations (runs in separate process)
echo ""
echo -e "${GREEN}[Step 6/6]${NC} Generating visualizations..."
bash "${PROJECT_ROOT}/scripts/bash/step6_visualize.sh" "$CONFIG_FILE"

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
