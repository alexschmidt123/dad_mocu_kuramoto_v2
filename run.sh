#!/bin/bash
# Main workflow script - ORCHESTRATES separate steps
# Each step runs in its own process for clean CUDA context isolation
# Usage: bash run.sh configs/N5_config.yaml [K_override]
#        K_OVERRIDE=4 bash run.sh configs/N5_config.yaml
#
# If K_OVERRIDE is set (via argument or env var), it will:
# - Update experiment.update_count = K_OVERRIDE
# - Update dad_data.K = K_OVERRIDE
# - Create a temporary config file with updated K

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
    echo "Usage: bash run.sh <config_file> [K_override]"
    echo "       K_OVERRIDE=4 bash run.sh configs/N5_config.yaml"
    echo "Example: bash run.sh configs/N5_config.yaml 4"
    exit 1
fi

CONFIG_FILE=$1

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Configuration file '$CONFIG_FILE' not found${NC}"
    exit 1
fi

# Resolve config file to absolute path
if [[ "$CONFIG_FILE" != /* ]]; then
    CONFIG_FILE="${PROJECT_ROOT}/${CONFIG_FILE}"
fi

ORIGINAL_CONFIG_FILE="$CONFIG_FILE"
CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
# Remove _K* suffix if present to get base config name
BASE_CONFIG_NAME=$(echo "$CONFIG_NAME" | sed 's/_K[0-9]*$//')
CONFIG_NAME="$BASE_CONFIG_NAME"  # Always use base name for folders
TIMESTAMP=$(date +"%m%d%Y_%H%M%S")

# Read K from config file (default from config)
CURRENT_UPDATE_COUNT=$(grep -A 3 "^experiment:" "$ORIGINAL_CONFIG_FILE" | grep "update_count:" | awk '{print $2}' || echo "")
CURRENT_DAD_K=$(grep -A 3 "^dad_data:" "$ORIGINAL_CONFIG_FILE" | grep "  K:" | awk '{print $2}' || echo "")

# Use config K as default, allow override via argument or env var
# Priority: argument > env var > config file
K_VALUE=${2:-${K_OVERRIDE:-}}
if [ -z "$K_VALUE" ]; then
    # No override provided, use value from config
    if [ -n "$CURRENT_UPDATE_COUNT" ]; then
        K_VALUE="$CURRENT_UPDATE_COUNT"
    elif [ -n "$CURRENT_DAD_K" ]; then
        K_VALUE="$CURRENT_DAD_K"
    else
        # Fallback to 10 if not in config
        K_VALUE=10
        echo -e "${YELLOW}Warning: K not found in config, using default K=10${NC}"
    fi
fi

# Validate K is a positive integer
if ! [[ "$K_VALUE" =~ ^[0-9]+$ ]] || [ "$K_VALUE" -le 0 ]; then
    echo -e "${RED}Error: K must be a positive integer, got: $K_VALUE${NC}"
    exit 1
fi

# Create experiment directory structure and resolved config
EXPERIMENTS_ROOT="${PROJECT_ROOT}/experiments"
EXPERIMENT_NAME="${CONFIG_NAME}_${TIMESTAMP}"
EXPERIMENT_DIR="${EXPERIMENTS_ROOT}/${EXPERIMENT_NAME}"
mkdir -p "${EXPERIMENT_DIR}/dad_data" "${EXPERIMENT_DIR}/dad_models" "${EXPERIMENT_DIR}/logs" "${EXPERIMENT_DIR}/eval"

export EXPERIMENT_DIR
export EXP_DAD_DATA_DIR="${EXPERIMENT_DIR}/dad_data"
export EXP_DAD_MODELS_DIR="${EXPERIMENT_DIR}/dad_models"
export EXP_LOGS_DIR="${EXPERIMENT_DIR}/logs"
export EXP_EVAL_DIR="${EXPERIMENT_DIR}/eval"

WORKFLOW_LOG="${EXP_LOGS_DIR}/workflow.log"
touch "$WORKFLOW_LOG"
exec > >(tee -a "$WORKFLOW_LOG")
exec 2>&1

RESOLVED_CONFIG_PATH="${EXPERIMENT_DIR}/${CONFIG_NAME}.yaml"

python3 << PYEOF
import yaml
from pathlib import Path

with open('$ORIGINAL_CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

if 'experiment' in config:
    config['experiment']['update_count'] = $K_VALUE

if 'dad_data' in config:
    config['dad_data']['K'] = $K_VALUE

output_path = Path('$RESOLVED_CONFIG_PATH')
with output_path.open('w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

print(f"Wrote experiment config to {output_path}")
PYEOF

CONFIG_FILE="$RESOLVED_CONFIG_PATH"

if [ -n "$2" ] || [ -n "${K_OVERRIDE:-}" ]; then
    echo -e "${BLUE}K override: Setting K=$K_VALUE (config had update_count=$CURRENT_UPDATE_COUNT, dad_data.K=$CURRENT_DAD_K)${NC}"
else
    echo -e "${BLUE}Using K=$K_VALUE from config${NC}"
fi

echo -e "${BLUE}Experiment directory:${NC} $EXPERIMENT_DIR"
echo -e "${BLUE}Logging to:${NC} $WORKFLOW_LOG"
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}MOCU-OED Experiment Workflow${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Config file: ${YELLOW}$ORIGINAL_CONFIG_FILE${NC}"
if [ -n "$2" ] || [ -n "${K_OVERRIDE:-}" ]; then
    echo -e "K value: ${YELLOW}K=$K_VALUE${NC} (override: config had update_count=$CURRENT_UPDATE_COUNT, dad_data.K=$CURRENT_DAD_K)"
else
    echo -e "K value: ${YELLOW}K=$K_VALUE${NC} (from config file)"
fi
echo -e "Run timestamp: ${YELLOW}$TIMESTAMP${NC}"
echo ""

# Parse basic config
N=$(grep "^N:" $CONFIG_FILE | awk '{print $2}')
N_GLOBAL=$(grep "^N_global:" $CONFIG_FILE | awk '{print $2}')

# Extract methods list - handle both original format (    - "METHOD") and dumped format (  - METHOD)
# Try Python first (more robust), fallback to grep
METHODS=$(python3 << PYEOF
import yaml
import sys

try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    
    if 'experiment' in config and 'methods' in config['experiment']:
        methods = config['experiment']['methods']
        if isinstance(methods, list):
            print(','.join(methods))
        else:
            print('')
    else:
        print('')
except Exception as e:
    print('', file=sys.stderr)
    sys.exit(1)
PYEOF
)

# Fallback to grep if Python fails
if [ -z "$METHODS" ]; then
    METHODS=$(grep -A 20 "^  methods:" $CONFIG_FILE | grep -E '^\s+-' | sed -E 's/^\s+-\s*"?([^"]*)"?.*/\1/' | grep -v '^#' | tr '\n' ',' | sed 's/,$//')
fi

echo -e "${BLUE}Experiment Configuration:${NC}"
echo "  System size (N): $N"
echo "  N_global: $N_GLOBAL"
echo "  K: $K_VALUE"
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

# Step 3: Evaluate baseline methods first and visualize (uses PyCUDA for MOCU computation - original paper workflow)
echo ""
echo -e "${GREEN}[Step 3/6]${NC} Running baseline evaluation and visualization..."
bash "${PROJECT_ROOT}/scripts/bash/step3_evaluate_baselines.sh" "$CONFIG_FILE"

# Step 4: Generate DAD training data and train DAD policies (if DAD_MOCU or IDAD_MOCU is in methods list)
# This runs AFTER baselines so DAD can use the same initial MOCU
if echo "$METHODS" | grep -qE "(DAD_MOCU|IDAD_MOCU)"; then
    echo ""
    echo -e "${GREEN}[Step 4/6]${NC} Generating DAD training data..."
    bash "${PROJECT_ROOT}/scripts/bash/step4_generate_dad_data.sh" "$CONFIG_FILE"

    echo ""
    echo -e "${GREEN}[Step 5/6]${NC} Training DAD policies..."
    bash "${PROJECT_ROOT}/scripts/bash/step5_train_dad_policy.sh" "$CONFIG_FILE"

    echo ""
    echo -e "${GREEN}[Step 6/6]${NC} Evaluating DAD methods and generating final visualizations..."
    bash "${PROJECT_ROOT}/scripts/bash/step6_evaluate_dad.sh" "$CONFIG_FILE"
else
    echo ""
    echo -e "${BLUE}[Step 4/6]${NC} Skipping DAD data generation (DAD_MOCU/IDAD_MOCU not in methods list)"
    echo -e "${BLUE}[Step 5/6]${NC} Skipping DAD policy training"
    echo -e "${BLUE}[Step 6/6]${NC} Skipping DAD evaluation/visualization"
fi

# Summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Workflow Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Results:${NC}"
echo "  $EXPERIMENT_DIR"
echo ""
