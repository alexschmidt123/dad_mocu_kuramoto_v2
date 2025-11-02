#!/bin/bash
# Script to find and display available MPNN models

CONFIG_NAME="${1:-fast_config}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_BASE_DIR="${PROJECT_ROOT}/models/${CONFIG_NAME}"

echo "=========================================="
echo "Finding MPNN Models for: $CONFIG_NAME"
echo "=========================================="
echo

if [ ! -d "$MODEL_BASE_DIR" ]; then
    echo "Error: Model directory not found: $MODEL_BASE_DIR"
    exit 1
fi

echo "Available model folders:"
echo "----------------------------------------"

# Find all timestamped folders
TIMESTAMP_FOLDERS=$(find "$MODEL_BASE_DIR" -maxdepth 1 -type d -name "*_*" | sort)

if [ -z "$TIMESTAMP_FOLDERS" ]; then
    echo "  No model folders found"
    exit 1
fi

while IFS= read -r folder; do
    TIMESTAMP=$(basename "$folder")
    MODEL_NAME="${CONFIG_NAME}_${TIMESTAMP}"
    MODEL_PATH="${folder}/model.pth"
    STATS_PATH="${folder}/statistics.pth"
    
    echo "  Folder: $TIMESTAMP"
    echo "    Model name: $MODEL_NAME"
    echo "    Model file: $([ -f "$MODEL_PATH" ] && echo "✓ EXISTS" || echo "✗ MISSING")"
    echo "    Stats file: $([ -f "$STATS_PATH" ] && echo "✓ EXISTS" || echo "✗ MISSING")"
    echo
done <<< "$TIMESTAMP_FOLDERS"

echo "=========================================="
echo "To use a model, set:"
echo "  export MOCU_MODEL_NAME=\"${CONFIG_NAME}_<TIMESTAMP>\""
echo ""
echo "Example (using most recent):"
LATEST=$(find "$MODEL_BASE_DIR" -maxdepth 1 -type d -name "*_*" | sort | tail -1)
if [ -n "$LATEST" ]; then
    LATEST_TIMESTAMP=$(basename "$LATEST")
    LATEST_MODEL_NAME="${CONFIG_NAME}_${LATEST_TIMESTAMP}"
    echo "  export MOCU_MODEL_NAME=\"$LATEST_MODEL_NAME\""
fi
echo "=========================================="

