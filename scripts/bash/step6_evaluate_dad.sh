#!/bin/bash
# Step 6: Evaluate DAD methods and generate final visualizations

set -e

CONFIG_FILE=$1
if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
BASELINE_RESULTS=$(cat /tmp/baseline_results_folder_${CONFIG_NAME}.txt 2>/dev/null || echo "")
DAD_MOCU_POLICY_PATH=$(cat /tmp/dad_mocu_policy_path_${CONFIG_NAME}.txt 2>/dev/null || echo "")
IDAD_MOCU_POLICY_PATH=$(cat /tmp/idad_mocu_policy_path_${CONFIG_NAME}.txt 2>/dev/null || echo "")

if [ -z "$BASELINE_RESULTS" ] || [ ! -d "$BASELINE_RESULTS" ]; then
    echo "Error: Baseline results not found. Run step3_evaluate_baselines.sh first."
    exit 1
fi

HAS_DAD_MOCU=false
HAS_IDAD_MOCU=false
[ -n "$DAD_MOCU_POLICY_PATH" ] && [ -f "$DAD_MOCU_POLICY_PATH" ] && HAS_DAD_MOCU=true
[ -n "$IDAD_MOCU_POLICY_PATH" ] && [ -f "$IDAD_MOCU_POLICY_PATH" ] && HAS_IDAD_MOCU=true

if [ "$HAS_DAD_MOCU" = false ] && [ "$HAS_IDAD_MOCU" = false ]; then
    echo "Error: No DAD policies found. Run step5_train_dad_policy.sh first."
    exit 1
fi

N=$(grep "^N:" $CONFIG_FILE | awk '{print $2}')
UPDATE_CNT=$(grep -A 10 "^experiment:" $CONFIG_FILE | grep "update_count:" | awk '{print $2}')
IT_IDX=$(grep -A 10 "^experiment:" $CONFIG_FILE | grep "it_idx:" | awk '{print $2}')
K_MAX=$(grep -A 10 "^experiment:" $CONFIG_FILE | grep "K_max:" | awk '{print $2}')
NUM_SIMULATIONS=$(grep -A 10 "^experiment:" $CONFIG_FILE | grep "num_simulations:" | awk '{print $2}')
[ -z "$N" ] && N=5
[ -z "$UPDATE_CNT" ] && UPDATE_CNT=10
[ -z "$IT_IDX" ] && IT_IDX=10
[ -z "$K_MAX" ] && K_MAX=20480
[ -z "$NUM_SIMULATIONS" ] && NUM_SIMULATIONS=10

RESULT_FOLDER="$BASELINE_RESULTS"
export RESULT_FOLDER="$RESULT_FOLDER"
export EVAL_N="$N"
export EVAL_UPDATE_CNT="$UPDATE_CNT"
export EVAL_IT_IDX="$IT_IDX"
export EVAL_K_MAX="$K_MAX"
export EVAL_NUM_SIMULATIONS="$NUM_SIMULATIONS"

echo "Running DAD evaluation (Step 6/6)..."
echo "  Using baseline results: $BASELINE_RESULTS"
echo "  N=$N, update_cnt=$UPDATE_CNT, it_idx=$IT_IDX, K_max=$K_MAX, num_simulations=$NUM_SIMULATIONS"
echo "  Results: $RESULT_FOLDER"

cd "${PROJECT_ROOT}/scripts"

if [ "$HAS_DAD_MOCU" = true ]; then
    echo ""
    echo "Evaluating DAD_MOCU method..."
    export DAD_POLICY_PATH="$DAD_MOCU_POLICY_PATH"
    python3 dad_eval.py --baseline_results "$BASELINE_RESULTS" --result_folder "$RESULT_FOLDER" --method_name "DAD_MOCU"
    echo "✓ DAD_MOCU evaluation complete"
fi

if [ "$HAS_IDAD_MOCU" = true ]; then
    echo ""
    echo "Evaluating IDAD_MOCU method..."
    export DAD_POLICY_PATH="$IDAD_MOCU_POLICY_PATH"
    python3 dad_eval.py --baseline_results "$BASELINE_RESULTS" --result_folder "$RESULT_FOLDER" --method_name "IDAD_MOCU"
    echo "✓ IDAD_MOCU evaluation complete"
fi

echo ""
echo "Generating final visualizations..."
ABS_RESULT_FOLDER=$(cd "$RESULT_FOLDER" && pwd)
if [ "${ABS_RESULT_FOLDER: -1}" != "/" ]; then
    ABS_RESULT_FOLDER="${ABS_RESULT_FOLDER}/"
fi

if [ -n "$UPDATE_CNT" ]; then
    python3 visualize.py --N $N --update_cnt $UPDATE_CNT --result_folder "$ABS_RESULT_FOLDER"
else
    python3 visualize.py --N $N --result_folder "$ABS_RESULT_FOLDER"
fi

echo "✓ Final visualizations generated in $ABS_RESULT_FOLDER"
echo ""
echo "✓ DAD evaluation complete: $RESULT_FOLDER"

