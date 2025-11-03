#!/bin/bash
# Optimized conda environment setup for DAD-MOCU project
# Uses pip-installed PyTorch to avoid MKL linking issues

set -e

ENV_NAME="dad_mocu"

echo "================================================================================"
echo "Creating optimized conda environment: $ENV_NAME"
echo "================================================================================"
echo ""
echo "This environment addresses both issues:"
echo "  - CUDA segfault: PyTorch 2.4.0 + CUDA 12.1 (stable version)"
echo "  - MKL linking: Using pip PyTorch (pre-built, avoids iJIT_NotifyEvent error)"
echo ""

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "WARNING: Environment '$ENV_NAME' already exists!"
    read -p "Do you want to remove it and create a new one? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n $ENV_NAME -y
    else
        echo "Aborted. Please choose a different environment name."
        exit 1
    fi
fi

echo "Step 1: Creating base environment with Python 3.10..."
conda create -n $ENV_NAME python=3.10 -y

echo ""
echo "Step 2: Activating environment..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo ""
echo "Step 3: Installing core dependencies via conda..."
conda install -y -c conda-forge \
    numpy \
    scipy \
    matplotlib \
    tqdm \
    pyyaml \
    pandas \
    pip \
    setuptools \
    wheel

echo ""
echo "Step 4: Installing PyTorch via pip (avoids MKL linking issues)..."
# Use pip PyTorch - pre-built wheels don't have MKL linking problems
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "Step 5: Installing PyTorch Geometric and extensions..."
pip install torch-geometric

# Install PyG extensions - use pre-built wheels matching PyTorch 2.4.0 + CUDA 12.1
PYTORCH_VERSION="2.4.0"
CUDA_VERSION="cu121"
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-${PYTORCH_VERSION}+${CUDA_VERSION}.html

echo ""
echo "Step 6: Installing CUDA Toolkit (required for PyCUDA kernel compilation)..."
# Install CUDA toolkit with nvcc compiler (required for PyCUDA)
conda install -y -c nvidia cuda-toolkit=12.1

echo ""
echo "Step 7: Installing PyCUDA..."
pip install pycuda

echo ""
echo "Step 8: Installing additional Python dependencies..."
pip install openpyxl

echo ""
echo "Step 9: Verifying installation..."
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}'); print(f'✓ CUDA available: {torch.cuda.is_available()}'); print(f'✓ CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
python -c "import torch_geometric; print(f'✓ PyTorch Geometric: {torch_geometric.__version__}')"
python -c "import pycuda; print('✓ PyCUDA: OK')"
# Verify nvcc is available
if command -v nvcc &> /dev/null; then
    echo "✓ nvcc (CUDA compiler): $(nvcc --version | grep release | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')"
else
    echo "⚠ WARNING: nvcc not found in PATH (PyCUDA kernel compilation may fail)"
    echo "   Try: conda install -c nvidia cuda-toolkit=12.1"
fi

echo ""
echo "================================================================================"
echo "Environment '$ENV_NAME' created successfully!"
echo "================================================================================"
echo ""
echo "To activate:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To test CUDA training:"
echo "  python scripts/train_dad_policy.py --device cuda ..."
echo ""

