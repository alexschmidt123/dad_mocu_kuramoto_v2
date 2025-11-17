# DAD-MOCU: Deep Adaptive Design for Optimal Experimental Design

This project implements Deep Adaptive Design (DAD) and Implicit Deep Adaptive Design (iDAD) methods for sequential optimal experimental design using MOCU (Model-based Objective-based Characterization of Uncertainty) as the objective. The framework uses neural message passing networks (MPNN) to accelerate MOCU prediction and reinforcement learning to learn optimal experimental selection policies.

## Hardware and Environment Setup

**Hardware**:
- **Processor**: 13th Gen Intel® Core™ i7-13700F (24 cores)
- **Memory**: 64.0 GiB
- **Graphics**: NVIDIA GeForce RTX 4090 (24GB)
- **OS**: Ubuntu 22.04.5 LTS (64-bit)

**Create conda environment and install dependencies**:

```bash
# Create conda environment (Python 3.10)
conda create -n dad_mocu python=3.10 -y
conda activate dad_mocu

# Install core dependencies
conda install -y -c conda-forge numpy scipy matplotlib tqdm pyyaml pandas pip setuptools wheel

# Install PyTorch with CUDA 12.1 (via pip to avoid MKL issues)
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric
pip install torch-geometric

# Install PyG extensions
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

# Install CUDA Toolkit (required for PyCUDA kernel compilation)
conda install -y -c nvidia cuda-toolkit=12.1

# Install PyCUDA (required for MOCU computation)
pip install pycuda

# Install additional dependencies
pip install openpyxl
```

## Running Experiments

### Using `run.sh` with Different Configs

Run the complete pipeline (data generation, training, evaluation) for a specific configuration:

```bash
conda activate dad_mocu

# Quick test (3-5 minutes)
bash run.sh configs/fast_config.yaml

# Full experiments
bash run.sh configs/N5_config.yaml    # 5-oscillator system
bash run.sh configs/N7_config.yaml    # 7-oscillator system
bash run.sh configs/N9_config.yaml    # 9-oscillator system

# Override K (number of sequential experiments)
bash run.sh configs/N5_config.yaml 4  # Use K=4 instead of default
```

The script creates a self-contained experiment directory: `experiments/<config>_<timestamp>/` containing all results, models, logs, and evaluations.

### Using `run_sweepK.sh` (Optional)

Run experiments with multiple K values to study the effect of sequence length:

```bash
# Run with default K values: 4, 6, 8, 10
bash run_sweepK.sh configs/N5_config.yaml

# Run with custom K values
bash run_sweepK.sh configs/N5_config.yaml 2 4 6 8 10
```

This will run the complete pipeline for each K value and save results in separate experiment directories. Each run reuses shared MPNN data and models but generates fresh DAD training data and models for each K value.
