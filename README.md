# AccelerateOED + DAD-MOCU

Code for paper **"Neural Message Passing for Objective-Based Uncertainty Quantification and Optimal Experimental Design"** with Deep Adaptive Design extension.

This repository implements MOCU-OED framework using neural message passing to accelerate optimal experimental design for coupled oscillator systems by **100-1000×**, plus a new Deep Adaptive Design (DAD) method for sequential experimental design.

## Environment Requirements

- **OS**: Linux (Ubuntu 22.04+), macOS, or Windows
- **Python**: 3.10
- **GPU**: NVIDIA GPU with CUDA 12.1+ (for CUDA acceleration)
- **Hardware**: Tested on GeForce RTX 4080

## Installation

### Option 1: Automated Installation (Recommended)

Use the provided installation script to create a conda environment with all dependencies:

```bash
bash installation/create_mocu_env.sh
```

This script will:
1. Create a conda environment named `dad_mocu`
2. Install Python 3.10 and core dependencies
3. Install PyTorch 2.4.0 with CUDA 12.1 support (via pip to avoid MKL linking issues)
4. Install PyTorch Geometric and extensions
5. Install PyCUDA for high-performance MOCU computation

**Note**: The script uses pip-installed PyTorch to avoid MKL library conflicts (`iJIT_NotifyEvent` errors).

### Option 2: Manual Installation via YAML

Alternatively, create the environment from the YAML file:

```bash
conda env create -f installation/mocu_env.yaml
conda activate dad_mocu

# Install PyTorch Geometric extensions manually
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

### Option 3: Manual Step-by-Step Installation

If you prefer manual control:

```bash
# 1. Create environment
conda create -n dad_mocu python=3.10 -y
conda activate dad_mocu

# 2. Install core dependencies
conda install -y -c conda-forge numpy scipy matplotlib tqdm pyyaml pandas pip setuptools wheel

# 3. Install PyTorch with CUDA 12.1 (via pip to avoid MKL issues)
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 4. Install PyTorch Geometric
pip install torch-geometric

# 5. Install PyG extensions
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

# 6. Install CUDA Toolkit (required for PyCUDA kernel compilation)
conda install -y -c nvidia cuda-toolkit=12.1

# 7. Install PyCUDA
pip install pycuda

# 8. Install additional dependencies
pip install openpyxl
```

### Verify Installation

After installation, verify that everything works:

```bash
conda activate dad_mocu

# Verify PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Verify PyTorch Geometric
python -c "import torch_geometric; print(f'PyTorch Geometric: {torch_geometric.__version__}')"

# Verify PyCUDA
python -c "import pycuda; print('PyCUDA: OK')"

# Verify nvcc (CUDA compiler) - REQUIRED for PyCUDA kernel compilation
nvcc --version || echo "⚠ WARNING: nvcc not found. Install with: conda install -c nvidia cuda-toolkit=12.1"
```

**Note**: CPU-only mode is supported but significantly slower. GPU acceleration is recommended for production use.

## Project Structure

```
dad_mocu_kuramoto_v2/
├── installation/                 # Environment setup files
│   ├── create_mocu_env.sh       # Automated installation script
│   └── mocu_env.yaml            # Conda environment YAML file
│
├── configs/                      # Configuration files
│   ├── fast_config.yaml         # Quick verification (3-5 min)
│   ├── N5_config.yaml           # 5-oscillator system
│   ├── N7_config.yaml           # 7-oscillator system
│   └── N9_config.yaml           # 9-oscillator system
│
├── scripts/                      # Execution scripts
│   ├── generate_mocu_data.py    # Generate MOCU predictor training data
│   ├── generate_dad_data.py     # Generate DAD policy training data
│   ├── train_predictor.py       # Train MPNN+ predictor (for iNN/NN)
│   ├── train_dad_policy.py      # Train DAD policy network
│   ├── compare_predictors.py    # Compare MOCU predictors (MSE, speed) 📊
│   ├── evaluate.py              # Run OED experiments (all methods) ⭐
│   └── visualize.py             # Generate plots
│
├── scripts/bash/               # Shell scripts (workflow orchestration)
│   ├── step1_generate_mocu_data.sh
│   ├── step2_train_mpnn.sh
│   ├── step3_train_dad.sh
│   ├── step4_evaluate.sh
│   └── step5_visualize.sh
│
├── src/
│   ├── methods/                    # OED selection methods
│   │   ├── base.py                # Abstract base class
│   │   ├── inn.py, nn.py          # MPNN-based methods
│   │   ├── ode.py                 # Sampling-based methods
│   │   ├── entropy.py             # Greedy heuristic
│   │   ├── random.py              # Random baseline
│   │   └── dad_mocu.py            # Deep Adaptive Design ⭐
│   │
│   ├── models/                     # Neural network models
│   │   ├── predictors/            # MOCU prediction models
│   │   │   ├── mlp.py             # MLP baseline
│   │   │   ├── cnn.py             # CNN baseline
│   │   │   ├── mpnn.py            # Basic MPNN
│   │   │   ├── mpnn_plus.py       # MPNN+ (winner)
│   │   │   ├── ensemble.py        # Ensemble predictor
│   │   │   ├── sampling_mocu.py  # Ground truth (sampling)
│   │   │   ├── utils.py           # Shared utilities
│   │   │   └── predictor_utils.py # Predictor loading utilities
│   │   └── policy_networks.py     # DAD policy network
│   │
│   ├── core/                       # Core computation
│   │   ├── mocu.py                 # PyTorch CUDA-accelerated MOCU computation
│   │   └── sync_detection.py      # Synchronization detection
│   │
│   └── utils/
│       └── utils.py
│
├── models/                      # Saved trained models
├── data/                        # Generated datasets
├── results/                     # Experiment results
└── run.sh                       # Main automation script
```

## Quick Start

### Full Workflow (Automated)

Run the complete experiment pipeline for a specific configuration:

```bash
conda activate dad_mocu
bash run.sh configs/fast_config.yaml
```

This will:
1. Generate MPNN training data
2. Train MPNN predictor
3. Train DAD policy (if DAD method is enabled)
4. Evaluate all methods
5. Generate visualizations

### Component Verification (3-5 minutes) ⚡

Quickly test if all components work:

```bash
conda activate dad_mocu
bash run.sh configs/fast_config.yaml
```

| Step | Time | What it does |
|------|------|--------------|
| Data Generation | ~2 min | Generate MPNN training data |
| Training | ~1 min | Train MPNN predictor |
| Evaluation | ~1 min | Test all OED methods |
| Visualization | ~5 sec | Generate plots (optional) |

### Full Experiments

Run complete experiments (for research):

```bash
# For N=5 oscillators (~5-10 hours)
bash run.sh configs/N5_config.yaml

# For N=7 oscillators (~15-20 hours)
bash run.sh configs/N7_config.yaml

# For N=9 oscillators (~24-30 hours)
bash run.sh configs/N9_config.yaml
```

## Two-Level Architecture

### Level 1: MOCU Prediction

**Location**: `src/models/predictors/`

Predict MOCU values from coupling bounds:
- **MLP**, **CNN** - Baseline neural networks
- **MPNN+** - Message Passing NN with ranking constraint (winner)
- **Sampling-based** - Ground truth (Monte Carlo with PyTorch CUDA)
- **Ensemble** - Combine multiple models

**Evaluate predictors**:
```bash
python scripts/compare_predictors.py --test_data ./data/test_data.pt
```

This compares all predictors on:
- ✅ Prediction accuracy (MSE, MAE, RMSE)
- ✅ Inference speed (time per sample)
- ✅ Model size (parameters, memory)

### Level 2: OED Methods

**Location**: `src/methods/`

Select next experiment to minimize terminal MOCU:
- **iNN** - Iterative MPNN (re-compute each step)
- **NN** - Static MPNN (compute once)
- **ODE**, **iODE** - Sampling-based (exact but slow, uses PyTorch CUDA)
- **ENTROPY** - Greedy uncertainty heuristic
- **RANDOM** - Random baseline
- **DAD-MOCU** ⭐ - Deep Adaptive Design (learned policy)

## Usage

### Using Unified API

```python
from src.methods import iNN_Method, ENTROPY_Method, DAD_MOCU_Method

# All methods have the same interface
method = iNN_Method(
    N=5, K_max=20480, deltaT=1/160,
    MReal=800, TReal=5, it_idx=10,
    model_name='cons5'
)

# Run OED episode
MOCUCurve, experiments, times = method.run_episode(
    w_init=w,
    a_lower_init=a_lower,
    a_upper_init=a_upper,
    criticalK_init=criticalK,
    isSynchronized_init=isSynchronized,
    update_cnt=10
)
```

## Training Workflow

### 1. Train MOCU Predictor (for iNN/NN methods)

```bash
# Generate training data
python scripts/generate_mocu_data.py --N 5 --samples_per_type 37500

# Train MPNN+ predictor
python scripts/train_predictor.py --data_path ./data/ --name cons5
```

**Note**: Sampling-based methods (ODE, iODE) don't need training - they compute MOCU directly using PyTorch CUDA acceleration.

### 2. Train DAD Policy (Optional)

```bash
# Generate DAD trajectories using iNN as expert
python scripts/generate_dad_data.py --N 5 --num_episodes 1000 --expert_method iNN

# Train DAD policy network
python scripts/train_dad_policy.py --data_path ./data/dad_training_data/ --name dad_policy_N5 --method reinforce
```

### 3. Evaluate All Methods

```bash
export MOCU_MODEL_NAME=cons5
python scripts/evaluate.py
```

This evaluates ALL methods: iNN, NN, ODE, ENTROPY, RANDOM, DAD

## Complete Research Workflow

### Step 1: Evaluate MOCU Predictors (Paper Table 1)

Compare different prediction models:

```bash
# Generate test data
python scripts/generate_mocu_data.py --N 5 --samples_per_type 5000

# Train different predictors
python scripts/train_predictor.py --model mlp --name mlp_predictor
python scripts/train_predictor.py --model cnn --name cnn_predictor
python scripts/train_predictor.py --model mpnn --name cons5

# Compare predictor performance
python scripts/compare_predictors.py --test_data ./data/test_data.pt
```

**Output**: MSE, MAE, inference speed, model size for each predictor

### Step 2: Evaluate OED Methods (Paper Table 2)

Compare different experimental design strategies:

```bash
# Set the predictor to use (for iNN/NN methods)
export MOCU_MODEL_NAME=cons5

# Run OED evaluation
python scripts/evaluate.py

# Visualize results
python scripts/visualize.py
```

**Output**: MOCU curves, terminal MOCU, time complexity for each method

### Step 3: Train and Test DAD-MOCU

Add your new method to the comparison:

```bash
# Generate DAD training data
python scripts/generate_dad_data.py --N 5 --num_episodes 1000

# Train DAD policy
python scripts/train_dad_policy.py --data_path ./data/dad_training_data/ --method reinforce

# Evaluate with DAD included
python scripts/evaluate.py  # DAD is already in method list
```

**Output**: DAD-MOCU vs. all baselines

## Key Features

✅ **Two-level structure**: Clear separation between MOCU prediction and OED selection  
✅ **Unified interface**: All methods inherit from `OEDMethod` base class  
✅ **Modular design**: Easy to add new methods and predictors  
✅ **CUDA acceleration**: 100-1000× speedup via PyTorch CUDA (no PyCUDA required)  
✅ **Deep Adaptive Design**: Learn sequential policies to minimize terminal MOCU  
✅ **Stable and safe**: No CUDA context conflicts (PyTorch-only implementation)  

## Technical Details

### MOCU Computation

The MOCU (Multi-Objective Control Uncertainty) computation is implemented in `src/core/mocu.py` using PyTorch CUDA acceleration. This replaces the original PyCUDA implementation for better compatibility and stability.

**Key advantages**:
- No CUDA context conflicts (safe to use with PyTorch training)
- Full CUDA acceleration (runs on GPU)
- Easier to maintain (standard PyTorch operations)
- Automatic device management (CPU fallback if CUDA unavailable)

### DAD Training

DAD (Deep Adaptive Design) uses REINFORCE policy gradient with MPNN predictor for fast MOCU estimation during training. The policy network learns to make sequential experimental selections to minimize terminal MOCU.

**Training process**:
1. Generate trajectories using an expert method (e.g., iNN) or random policy
2. Train policy network using REINFORCE with MPNN-predicted MOCU as reward
3. Evaluate trained policy against other OED methods

## Citation

If you use this code, please cite the original AccelerateOED paper and the DAD framework.

## License

See LICENSE file for details.

## Summary of Architecture

### Clean Architecture

| Component | Location | Purpose |
|-----------|----------|---------|
| **OED Methods** | `src/methods/` | Selection strategies (iNN, NN, ODE, ENTROPY, RANDOM, DAD) |
| **MOCU Predictors** | `src/models/predictors/` | Neural networks for MOCU prediction |
| **DAD Policy** | `src/models/policy_networks.py` | Sequential decision policy |
| **Core MOCU** | `src/core/mocu.py` | PyTorch CUDA-accelerated computation |
| **Scripts** | `scripts/` | Data generation, training, evaluation |

### Evaluation Capabilities

✅ **Predictor Comparison** (Table 1 in paper):
```bash
python scripts/compare_predictors.py
```
Compares MLP, CNN, MPNN+ on MSE, MAE, inference speed, model size

✅ **OED Method Comparison** (Table 2 in paper):
```bash
python scripts/evaluate.py
```
Compares iNN, NN, ODE, ENTROPY, RANDOM, DAD on terminal MOCU, time complexity

✅ **Visualization**:
```bash
python scripts/visualize.py
```
Generates MOCU curves, time complexity plots, performance tables
