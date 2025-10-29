# AccelerateOED + DAD-MOCU

Code for paper **"Neural Message Passing for Objective-Based Uncertainty Quantification and Optimal Experimental Design"** with Deep Adaptive Design extension.

This repository implements MOCU-OED framework using neural message passing to accelerate optimal experimental design for coupled oscillator systems by **100-1000×**, plus a new Deep Adaptive Design (DAD) method for sequential experimental design.

## Environment Requirements

- **OS**: Linux (Ubuntu 22.04+)
- **Python**: 3.10
- **GPU**: NVIDIA GPU with CUDA 12.1+
- **Hardware**: Tested on GeForce RTX 4080 

## Installation

### 1. Create Environment

```bash
conda create -n mocu python=3.10
conda activate mocu
```

### 2. Install CUDA Toolkit

```bash
conda install -c nvidia cuda-toolkit=12.1
nvcc --version
```

### 3. Install PyCUDA

```bash
# System dependencies (Ubuntu/Debian)
sudo apt-get install -y build-essential python3-dev libboost-python-dev libboost-thread-dev

# PyCUDA
pip install pycuda
```

### 4. Install PyTorch & PyTorch Geometric

```bash
# PyTorch with CUDA 12.1
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# PyTorch Geometric
pip install torch-geometric==2.6.1
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

```
dad_mocu_kuramoto_v2/
├── configs/                      # Configuration files
│   ├── fast_config.yaml         # Quick verification (3-5 min)
│   ├── N5_config.yaml           # 5-oscillator system
│   ├── N7_config.yaml           # 7-oscillator system
│   └── N9_config.yaml           # 9-oscillator system
│
├── scripts/                      # Execution scripts
│   ├── generate_mocu_data.py    # Generate MOCU predictor training data
│   ├── generate_dad_data.py     # Generate DAD policy training data
│   ├── train_mocu_predictor.py  # Train MPNN+ predictor (for iNN/NN)
│   ├── train_dad_policy.py      # Train DAD policy network
│   ├── evaluate_predictors.py   # Compare MOCU predictors (MSE, speed) 📊
│   ├── evaluation.py            # Run OED experiments (all methods) ⭐
│   └── visualization.py         # Generate plots
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
│   │   │   ├── all_predictors.py  # Unified: MLP, CNN, MPNN+, Sampling, Ensemble
│   │   │   ├── legacy_baselines.py # Original CNN/MLP (2023 paper)
│   │   │   └── legacy_mpnn_train.py # Original MPNN training (2023 paper)
│   │   └── policy_networks.py     # DAD policy network
│   │
│   ├── core/                       # Core computation
│   │   ├── mocu_cuda.py           # CUDA-accelerated MOCU
│   │   └── sync_detection.py
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

### Super Quick Test (< 1 minute) ⚡⚡

Test core functionality without training (no model needed):

```bash
conda activate mocu
python quick_test.py
```

This tests:
- ✓ MOCU computation works
- ✓ RANDOM method works (no model needed)
- ✓ Code structure is correct

### Component Verification (3-5 minutes) ⚡

Quickly test if all components work (minimal data, NOT for actual experiments):

```bash
# Option 1: Automated (using run.sh)
conda activate mocu
bash run.sh configs/fast_config.yaml

# Option 2: Manual steps
conda activate mocu

# Step 1: Generate minimal data
python scripts/generate_mocu_data.py --N 5 --samples_per_type 150 --K_max 1024 --train_size 100

# Step 2: Train for 10 epochs
python scripts/train_mocu_predictor.py --data_path ./data/ --name fast_test --epochs 10

# Step 3: Test with trained model
python quick_test.py
```

| Step | Time | What it does |
|------|------|--------------|
| Data Generation | ~2 min | 300 samples → ~150 valid (after sync filtering) |
| Training | ~1 min | 10 epochs on ~100 samples |
| Testing | ~30 sec | Test RANDOM + iNN methods |
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

**Location**: `src/models/predictors/all_predictors.py`

Predict MOCU values from coupling bounds:
- **MLP**, **CNN** - Baseline neural networks
- **MPNN+** - Message Passing NN with ranking constraint
- **Sampling-based** - Ground truth (Monte Carlo)
- **Ensemble** - Combine multiple models

**Evaluate predictors**:
```bash
python scripts/evaluate_predictors.py --test_data ./data/test_data.pt
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
- **ODE**, **iODE** - Sampling-based (exact but slow)
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
python scripts/train_mocu_predictor.py --data_path ./data/ --name cons5
```

**Note**: Sampling-based methods (ODE, iODE) don't need training - they compute MOCU directly.

### 2. Train DAD Policy (Optional)

```bash
# Generate DAD trajectories using iNN as expert
python scripts/generate_dad_data.py --N 5 --num_episodes 1000 --expert_method iNN

# Train DAD policy network
python scripts/train_dad_policy.py --data_path ./data/dad_training_data/ --name dad_policy_N5
```

### 3. Evaluate All Methods

```bash
export MOCU_MODEL_NAME=cons5
python scripts/evaluation.py
```

This evaluates ALL methods: iNN, NN, ODE, ENTROPY, RANDOM, DAD

## Complete Research Workflow

### Step 1: Evaluate MOCU Predictors (Paper Table 1)

Compare different prediction models:

```bash
# Generate test data
python scripts/generate_mocu_data.py --N 5 --samples_per_type 5000

# Train different predictors
python scripts/train_mocu_predictor.py --model mlp --name mlp_predictor
python scripts/train_mocu_predictor.py --model cnn --name cnn_predictor
python scripts/train_mocu_predictor.py --model mpnn --name cons5

# Compare predictor performance
python scripts/evaluate_predictors.py --test_data ./data/test_data.pt
```

**Output**: MSE, MAE, inference speed, model size for each predictor

### Step 2: Evaluate OED Methods (Paper Table 2)

Compare different experimental design strategies:

```bash
# Set the predictor to use (for iNN/NN methods)
export MOCU_MODEL_NAME=cons5

# Run OED evaluation
python scripts/evaluation.py

# Visualize results
python scripts/visualization.py
```

**Output**: MOCU curves, terminal MOCU, time complexity for each method

### Step 3: Train and Test DAD-MOCU

Add your new method to the comparison:

```bash
# Generate DAD training data
python scripts/generate_dad_data.py --N 5 --num_episodes 1000

# Train DAD policy
python scripts/train_dad_policy.py --data_path ./data/dad_training_data/

# Evaluate with DAD included
python scripts/evaluation.py  # DAD is already in method list
```

**Output**: DAD-MOCU vs. all baselines

## Key Features

✅ **Two-level structure**: Clear separation between MOCU prediction and OED selection
✅ **Unified interface**: All methods inherit from `OEDMethod` base class
✅ **Modular design**: Easy to add new methods and predictors
✅ **CUDA acceleration**: 100-1000× speedup via PyCUDA
✅ **Deep Adaptive Design**: Learn sequential policies to minimize terminal MOCU

## Citation

If you use this code, please cite the original AccelerateOED paper and the DAD framework.

## License

See LICENSE file for details.

## Summary of Changes

### Cleaned & Organized Structure

**Removed clutter**:
- ✅ Removed 14 MD documentation files - Only README.md remains
- ✅ Removed `src/strategies/` folder - Replaced by unified `src/methods/`
- ✅ Removed redundant files - Test files and temporary artifacts

**Renamed for clarity**:
- `data_generation.py` → `generate_mocu_data.py`
- `training.py` → `train_mocu_predictor.py`
- `evaluation_unified.py` → `evaluation.py`

**Organized models/** directory:
- Created `models/predictors/` subdirectory
- `all_predictors.py` - Unified implementations (use this)
- `legacy_baselines.py` - Original CNN/MLP (backward compatibility)
- `legacy_mpnn_train.py` - Original MPNN training (backward compatibility)

### Clean Architecture

| Component | Location | Purpose |
|-----------|----------|---------|
| **OED Methods** | `src/methods/` | Selection strategies (iNN, NN, ODE, ENTROPY, RANDOM, DAD) |
| **MOCU Predictors** | `src/models/predictors/` | Neural networks for MOCU prediction |
| **DAD Policy** | `src/models/policy_networks.py` | Sequential decision policy |
| **Core MOCU** | `src/core/mocu_cuda.py` | CUDA-accelerated computation |
| **Scripts** | `scripts/` | Data generation, training, evaluation |

### Evaluation Capabilities

✅ **Predictor Comparison** (Table 1 in paper):
```bash
python scripts/evaluate_predictors.py
```
Compares MLP, CNN, MPNN+ on MSE, MAE, inference speed, model size

✅ **OED Method Comparison** (Table 2 in paper):
```bash
python scripts/evaluation.py
```
Compares iNN, NN, ODE, ENTROPY, RANDOM, DAD on terminal MOCU, time complexity

✅ **Visualization**:
```bash
python scripts/visualization.py
```
Generates MOCU curves, time complexity plots, performance tables

### Key Benefits
✅ **No redundancy** - Single source of truth for each component  
✅ **Standard structure** - Follows Python best practices  
✅ **Easy to navigate** - Clear organization  
✅ **Production-ready** - Professional codebase  

