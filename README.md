# AccelerateOED

Code for paper **"Neural Message Passing for Objective-Based Uncertainty Quantification and Optimal Experimental Design"**

This repository implements MOCU-OED framework using neural message passing to accelerate optimal experimental design for coupled oscillator systems by **100-1000×**.

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
AccelerateOED/
├── configs/                      # Configuration files
│   ├── fast_config.yaml         # 🚀 Component verification (3-5 min)
│   ├── N5_config.yaml           # 5-oscillator system (5-10 hours)
│   ├── N7_config.yaml           # 7-oscillator system (15-20 hours)
│   └── N9_config.yaml           # 9-oscillator system (24-30 hours)
├── run.sh                       # Main automation script
├── scripts/
│   ├── data_generation.py      # Generate & convert dataset
│   ├── training.py             # Train GNN model
│   ├── evaluation.py           # Run OED experiments
│   └── visualization.py        # Generate plots
├── src/                         # Source code
├── data/                        # [Auto-created] Datasets
├── models/                      # [Auto-created] Trained models
└── results/                     # [Auto-created] Results
```

## Quick Start

### Component Verification (3-5 minutes) ⚡

**Quickly test if all components work** (minimal data, NOT for actual experiments):

```bash
bash run.sh configs/fast_config.yaml
```

| Step | Time | What it does |
|------|------|--------------|
| Data Generation | ~2 min | 300 samples → ~150 valid (after sync filtering) |
| Training | ~1 min | 10 epochs on ~100 samples |
| Experiments | ~2 min | 1 simulation, 3 iterations, 2 methods |
| Visualization | ~5 sec | Generate plots |

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

**Note:** The script automatically updates `N_global` in the CUDA code to match your system size (e.g., N=5 → N_global=6) and continues execution seamlessly.
