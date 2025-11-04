# DAD (Deep Adaptive Design) Complete Design Specification

## Overview

This document provides a complete specification for the DAD method implementation, including network architecture, training data generation, training loop, and evaluation. This design replaces PyCUDA with `torchdiffeq` for ODE solving in all processes **except** the initial MPNN training data generation step, which still uses PyCUDA for maximum performance.

---

## Table of Contents

1. [Network Structure](#1-network-structure)
2. [Training Data Generation](#2-training-data-generation)
3. [Training Loop (REINFORCE)](#3-training-loop-reinforce)
4. [Evaluation](#4-evaluation)
5. [PyCUDA → torchdiffeq Migration](#5-pycuda--torchdiffeq-migration)
6. [Implementation Details](#6-implementation-details)

---

## 1. Network Structure

### 1.1 Policy Network Architecture

The DAD policy network (`DADPolicyNetwork`) is a graph neural network that learns to select optimal experiments sequentially.

```
┌─────────────────────────────────────────────────────────────┐
│                    DAD Policy Network                        │
└─────────────────────────────────────────────────────────────┘

INPUT: Current State
├── w: Natural frequencies [N]
├── a_lower: Lower coupling bounds [N, N]
├── a_upper: Upper coupling bounds [N, N]
└── history: Past (action, observation) pairs [(i, j, obs), ...]

    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Graph State Encoder (GNN)                                │
├─────────────────────────────────────────────────────────────┤
│ • Node features: w[i] → Linear(1, encoding_dim)             │
│ • Edge features: [a_lower[i,j], a_upper[i,j]]               │
│ • Message passing: NNConv with GRU (3 iterations)           │
│ • Graph pooling: Set2Set → [mean, max] → MLP                │
│ • Output: State embedding [hidden_dim]                      │
└─────────────────────────────────────────────────────────────┘

    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. History Encoder (LSTM)                                    │
├─────────────────────────────────────────────────────────────┤
│ • Embed: (i, j) → embedding, obs → embedding               │
│ • LSTM: 2 layers, hidden_dim                                │
│ • Output: History embedding [hidden_dim]                    │
└─────────────────────────────────────────────────────────────┘

    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Action Decoder                                           │
├─────────────────────────────────────────────────────────────┤
│ • Concatenate: [state_embed, history_embed]                 │
│ • MLP: Linear(hidden_dim*2 → hidden_dim → num_actions)      │
│ • Mask: Zero out already-observed pairs                     │
│ • Softmax: Action probabilities [num_actions]               │
│ • Output: Selected (i, j) pair (greedy)                     │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Network Components

#### Graph State Encoder
- **Input**: Graph with nodes (oscillators) and edges (coupling bounds)
- **Node features**: `w[i]` (natural frequency) → `[encoding_dim]`
- **Edge features**: `[a_lower[i,j], a_upper[i,j]]` → `[encoding_dim × encoding_dim]` via edge NN
- **Message passing**: 3 iterations of NNConv + GRU
- **Pooling**: Set2Set (mean + max pooling)
- **Output dimension**: `hidden_dim` (default: 64)

#### History Encoder
- **Input**: Sequence of `(i, j, observation)` tuples
- **Embedding**: 
  - `i, j` → `history_embed` (embedding size: `hidden_dim // 4`)
  - `observation` → `obs_embed` (embedding size: `hidden_dim // 4`)
- **LSTM**: 2 layers, `hidden_dim` units
- **Output dimension**: `hidden_dim`

#### Action Decoder
- **Input**: Concatenated `[state_embed, history_embed]` → `[hidden_dim * 2]`
- **MLP**: 
  ```
  Linear(hidden_dim*2 → hidden_dim) → ReLU → Dropout(0.1)
  → Linear(hidden_dim → hidden_dim) → ReLU
  → Linear(hidden_dim → num_actions)
  ```
- **Masking**: Zero out probabilities for already-observed pairs
- **Selection**: Greedy (argmax) during evaluation, sampling during training

### 1.3 Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 64 | Hidden dimension for LSTM and MLPs |
| `encoding_dim` | 32 | Dimension for graph embeddings |
| `num_message_passing` | 3 | Number of GNN message passing iterations |
| `dropout` | 0.1 | Dropout rate |
| `num_lstm_layers` | 2 | Number of LSTM layers |

---

## 2. Training Data Generation

### 2.1 Data Generation Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Generate MPNN Training Data (PyCUDA)                │
├─────────────────────────────────────────────────────────────┤
│ • Use PyCUDA for MOCU computation (fast, parallel)          │
│ • Generate (w, a_lower, a_upper, MOCU) pairs                │
│ • Type 1: Per-edge multiplier bounds                        │
│ • Type 2: Per-oscillator multiplier bounds                 │
│ • Output: MPNN training dataset                             │
└─────────────────────────────────────────────────────────────┘

    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Train MPNN Predictor                                │
├─────────────────────────────────────────────────────────────┤
│ • Train MPNN to predict MOCU from (w, a_lower, a_upper)     │
│ • Use PyTorch (torchdiffeq for ODE solving if needed)       │
│ • Output: Trained MPNN model                                │
└─────────────────────────────────────────────────────────────┘

    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Generate DAD Training Trajectories                  │
├─────────────────────────────────────────────────────────────┤
│ • Generate random systems (w, a_lower_init, a_upper_init)   │
│ • Use random expert policy to select K experiments          │
│ • Simulate experiments using torchdiffeq (not PyCUDA)       │
│ • Update bounds after each experiment                       │
│ • Compute terminal MOCU using MPNN predictor                │
│ • Output: Trajectories with pre-computed terminal_MOCU       │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 DAD Trajectory Structure

Each trajectory contains:

```python
trajectory = {
    'w': np.array([N]),                    # Natural frequencies
    'a_true': np.array([N, N]),           # Ground truth coupling matrix
    'a_lower_init': np.array([N, N]),     # Initial lower bounds
    'a_upper_init': np.array([N, N]),     # Initial upper bounds
    'actions': [(i1, j1), (i2, j2), ...], # Selected experiments
    'observations': [obs1, obs2, ...],     # Sync/non-sync observations
    'terminal_MOCU': float,                # Pre-computed terminal MOCU
    'states': [                            # State sequence
        (a_lower_0, a_upper_0),
        (a_lower_1, a_upper_1),
        ...
    ]
}
```

### 2.3 Data Generation Details

#### System Generation
- **Natural frequencies**: `w[i] = 12 * (0.5 - random())` (uniform in [-6, 6])
- **Initial bounds**: Random multiplier approach (uncertainty = 0.3 * random())
- **True coupling**: Random within bounds

#### Experiment Simulation
- **Use torchdiffeq** (not PyCUDA) to solve Kuramoto ODE:
  ```python
  dθ/dt = w + Σ a[i,j] * sin(θ[j] - θ[i])
  ```
- **Observation**: 1 if synchronized, 0 if not
- **Bound update**: Based on observation (sync → raise lower bound, non-sync → lower upper bound)

#### Terminal MOCU Computation
- **Pre-compute** using MPNN predictor during data generation
- **Avoids** MPNN calls during training (faster, no CUDA conflicts)
- **Alternative**: Can compute during training if needed

### 2.4 Training Data Requirements

| Parameter | Recommended Value | Description |
|-----------|------------------|-------------|
| `num_episodes` | 1000-5000 | Number of trajectories |
| `K` | 4-10 | Experiments per trajectory |
| `use_precomputed_mocu` | `true` | Pre-compute terminal MOCU |

---

## 3. Training Loop (REINFORCE)

### 3.1 REINFORCE Algorithm

REINFORCE is a policy gradient method that directly optimizes terminal MOCU as the reward signal.

```
FOR epoch in range(num_epochs):
    FOR trajectory in trajectories:
        # 1. Policy Rollout: Generate new trajectory using current policy
        log_probs = []
        a_lower, a_upper = a_lower_init, a_upper_init
        
        FOR step in range(K):
            # Policy selects action
            state_data = create_state_data(w, a_lower, a_upper)
            action_logits, action_probs = policy_net(state_data, history, mask)
            action_idx = sample(action_probs)  # Sample (training) or argmax (eval)
            log_prob = log(action_probs[action_idx])
            log_probs.append(log_prob)
            
            # Simulate experiment (use torchdiffeq, not PyCUDA)
            observation = simulate_experiment(a_true, action_i, action_j, w, h, M)
            update_bounds(a_lower, a_upper, action_i, action_j, observation)
            history.append((action_i, action_j, observation))
        
        # 2. Compute Reward: Terminal MOCU (negative = reward)
        terminal_MOCU = trajectory['terminal_MOCU']  # Pre-computed
        reward = -terminal_MOCU  # Minimize MOCU = Maximize reward
        
        # 3. REINFORCE Update: Update policy to maximize reward
        loss = -sum(log_prob * (reward - baseline) for log_prob in log_probs)
        loss.backward()
        optimizer.step()
```

### 3.2 Loss Function

```python
# REINFORCE with baseline
rewards = []
log_probs_list = []

for trajectory in trajectories:
    # ... rollout ...
    reward = -terminal_MOCU
    rewards.append(reward)
    log_probs_list.append(log_probs)

# Compute baseline (moving average or value function)
baseline = np.mean(rewards)  # Simple baseline
# OR: baseline = value_function(state)  # Learned baseline

# REINFORCE loss
loss = 0
for log_probs, reward in zip(log_probs_list, rewards):
    for log_prob in log_probs:
        loss -= log_prob * (reward - baseline)
```

### 3.3 Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 0.001 | Adam optimizer learning rate |
| `epochs` | 100-300 | Number of training epochs |
| `gamma` | 0.99 | Discount factor (for future rewards, if used) |
| `use_baseline` | `true` | Use baseline to reduce variance |
| `batch_size` | N/A | Not used for REINFORCE (sequential) |

### 3.4 Training Tips

1. **Pre-compute terminal MOCU**: Avoids MPNN calls during training (faster, no conflicts)
2. **Use baseline**: Reduces variance in policy gradient estimates
3. **Monitor loss and reward**: Loss should decrease, reward should increase (less negative)
4. **Early stopping**: Stop if validation MOCU doesn't improve for 10 epochs
5. **Curriculum learning**: Start with easy trajectories (K=2), progress to hard (K=10)

---

## 4. Evaluation

### 4.1 Evaluation Process

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Load Trained Policy                                      │
├─────────────────────────────────────────────────────────────┤
│ • Load checkpoint: models/{config}/{timestamp}/dad_policy_N{N}.pth
│ • Initialize policy network with saved weights              │
│ • Set to eval mode (no dropout, no sampling)               │
└─────────────────────────────────────────────────────────────┘

    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Sequential Experiment Selection                          │
├─────────────────────────────────────────────────────────────┤
│ FOR iteration in range(update_cnt):
│    1. Policy selects next experiment (i, j) [greedy]
│    2. Simulate experiment (use torchdiffeq)
│    3. Update bounds based on observation
│    4. Predict MOCU using MPNN predictor
│    5. Record MOCU value
│ END FOR
└─────────────────────────────────────────────────────────────┘

    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Compare with Baselines                                    │
├─────────────────────────────────────────────────────────────┤
│ • Compare MOCU curves: DAD vs. iNN, RANDOM, ODE, etc.
│ • Metrics: Final MOCU, MOCU reduction, convergence speed
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Evaluation Metrics

- **Final MOCU**: Terminal MOCU after all experiments
- **MOCU reduction**: `(initial_MOCU - final_MOCU) / initial_MOCU`
- **Convergence speed**: Number of iterations to reach 50% reduction
- **Comparison**: DAD vs. iNN, RANDOM, ODE, ENTROPY methods

### 4.3 Evaluation Configuration

```yaml
experiment:
  methods: ["DAD", "iNN", "RANDOM", "ODE", "ENTROPY"]
  update_count: 10  # Number of experiments
  num_simulations: 20  # Number of test systems

dad:
  policy_path: "models/{config}/{timestamp}/dad_policy_N{N}.pth"
  use_greedy: true  # Deterministic selection (not sampling)
```

---

## 5. PyCUDA → torchdiffeq Migration

### 5.1 Migration Strategy

**KEEP PyCUDA for:**
- ✅ Initial MPNN training data generation (`scripts/generate_mocu_data.py`)
  - Reason: Maximum performance for large-scale data generation
  - Runs in separate process (no PyTorch CUDA conflicts)

**REPLACE PyCUDA with torchdiffeq for:**
- ❌ ODE/RANDOM/ENTROPY methods (`src/methods/ode.py`, `src/methods/random.py`, `src/methods/entropy.py`)
- ❌ DAD training data generation (`scripts/generate_dad_data.py`)
- ❌ DAD evaluation (`src/methods/dad_mocu.py`)
- ❌ Initial MOCU computation in evaluation (`scripts/evaluate.py`)
- ❌ Iterative MOCU computation in `base.py` (`src/methods/base.py`)

### 5.2 torchdiffeq Implementation

#### Kuramoto ODE System

```python
import torch
from torchdiffeq import odeint

class KuramotoODE(torch.nn.Module):
    """
    Kuramoto oscillator system: dθ/dt = w + Σ a[i,j] * sin(θ[j] - θ[i])
    """
    def __init__(self, w, a):
        super().__init__()
        self.w = torch.tensor(w, dtype=torch.float32)
        self.a = torch.tensor(a, dtype=torch.float32)
        self.N = len(w)
    
    def forward(self, t, theta):
        """
        Args:
            t: Time (scalar, not used but required by torchdiffeq)
            theta: Phase angles [N]
        
        Returns:
            dtheta/dt: Phase derivatives [N]
        """
        # Expand theta for broadcasting
        theta_i = theta.unsqueeze(1)  # [N, 1]
        theta_j = theta.unsqueeze(0)   # [1, N]
        
        # Compute coupling term: Σ a[i,j] * sin(θ[j] - θ[i])
        coupling = torch.sum(self.a * torch.sin(theta_j - theta_i), dim=1)
        
        # dθ/dt = w + coupling
        dtheta_dt = self.w + coupling
        
        return dtheta_dt

def solve_kuramoto_ode(w, a, h, M, device='cuda'):
    """
    Solve Kuramoto ODE using torchdiffeq.
    
    Args:
        w: Natural frequencies [N]
        a: Coupling matrix [N, N]
        h: Time step
        M: Number of time steps
        device: 'cuda' or 'cpu'
    
    Returns:
        theta_trajectory: [M, N] phase angles over time
    """
    # Create ODE system
    ode_system = KuramotoODE(w, a).to(device)
    
    # Initial condition: all phases start at 0
    theta0 = torch.zeros(len(w), dtype=torch.float32, device=device)
    
    # Time points
    t = torch.linspace(0, M * h, M, dtype=torch.float32, device=device)
    
    # Solve ODE
    theta_trajectory = odeint(ode_system, theta0, t, method='rk4')
    
    return theta_trajectory.cpu().numpy()

def check_synchronization(theta_trajectory, M):
    """
    Check if system is synchronized based on phase trajectory.
    
    Args:
        theta_trajectory: [M, N] phase angles
        M: Number of time steps
    
    Returns:
        is_synchronized: 1 if synchronized, 0 if not
    """
    # Use second half of trajectory to check stability
    second_half = theta_trajectory[M//2:, :]
    
    # Compute phase differences
    diff_t = np.diff(second_half, axis=0)
    
    # Check if all differences are small (synchronized)
    tol = np.max(diff_t) - np.min(diff_t)
    
    return 1 if tol <= 1e-3 else 0
```

### 5.3 MOCU Computation with torchdiffeq

Replace PyCUDA-based MOCU computation with torchdiffeq-based version:

```python
def MOCU_torchdiffeq(K_max, w, N, h, M, T, a_lower, a_upper, seed=0, device='cuda'):
    """
    Compute MOCU using torchdiffeq for ODE solving.
    
    Args:
        K_max: Number of Monte Carlo samples
        w: Natural frequencies [N]
        N: Number of oscillators
        h: Time step
        M: Number of time steps
        T: Time horizon (not used, kept for compatibility)
        a_lower: Lower bounds [N, N]
        a_upper: Upper bounds [N, N]
        seed: Random seed
        device: 'cuda' or 'cpu'
    
    Returns:
        MOCU value (float)
    """
    import torch
    from torchdiffeq import odeint
    import numpy as np
    
    # Set random seed
    if seed != 0:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Generate random coupling matrices
    a_samples = []
    for k in range(K_max):
        a = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                a_ij = np.random.uniform(a_lower[i, j], a_upper[i, j])
                a[i, j] = a_ij
                a[j, i] = a_ij
        a_samples.append(a)
    
    # Solve ODE for each sample and find critical coupling
    a_star_values = []
    for a in a_samples:
        # Binary search for critical coupling (same logic as PyCUDA version)
        a_star = binary_search_critical_coupling(w, a_lower, a_upper, a, h, M, device)
        a_star_values.append(a_star)
    
    # Compute MOCU
    a_star_values = np.array(a_star_values)
    if K_max >= 1000:
        # Filter outliers (same as PyCUDA version)
        sorted_vals = np.sort(a_star_values)
        ll = int(K_max * 0.005)
        uu = int(K_max * 0.995)
        filtered_vals = sorted_vals[ll-1:uu]
        a_star_max = np.max(filtered_vals)
        MOCU_val = np.sum(a_star_max - filtered_vals) / (K_max * 0.99)
    else:
        a_star_max = np.max(a_star_values)
        MOCU_val = np.sum(a_star_max - a_star_values) / K_max
    
    return float(MOCU_val)

def binary_search_critical_coupling(w, a_lower, a_upper, a_sample, h, M, device='cuda'):
    """
    Binary search for critical coupling strength using torchdiffeq.
    """
    # Implementation similar to PyCUDA version but using torchdiffeq
    # ... (details omitted for brevity)
    pass
```

### 5.4 Files to Modify

1. **`src/core/mocu_torchdiffeq.py`** (NEW): Create torchdiffeq-based MOCU computation
2. **`src/methods/ode.py`**: Replace `MOCU_pycuda` with `MOCU_torchdiffeq`
3. **`src/methods/random.py`**: Replace PyCUDA MOCU with torchdiffeq
4. **`src/methods/entropy.py`**: Replace PyCUDA MOCU with torchdiffeq
5. **`src/methods/base.py`**: Replace iterative MOCU computation with torchdiffeq
6. **`scripts/generate_dad_data.py`**: Use torchdiffeq for experiment simulation
7. **`scripts/evaluate.py`**: Replace initial MOCU with torchdiffeq
8. **`scripts/train_dad_policy.py`**: Remove PyCUDA fallback (already uses MPNN)

### 5.5 Benefits of torchdiffeq

1. **No CUDA context conflicts**: Fully compatible with PyTorch CUDA
2. **Automatic differentiation**: Can be used for gradient-based optimization
3. **Flexible solvers**: Multiple ODE solvers (RK4, adaptive, etc.)
4. **GPU acceleration**: Native PyTorch CUDA support
5. **Easier maintenance**: Single ecosystem (PyTorch) instead of PyCUDA

---

## 6. Implementation Details

### 6.1 Code Structure

```
src/
├── core/
│   ├── mocu_pycuda.py          # PyCUDA MOCU (ONLY for MPNN data generation)
│   ├── mocu_torchdiffeq.py     # torchdiffeq MOCU (NEW)
│   └── sync_detection.py       # Synchronization detection (CPU)
├── methods/
│   ├── base.py                 # Base class (uses torchdiffeq MOCU)
│   ├── dad_mocu.py             # DAD method (uses torchdiffeq)
│   ├── ode.py                  # ODE method (uses torchdiffeq)
│   ├── random.py               # RANDOM method (uses torchdiffeq)
│   └── entropy.py              # ENTROPY method (uses torchdiffeq)
├── models/
│   ├── policy_networks.py      # DAD policy network
│   └── predictors/
│       └── predictor_utils.py  # MPNN predictor utilities
└── ...
```

### 6.2 Dependencies

```yaml
# Required packages
torch: >= 2.0.0
torchdiffeq: >= 0.2.0
torch-geometric: >= 2.0.0
numpy: >= 1.20.0

# Optional (for MPNN data generation only)
pycuda: >= 2021.1  # ONLY for generate_mocu_data.py
```

### 6.3 Configuration Example

```yaml
# configs/N5_config.yaml
system:
  N: 5
  deltaT: 0.00625
  MReal: 800
  TReal: 5.0

experiment:
  update_count: 10
  methods: ["DAD", "iNN", "RANDOM", "ODE", "ENTROPY"]

mpnn_data:
  K_max: 20480
  samples_per_type: 5000
  use_pycuda: true  # ONLY for this step

dad_data:
  num_episodes: 1000
  K: 10
  use_precomputed_mocu: true
  use_torchdiffeq: true  # Use torchdiffeq for simulation

training:
  method: "reinforce"
  epochs: 200
  learning_rate: 0.001
  hidden_dim: 64
  encoding_dim: 32

mocu_computation:
  backend: "torchdiffeq"  # Use torchdiffeq for all MOCU computation
  device: "cuda"
```

### 6.4 Workflow

```bash
# Step 1: Generate MPNN training data (PyCUDA)
bash scripts/bash/step1_generate_mocu_data.sh configs/N5_config.yaml

# Step 2: Train MPNN predictor
bash scripts/bash/step2_train_mpnn.sh configs/N5_config.yaml

# Step 3: Generate DAD training data (torchdiffeq)
bash scripts/bash/step3_generate_dad_data.sh configs/N5_config.yaml

# Step 4: Train DAD policy
bash scripts/bash/step3_train_dad.sh configs/N5_config.yaml

# Step 5: Evaluate (torchdiffeq)
bash scripts/bash/step4_evaluate.sh configs/N5_config.yaml
```

---

## Summary

This design provides:

1. **Complete DAD architecture**: Graph encoder + History encoder + Action decoder
2. **Training data pipeline**: MPNN data (PyCUDA) → MPNN training → DAD data (torchdiffeq) → DAD training
3. **REINFORCE training**: Policy gradient with terminal MOCU as reward
4. **Evaluation**: Sequential experiment selection with MOCU tracking
5. **PyCUDA → torchdiffeq migration**: Replace PyCUDA everywhere except initial MPNN data generation

**Key Benefits:**
- ✅ No CUDA context conflicts (torchdiffeq compatible with PyTorch)
- ✅ Flexible ODE solving (multiple solvers, adaptive steps)
- ✅ Easier maintenance (single ecosystem)
- ✅ GPU acceleration (native PyTorch CUDA)
- ✅ Automatic differentiation (for future gradient-based optimization)

**Next Steps:**
1. Implement `src/core/mocu_torchdiffeq.py`
2. Update all methods to use torchdiffeq MOCU
3. Update DAD data generation to use torchdiffeq
4. Test and verify MOCU values match PyCUDA version
5. Run full pipeline with N5_config.yaml

