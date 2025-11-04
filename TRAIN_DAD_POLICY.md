# How to Train DAD Policy Network

## Overview

The DAD policy network is trained using **REINFORCE** (Reinforcement Learning) to learn sequential experiment selection that minimizes terminal MOCU.

---

## Prerequisites

Before training DAD policy, you need:

1. ✅ **Trained MPNN Predictor** (required for computing terminal MOCU)
2. ✅ **DAD Training Data** (trajectories with pre-computed terminal MOCU)

---

## Quick Start

### Option 1: Automated (Recommended)

```bash
# This will automatically:
# 1. Generate DAD data (if missing)
# 2. Train DAD policy
bash scripts/bash/step3_train_dad.sh configs/fast_config.yaml
```

### Option 2: Manual Step-by-Step

```bash
# Step 1: Generate DAD training data
python3 scripts/generate_dad_data.py \
    --N 5 \
    --K 4 \
    --num-episodes 1000 \
    --use-mpnn-predictor \
    --mpnn-model-name "fast_test_11012025_212842" \
    --output-dir data/dad_trajectories/

# Step 2: Train DAD policy
python3 scripts/train_dad_policy.py \
    --data-path data/dad_trajectories/dad_trajectories_N5_K4_1000.pth \
    --method reinforce \
    --name "dad_policy_N5" \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.001 \
    --hidden-dim 64 \
    --encoding-dim 32 \
    --output-dir models/dad_policies/ \
    --use-predicted-mocu
```

---

## Command Line Arguments

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--data-path` | Path to DAD trajectory data file | `data/dad_trajectories/dad_trajectories_N5_K4_1000.pth` |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--method` | `"reinforce"` | Training method: `"reinforce"` (RL) or `"imitation"` (behavior cloning) |
| `--epochs` | `100` | Number of training epochs |
| `--batch-size` | `32` | Batch size (not used for REINFORCE, used for imitation) |
| `--lr` | `0.001` | Learning rate |
| `--hidden-dim` | `64` | Hidden dimension for LSTM and MLPs |
| `--encoding-dim` | `32` | Dimension for graph embeddings |
| `--device` | `"cuda"` | Device: `"cuda"` or `"cpu"` |
| `--output-dir` | `"../models/"` | Directory to save trained model |
| `--name` | `"dad_policy"` | Model name (saved as `{name}.pth`) |
| `--use-predicted-mocu` | `False` | Use MPNN predictor during training (if terminal_MOCU not pre-computed) |

---

## Training Process Explained

### 1. **REINFORCE Training** (Recommended)

REINFORCE is a policy gradient method that:
- Uses terminal MOCU directly as the reward signal
- Updates policy to maximize reward (minimize MOCU)
- Generates new trajectories on-the-fly during training

**Training Loop**:

```python
for epoch in range(epochs):
    for trajectory in trajectories:
        # 1. Policy Rollout: Generate new trajectory using current policy
        for step in range(K):
            # Policy selects action
            action_logits, action_probs = model(state, history, available_mask)
            action = sample(action_probs)  # Sample from distribution
            log_prob = log_prob(action)
            
            # Simulate experiment
            observation = perform_experiment(a_true, action_i, action_j)
            update_bounds(a_lower, a_upper, action_i, action_j, observation)
            
            # Record for REINFORCE
            log_probs.append(log_prob)
            observations.append(observation)
        
        # 2. Compute Reward: Terminal MOCU (negative = reward)
        terminal_MOCU = trajectory['terminal_MOCU']  # Pre-computed
        reward = -terminal_MOCU  # Negative because we want to minimize MOCU
        
        # 3. REINFORCE Update: Update policy to maximize reward
        loss = -sum(log_prob * reward for log_prob in log_probs)
        loss.backward()
        optimizer.step()
```

**Key Points**:
- Policy generates **new trajectories** during training (not just using training data)
- Reward = `-terminal_MOCU` (minimize MOCU = maximize reward)
- Uses pre-computed `terminal_MOCU` from training data (if available)
- Otherwise uses MPNN predictor on-the-fly

### 2. **Imitation Learning** (Alternative)

Imitation learning:
- Learns to mimic expert demonstrations
- Requires expert actions in training data
- Currently not recommended (expert demonstrations not available)

---

## Policy Network Architecture

The DAD policy network consists of:

### 1. **Graph State Encoder**
- Encodes current state `(w, a_lower, a_upper)` using Graph Neural Network (GNN)
- Similar to MPNN predictor architecture
- Output: State embedding `[batch_size, hidden_dim]`

### 2. **History Encoder**
- Encodes past `(action, observation)` pairs using LSTM
- Captures sequential decision-making context
- Output: History embedding `[batch_size, hidden_dim]`

### 3. **Action Decoder**
- Combines state + history embeddings
- Outputs logits for each possible `(i, j)` pair
- Applies softmax to get probability distribution
- Masks out already observed pairs

**Architecture Diagram**:
```
State (w, a_lower, a_upper) → GNN Encoder → State Embedding
                                                      ↓
History (actions, observations) → LSTM Encoder → History Embedding
                                                      ↓
                                    Combined Embedding
                                                      ↓
                                    Action Decoder → Action Logits
                                                      ↓
                                    Softmax + Mask → Action Probabilities
```

---

## Configuration File Setup

Edit your config file (e.g., `configs/fast_config.yaml`):

```yaml
# DAD policy training data generation
dad_data:
  num_episodes: 1000       # Number of trajectories
  K: 4                     # Experiments per trajectory
  use_precomputed_mocu: true  # Pre-compute terminal MOCU (recommended)

experiment:
  dad_method: "reinforce"  # Use REINFORCE (RL)
  update_count: 4          # Should match K

training:
  epochs: 100              # Training epochs
  batch_size: 64           # Not used for REINFORCE
  learning_rate: 0.001      # Learning rate
```

---

## Training Output

During training, you'll see:

```
Loading trajectory data...
Loaded 1000 trajectories
System: N=5, K=4
Using device: cuda
Model parameters: 45,234

============================================================
Training using reinforce
============================================================
[REINFORCE] Using pre-computed MOCU values

Training: 100%|████████████| 100/100 [15:23<00:00,  9.23s/epoch]
  loss: 0.2345, reward: -0.2981, time: 9.2s

Saving model to models/dad_policies/dad_policy_N5.pth
✓ Training complete!
```

**Metrics**:
- **loss**: REINFORCE loss (negative expected reward)
- **reward**: Average reward (negative terminal MOCU)
- **time**: Time per epoch

---

## Troubleshooting

### Issue: "terminal_MOCU not in trajectory"

**Solution**: Generate DAD data with MPNN predictor:
```bash
python3 scripts/generate_dad_data.py \
    --use-mpnn-predictor \
    --mpnn-model-name "{model_name}" \
    ...
```

### Issue: "MPNN predictor not found"

**Solution**: Train MPNN predictor first:
```bash
bash scripts/bash/step1_generate_mocu_data.sh configs/fast_config.yaml
bash scripts/bash/step2_train_mpnn.sh configs/fast_config.yaml
```

### Issue: Training loss not decreasing

**Possible causes**:
1. **Learning rate too high**: Reduce `--lr` (0.001 → 0.0005)
2. **Not enough data**: Increase `num_episodes` (1000 → 5000)
3. **MPNN predictor quality**: Retrain MPNN with more data
4. **Baseline not working**: Check if baseline is being computed correctly

### Issue: CUDA out of memory

**Solution**:
1. Reduce batch size (not applicable for REINFORCE)
2. Use CPU: `--device cpu`
3. Reduce number of trajectories per epoch
4. Clear GPU cache: `torch.cuda.empty_cache()`

### Issue: Segmentation fault

**Solution**:
1. Ensure PyCUDA is not imported (check imports)
2. Use separate processes for data generation and training
3. Use `run.sh` which isolates processes

---

## Training Tips

### 1. **Start with Small Config**
```bash
# Test with fast_config first
bash scripts/bash/step3_train_dad.sh configs/fast_config.yaml
```

### 2. **Monitor Training**
- Watch loss: Should decrease over epochs
- Watch reward: Should increase (less negative)
- Check terminal MOCU values: Should be reasonable (not NaN/Inf)

### 3. **Tune Hyperparameters**
- **Learning rate**: Start with 0.001, reduce if unstable
- **Hidden dim**: 64 is good default, increase for complex systems
- **Encoding dim**: 32 is good default
- **Epochs**: 100-300 depending on system complexity

### 4. **Use Pre-computed MOCU**
- Always use `use_precomputed_mocu: true` in config
- Faster training (no MPNN calls during training)
- More consistent (same MOCU values)

### 5. **Match K and update_count**
- Ensure `dad_data.K` matches `experiment.update_count`
- Prevents distribution mismatch

---

## Advanced: Training with Curriculum Learning

For better performance, use curriculum learning:

```python
# Generate trajectories with varying K
# Stage 1: Easy (K=2)
# Stage 2: Medium (K=3)
# Stage 3: Hard (K=4)

# Training schedule:
# Epochs 1-20: Train on Stage 1 only
# Epochs 21-50: Train on Stage 1 + Stage 2
# Epochs 51-100: Train on all stages
```

See `DAD_SETUP_GUIDE.md` for curriculum learning implementation.

---

## Model Checkpoint

After training, the model is saved as:

```
{output_dir}/{name}.pth
```

Example: `models/dad_policies/dad_policy_N5.pth`

**Model contains**:
- `model_state_dict`: Policy network weights
- `config`: Network configuration (N, hidden_dim, encoding_dim)
- `train_config`: Training configuration (method, epochs, lr, etc.)
- `train_losses`: Loss history for analysis

---

## Evaluation

After training, evaluate DAD:

```bash
bash scripts/bash/step4_evaluate.sh configs/fast_config.yaml
```

DAD will:
1. Load trained policy network
2. Use policy to select experiments sequentially
3. Use MPNN predictor to compute MOCU at each step
4. Compare with other methods (iNN, RANDOM, etc.)

---

## Summary

**Training Command**:
```bash
python3 scripts/train_dad_policy.py \
    --data-path <trajectory_file> \
    --method reinforce \
    --epochs 100 \
    --lr 0.001 \
    --output-dir <output_dir> \
    --use-predicted-mocu
```

**Key Points**:
1. ✅ Train MPNN predictor first
2. ✅ Generate DAD data with pre-computed terminal MOCU
3. ✅ Use REINFORCE method (not imitation)
4. ✅ Match K and update_count
5. ✅ Monitor loss and reward during training

**Expected Results**:
- Loss decreases over epochs
- Reward (negative MOCU) increases
- Terminal MOCU values decrease (better policy)
- Evaluation performance matches or exceeds iNN

