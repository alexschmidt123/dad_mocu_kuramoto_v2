# DAD Setup Guide - Based on MPNN Understanding

## Overview

This guide explains how to properly set up DAD (Deep Adaptive Design) training and evaluation based on:
1. How MPNN works (trained on initial bounds, used on updated bounds)
2. Distribution alignment (training vs evaluation)
3. Reward signal design (terminal MOCU)

---

## Key Principles

### 1. **Use MPNN Predictor for Terminal MOCU**
- **Why**: DAD evaluation uses MPNN predictor, so training should use the same
- **How**: Pre-compute terminal MOCU using MPNN predictor during data generation
- **Benefit**: Avoids distribution mismatch and CUDA conflicts

### 2. **Distribution Alignment**
- **Training**: Trajectories with updated bounds (after K experiments)
- **Evaluation**: Uses learned policy with updated bounds (after each experiment)
- **Match**: Both use MPNN predictor on updated bounds

### 3. **Reward Signal**
- **Reward**: `-terminal_MOCU` (negative because we want to minimize MOCU)
- **Source**: MPNN predictor (estimated MOCU, not real MOCU)
- **Consistency**: Same as evaluation (MPNN predictor)

---

## Step-by-Step Setup

### Step 1: Train MPNN Predictor First

**Requirement**: MPNN predictor must be trained before generating DAD data.

```bash
# Step 1: Generate MPNN training data
bash scripts/bash/step1_generate_mocu_data.sh configs/fast_config.yaml

# Step 2: Train MPNN predictor
bash scripts/bash/step2_train_mpnn.sh configs/fast_config.yaml
```

**Check**: Verify MPNN model exists:
```bash
ls models/{config_name}/{timestamp}/model.pth
ls models/{config_name}/{timestamp}/statistics.pth
```

---

### Step 2: Configure DAD Data Generation

Edit your config file (e.g., `configs/fast_config.yaml`):

```yaml
# DAD policy training data generation
dad_data:
  num_episodes: 1000       # Number of trajectories (recommended: 1000-5000)
  K: 4                     # Number of sequential experiments per trajectory
  use_precomputed_mocu: true  # CRITICAL: Use MPNN predictor (recommended)
```

**Key Settings**:
- `num_episodes`: More is better (1000 minimum, 5000+ for better performance)
- `K`: Should match evaluation `update_count` (typically 4-10)
- `use_precomputed_mocu: true`: **REQUIRED** - uses MPNN predictor

---

### Step 3: Generate DAD Training Data

The script automatically:
1. Loads MPNN predictor from the trained model
2. Generates trajectories with random expert policy
3. Computes terminal MOCU using MPNN predictor for final bounds
4. Saves trajectories with pre-computed terminal MOCU

```bash
# This is done automatically by run.sh, but can be run manually:
python3 scripts/generate_dad_data.py \
    --N 5 \
    --K 4 \
    --num-episodes 1000 \
    --use-mpnn-predictor \
    --mpnn-model-name "{config_name}_{timestamp}" \
    --output-dir data/dad_trajectories/
```

**What gets generated**:
- Trajectories with: `w`, `a_true`, `states[]`, `actions[]`, `terminal_MOCU`
- `states[]`: Sequence of bounds after each experiment
- `terminal_MOCU`: Estimated MOCU using MPNN predictor (for final bounds)

---

### Step 4: Configure DAD Training

Edit config file:

```yaml
experiment:
  dad_method: "reinforce"  # Use REINFORCE (RL) not imitation (behavior cloning)
  
training:
  epochs: 100              # Training epochs
  batch_size: 64           # Batch size for REINFORCE
  learning_rate: 0.001     # Learning rate
```

**Training Method**:
- **REINFORCE** (recommended): Uses terminal MOCU directly as reward
- **Imitation**: Requires expert demonstrations (not recommended)

---

### Step 5: Train DAD Policy

```bash
# This is done automatically by run.sh, but can be run manually:
python3 scripts/train_dad_policy.py \
    --data-path data/dad_trajectories/dad_trajectories_N5_K4_1000.pth \
    --method reinforce \
    --name "dad_policy_N5" \
    --epochs 100 \
    --batch-size 64 \
    --output-dir models/dad_policies/ \
    --use-predicted-mocu  # Use MPNN predictor if terminal_MOCU not pre-computed
```

**What happens during training**:
1. Load trajectories with pre-computed `terminal_MOCU`
2. For each trajectory:
   - Policy network generates new trajectory (rollout)
   - Uses `a_true` to simulate experiments
   - Updates bounds based on observations
   - Reward = `-terminal_MOCU` (from pre-computed value)
3. REINFORCE updates policy to maximize reward (minimize MOCU)

---

### Step 6: Evaluate DAD

DAD evaluation uses MPNN predictor iteratively (same as iNN):

```python
# During evaluation (in base.py):
for iteration in range(update_cnt):
    # Policy selects experiment
    i, j = dad_policy.select_experiment(state, history)
    
    # Perform experiment, update bounds
    observation = perform_experiment(a_true, i, j, w, h, M)
    a_lower, a_upper = update_bounds(a_lower, a_upper, i, j, observation, w)
    
    # Predict MOCU using MPNN predictor (same as iNN)
    mocu_pred = predict_mocu(mpnn_model, mean, std, w, a_lower, a_upper)
    MOCUCurve[iteration + 1] = mocu_pred
```

**Key**: DAD uses the same MPNN predictor that was used during training!

---

## Configuration Examples

### Minimal Setup (Fast Testing)

```yaml
# configs/fast_config.yaml
dad_data:
  num_episodes: 100        # Minimal for testing
  K: 4                     # 4 experiments
  use_precomputed_mocu: true

experiment:
  update_count: 4          # Match K
  dad_method: "reinforce"
```

### Production Setup (N=5)

```yaml
# configs/N5_config.yaml
dad_data:
  num_episodes: 2000       # More trajectories for better performance
  K: 10                    # 10 experiments (match update_count)
  use_precomputed_mocu: true

experiment:
  update_count: 10         # Match K
  dad_method: "reinforce"

training:
  epochs: 200              # More epochs
  batch_size: 64
  learning_rate: 0.001
```

### Production Setup (N=7, N=9)

```yaml
# configs/N7_config.yaml or N9_config.yaml
dad_data:
  num_episodes: 5000       # Many trajectories for complex systems
  K: 10                    # 10 experiments
  use_precomputed_mocu: true

experiment:
  update_count: 10
  dad_method: "reinforce"

training:
  epochs: 300              # More epochs for complex systems
  batch_size: 64
  learning_rate: 0.0005    # Lower learning rate for stability
```

---

## Important Design Decisions

### ✅ DO: Use MPNN Predictor for Terminal MOCU

**Why**:
- DAD evaluation uses MPNN predictor iteratively
- Training should match evaluation distribution
- Avoids CUDA context conflicts

**How**:
```yaml
dad_data:
  use_precomputed_mocu: true  # Pre-compute using MPNN
```

### ✅ DO: Use REINFORCE (Not Imitation)

**Why**:
- REINFORCE optimizes terminal MOCU directly
- No need for expert demonstrations
- More flexible and adaptive

**How**:
```yaml
experiment:
  dad_method: "reinforce"
```

### ✅ DO: Match K and update_count

**Why**:
- Training trajectories should match evaluation length
- Ensures policy learns for correct horizon

**How**:
```yaml
dad_data:
  K: 4                     # Training trajectory length

experiment:
  update_count: 4          # Evaluation length (should match K)
```

### ❌ DON'T: Use Real MOCU (PyCUDA) During Training

**Why**:
- Causes CUDA context conflicts with PyTorch
- Doesn't match evaluation (evaluation uses MPNN)
- Slower and unnecessary

**Avoid**:
```yaml
dad_data:
  use_precomputed_mocu: false  # DON'T do this
```

### ❌ DON'T: Use Different K for Training vs Evaluation

**Why**:
- Causes distribution mismatch
- Policy trained for wrong horizon

**Avoid**:
```yaml
dad_data:
  K: 4                     # Training: 4 experiments

experiment:
  update_count: 10         # Evaluation: 10 experiments (WRONG!)
```

---

## Complete Workflow

### Full Pipeline

```bash
# 1. Generate MPNN training data
bash scripts/bash/step1_generate_mocu_data.sh configs/fast_config.yaml

# 2. Train MPNN predictor
bash scripts/bash/step2_train_mpnn.sh configs/fast_config.yaml

# 3. Generate DAD training data (uses MPNN predictor)
# This happens automatically in run.sh, but can be done manually:
python3 scripts/generate_dad_data.py \
    --N 5 --K 4 --num-episodes 1000 \
    --use-mpnn-predictor \
    --mpnn-model-name "{config_name}_{timestamp}" \
    --output-dir data/dad_trajectories/

# 4. Train DAD policy
bash scripts/bash/step3_train_dad.sh configs/fast_config.yaml

# 5. Evaluate (includes DAD)
bash scripts/bash/step4_evaluate.sh configs/fast_config.yaml

# 6. Visualize results
bash scripts/bash/step5_visualize.sh configs/fast_config.yaml
```

### Or Use Automated Script

```bash
# Run everything automatically
bash run.sh configs/fast_config.yaml
```

---

## Troubleshooting

### Issue: "MPNN predictor not found"

**Solution**: Train MPNN first:
```bash
bash scripts/bash/step1_generate_mocu_data.sh configs/fast_config.yaml
bash scripts/bash/step2_train_mpnn.sh configs/fast_config.yaml
```

### Issue: "terminal_MOCU not in trajectory"

**Solution**: Enable MPNN predictor in data generation:
```yaml
dad_data:
  use_precomputed_mocu: true
```

Or pass `--use-mpnn-predictor` flag when generating data.

### Issue: DAD performance is poor

**Possible causes**:
1. **Not enough training data**: Increase `num_episodes` (1000 → 5000)
2. **K mismatch**: Ensure `dad_data.K` matches `experiment.update_count`
3. **MPNN predictor quality**: Retrain MPNN with more data
4. **Learning rate too high**: Reduce `learning_rate` (0.001 → 0.0005)
5. **Not enough epochs**: Increase `epochs` (100 → 200-300)

### Issue: Distribution mismatch

**Symptoms**: Training loss decreases but evaluation performance is poor

**Solution**: 
1. Ensure `use_precomputed_mocu: true` (uses MPNN predictor)
2. Match `K` and `update_count`
3. Consider augmenting MPNN training data with intermediate states

---

## Advanced: Distribution Alignment

### Problem: Distribution Shift

Even with MPNN predictor, there's still a subtle mismatch:
- **Training**: Trajectories generated with **random policy**
- **Evaluation**: Trajectories generated with **learned policy**

### Solution 1: Curriculum Learning (Recommended)

Generate trajectories with varying difficulty:

```python
# In generate_dad_data.py, add curriculum:
def generate_trajectories_curriculum(N, K, num_episodes):
    trajectories = []
    
    # Stage 1: Easy (K=2, 30%)
    for _ in range(int(0.3 * num_episodes)):
        traj = generate_trajectory(N, K=2, ...)
        trajectories.append(traj)
    
    # Stage 2: Medium (K=3, 40%)
    for _ in range(int(0.4 * num_episodes)):
        traj = generate_trajectory(N, K=3, ...)
        trajectories.append(traj)
    
    # Stage 3: Hard (K=K_full, 30%)
    for _ in range(int(0.3 * num_episodes)):
        traj = generate_trajectory(N, K=K, ...)
        trajectories.append(traj)
    
    return trajectories
```

**Training schedule**:
- Epochs 1-20: Train on Stage 1 only
- Epochs 21-50: Train on Stage 1 + Stage 2
- Epochs 51-100: Train on all stages

### Solution 2: Self-Play (Advanced)

Generate training data using current policy:

```python
def generate_trajectories_selfplay(model, N, K, num_episodes):
    """Generate trajectories using current policy (self-play)"""
    model.eval()
    trajectories = []
    
    for _ in range(num_episodes):
        # Generate trajectory using current policy
        traj = generate_trajectory_with_policy(model, N, K, ...)
        trajectories.append(traj)
    
    return trajectories
```

**Training loop**:
```python
for epoch in range(num_epochs):
    # Generate data using current policy
    trajectories = generate_trajectories_selfplay(model, N, K, num_episodes)
    
    # Train on self-play data
    train_reinforce(model, trajectories, ...)
```

---

## Summary Checklist

Before training DAD, ensure:

- [ ] MPNN predictor is trained first
- [ ] `use_precomputed_mocu: true` in config
- [ ] `dad_data.K` matches `experiment.update_count`
- [ ] `num_episodes` is sufficient (1000+)
- [ ] `dad_method: "reinforce"` (not imitation)
- [ ] Training data has `terminal_MOCU` field
- [ ] MPNN model path is correct

During training, monitor:

- [ ] Training loss decreases
- [ ] Reward (negative MOCU) increases
- [ ] No CUDA errors or segfaults
- [ ] Terminal MOCU values are reasonable (not NaN/Inf)

During evaluation, verify:

- [ ] DAD uses MPNN predictor (same as iNN)
- [ ] MOCU values decrease over iterations
- [ ] Performance matches or exceeds iNN
- [ ] No distribution mismatch errors

---

## Quick Reference

| Setting | Recommended Value | Purpose |
|---------|------------------|---------|
| `num_episodes` | 1000-5000 | More = better, but slower |
| `K` | 4-10 | Match `update_count` |
| `use_precomputed_mocu` | `true` | Use MPNN predictor |
| `dad_method` | `"reinforce"` | RL, not imitation |
| `epochs` | 100-300 | More for complex systems |
| `batch_size` | 64 | Standard |
| `learning_rate` | 0.001 | Reduce if unstable |

---

## Next Steps

1. **Start with fast_config**: Test with minimal settings
2. **Scale up**: Use N5_config, N7_config, N9_config
3. **Monitor performance**: Compare DAD vs iNN in evaluation
4. **Improve MPNN**: If DAD performance is poor, retrain MPNN with more data
5. **Consider curriculum learning**: If distribution mismatch persists

