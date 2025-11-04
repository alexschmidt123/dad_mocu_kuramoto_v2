# DAD Training and Testing Design Recommendations

## Current Situation Analysis

### MPNN Training Data Distribution
- **Input**: `(w, a_lower_init, a_upper_init)` - **Initial bounds before any experiments**
- **Output**: True MOCU computed via PyCUDA
- **Distribution**: Type 1 (per-edge multiplier) or Type 2 (per-oscillator multiplier)
- **All samples at iteration 0** (initial state)

### DAD Training Data (Current)
- **Input**: Random expert policy trajectories
- **State sequence**: Initial → K experiments → Final bounds
- **Terminal MOCU**: Estimated via MPNN predictor (not true MOCU)
- **Issue**: Trajectories generated with **random policy**, not learned policy

### DAD Evaluation (Current)
- **Uses learned policy** to select experiments
- **MPNN predictor** called at each iteration with **updated bounds**
- **Distribution shift**: MPNN sees updated bounds it wasn't trained on

---

## Recommendations

### Option 1: Improve MPNN Training Data (RECOMMENDED)

**Problem**: MPNN trained only on initial bounds, but used on updated bounds during DAD evaluation.

**Solution**: Augment MPNN training data with **intermediate states** from experimental trajectories.

```python
# Pseudo-code for enhanced MPNN training data generation
def generate_mpnn_training_data_with_intermediates(N, samples_per_type, K_max):
    """
    Generate MPNN training data that includes:
    1. Initial bounds (50% of data) - current approach
    2. Intermediate bounds (30% of data) - after 1-3 experiments
    3. Updated bounds (20% of data) - after 4-6 experiments
    """
    data_list = []
    
    # 50% initial bounds (current approach)
    for _ in range(int(0.5 * samples_per_type)):
        w, a_lower, a_upper = generate_random_system(N)
        mocu = compute_true_mocu(w, a_lower, a_upper, K_max)
        data_list.append((w, a_lower, a_upper, mocu, iteration=0))
    
    # 30% intermediate bounds (after 1-3 experiments)
    for _ in range(int(0.3 * samples_per_type)):
        w, a_lower_init, a_upper_init, a_true, _ = generate_random_system(N)
        num_experiments = random.randint(1, 3)
        # Run random experiments
        a_lower, a_upper = run_random_experiments(
            w, a_lower_init, a_upper_init, a_true, num_experiments
        )
        mocu = compute_true_mocu(w, a_lower, a_upper, K_max)
        data_list.append((w, a_lower, a_upper, mocu, iteration=num_experiments))
    
    # 20% updated bounds (after 4-6 experiments)
    for _ in range(int(0.2 * samples_per_type)):
        w, a_lower_init, a_upper_init, a_true, _ = generate_random_system(N)
        num_experiments = random.randint(4, 6)
        a_lower, a_upper = run_random_experiments(
            w, a_lower_init, a_upper_init, a_true, num_experiments
        )
        mocu = compute_true_mocu(w, a_lower, a_upper, K_max)
        data_list.append((w, a_lower, a_upper, mocu, iteration=num_experiments))
    
    return data_list
```

**Benefits**:
- MPNN learns to predict MOCU for **both initial and updated bounds**
- Reduces distribution shift during DAD evaluation
- Better generalization to intermediate states

---

### Option 2: Curriculum Learning for DAD Training

**Problem**: DAD training uses random policy, but evaluation uses learned policy.

**Solution**: Use **curriculum learning** - start with easy trajectories, progressively increase difficulty.

```python
# Pseudo-code for curriculum learning
def generate_dad_data_curriculum(N, K, num_episodes, curriculum_stages=3):
    """
    Generate DAD training data with curriculum learning:
    Stage 1: Simple trajectories (fewer experiments, easier systems)
    Stage 2: Medium trajectories (more experiments)
    Stage 3: Hard trajectories (full K experiments, complex systems)
    """
    trajectories = []
    
    # Stage 1: Simple (K=2 experiments, 30% of data)
    for _ in range(int(0.3 * num_episodes)):
        traj = generate_trajectory(N, K=2, ...)
        trajectories.append(traj)
    
    # Stage 2: Medium (K=3 experiments, 40% of data)
    for _ in range(int(0.4 * num_episodes)):
        traj = generate_trajectory(N, K=3, ...)
        trajectories.append(traj)
    
    # Stage 3: Hard (K=K_full experiments, 30% of data)
    for _ in range(int(0.3 * num_episodes)):
        traj = generate_trajectory(N, K=K, ...)
        trajectories.append(traj)
    
    return trajectories
```

**Training Strategy**:
1. **Epoch 1-20**: Train on Stage 1 data (easy)
2. **Epoch 21-50**: Train on Stage 1 + Stage 2 (medium)
3. **Epoch 51-100**: Train on all stages (hard)

---

### Option 3: Self-Play / On-Policy Data Generation

**Problem**: Distribution mismatch between training (random policy) and evaluation (learned policy).

**Solution**: Use **self-play** - generate training data using the **current learned policy**.

```python
# Pseudo-code for self-play data generation
def generate_dad_data_selfplay(model, N, K, num_episodes):
    """
    Generate DAD training data using current policy (self-play).
    This ensures training data matches evaluation distribution.
    """
    trajectories = []
    model.eval()  # Use current policy
    
    for _ in range(num_episodes):
        w, a_lower_init, a_upper_init, a_true, _ = generate_random_system(N)
        trajectory = {
            'w': w,
            'a_true': a_true,
            'states': [(a_lower_init.copy(), a_upper_init.copy())],
            'actions': []
        }
        
        a_lower, a_upper = a_lower_init.copy(), a_upper_init.copy()
        
        for step in range(K):
            # Use CURRENT POLICY to select action
            state_data = create_state_data(w, a_lower, a_upper, device)
            action_logits, action_probs = model(state_data, ...)
            action_idx = torch.multinomial(action_probs, 1).item()
            i, j = model.idx_to_pair(action_idx)
            
            # Perform experiment and update bounds
            observation = perform_experiment(a_true, i, j, w, h, M)
            a_lower, a_upper = update_bounds(a_lower, a_upper, i, j, observation, w)
            
            trajectory['actions'].append((i, j))
            trajectory['states'].append((a_lower.copy(), a_upper.copy()))
        
        # Compute terminal MOCU
        terminal_MOCU = predict_mocu(mpnn_predictor, ..., w, a_lower, a_upper)
        trajectory['terminal_MOCU'] = terminal_MOCU
        
        trajectories.append(trajectory)
    
    return trajectories
```

**Training Loop**:
```python
for epoch in range(num_epochs):
    # Generate data using current policy (self-play)
    trajectories = generate_dad_data_selfplay(model, N, K, num_episodes)
    
    # Train on self-play data
    loss = train_reinforce(model, trajectories, ...)
    
    # Update policy
    optimizer.step()
```

**Benefits**:
- Training data matches evaluation distribution
- Policy learns from its own behavior
- Reduces distribution shift

---

### Option 4: Hybrid Approach (RECOMMENDED FOR INITIAL IMPLEMENTATION)

**Combine Options 1 + 2**: Improve MPNN + Use curriculum learning for DAD.

**Implementation Steps**:

1. **Modify `generate_mocu_data.py`**:
   - Add option to generate intermediate states (after 1-6 experiments)
   - Mix initial bounds (50%) + intermediate bounds (50%)

2. **Modify `generate_dad_data.py`**:
   - Add curriculum learning stages
   - Generate trajectories with varying K (2, 3, 4, ...)

3. **Modify `train_dad_policy.py`**:
   - Implement curriculum learning schedule
   - Start with easy trajectories, progress to hard

---

## Testing Strategy

### Test Set Design
1. **Separate test systems**: Generate test systems independently from training
2. **Diverse initial bounds**: Use same distribution as training (Type 1 + Type 2)
3. **Multiple runs**: Test on 10-20 different random systems
4. **True MOCU evaluation**: Use PyCUDA to compute true MOCU for comparison (not just MPNN estimates)

### Validation During Training
1. **Hold-out validation set**: Keep 10-20% of trajectories for validation
2. **Monitor terminal MOCU**: Track terminal MOCU on validation set (should decrease)
3. **Early stopping**: Stop if validation MOCU doesn't improve for 10 epochs

---

## Implementation Priority

### Phase 1: Quick Win (Immediate)
1. ✅ **Add intermediate states to MPNN training** (Option 1)
   - Modify `generate_mocu_data.py` to include bounds after 1-6 experiments
   - Retrain MPNN with augmented data

### Phase 2: Improvement (Next)
2. ✅ **Implement curriculum learning for DAD** (Option 2)
   - Modify `generate_dad_data.py` to generate trajectories with varying K
   - Update `train_dad_policy.py` to use curriculum schedule

### Phase 3: Advanced (Future)
3. ⚠️ **Self-play data generation** (Option 3)
   - More complex, requires careful implementation
   - Best for fine-tuning after initial training

---

## Expected Improvements

1. **Better MPNN generalization**: Predicts MOCU accurately for both initial and updated bounds
2. **Reduced distribution shift**: Training data matches evaluation distribution better
3. **Faster convergence**: Curriculum learning helps policy learn progressively
4. **Better final performance**: Self-play ensures optimal policy-data alignment

---

## Code Changes Required

### 1. `scripts/generate_mocu_data.py`
- Add `--include_intermediates` flag
- Generate intermediate states (after 1-6 experiments)
- Mix initial + intermediate states

### 2. `scripts/generate_dad_data.py`
- Add `--curriculum` flag
- Generate trajectories with varying K (2, 3, 4, ...)
- Return curriculum stage labels

### 3. `scripts/train_dad_policy.py`
- Add curriculum learning schedule
- Load trajectories by stage
- Train progressively on harder stages

### 4. `configs/*.yaml`
- Add `mpnn_data.include_intermediates: true`
- Add `dad_data.curriculum_stages: [2, 3, 4]`
- Add `dad_data.curriculum_schedule: {epochs_1: 20, epochs_2: 30, epochs_3: 50}`

