# Data Generation Comparison with Original 2023 Paper Implementation

## Summary

The data generation code matches the original implementation, with one fix applied to ensure consistency with the training script.

## ✅ Verified Parameters (Matching Original)

### 1. Natural Frequencies Generation
- **Format**: `w[i] = 12 * (0.5 - random.random())`
- **Range**: [-6, 6]
- **Status**: ✅ Matches original

### 2. MOCU Computation Parameters
- **T (Time horizon)**: 4.0
- **h (Time step)**: 1.0 / 160.0
- **M (Number of steps)**: `int(T / h) = 640`
- **K_max (Monte Carlo samples)**: 20480 (default)
- **Computation**: MOCU computed twice and averaged for stability
- **Status**: ✅ Matches original

### 3. Coupling Distributions (Two Types)

**Type 1: Per-edge random multiplier**
- Uncertainty: `0.3 * random.random()`
- Multiplier: 
  - 50% chance: `0.6 * random.random()`
  - 50% chance: `1.1 * random.random()`
- Formula: `f_inv * (1 ± uncertainty) * mul`
- **Status**: ✅ Matches original

**Type 2: Per-oscillator random multiplier**
- Uncertainty: `0.3 * random.random()`
- Per-oscillator base multiplier: 
  - 50% chance: 0.6
  - 50% chance: 1.1
- Edge multiplier: `mul_ * random.random()`
- **Status**: ✅ Matches original

### 4. System Filtering
- **Skip synchronized systems**: Uses `mocu_comp()` to check initial synchronization
- Systems with `init_sync_check == 1` are filtered out
- **Status**: ✅ Matches original

### 5. PyTorch Geometric Conversion
- **Node features**: Natural frequencies `w` → `[N, 1]` tensor
- **Edge indices**: Fully connected directed graph (excludes self-loops)
- **Edge attributes**: `[a_lower, a_upper]` for each edge → `[num_edges, 2]`
- **Target**: Mean MOCU value
- **Format**: Uses `getEdgeAtt()` function (same pattern as original)
- **Status**: ✅ Matches original

### 6. Data Statistics
- **Samples per type**: 37500 (default)
- **Total samples**: 2 × samples_per_type = 75000 (before filtering)
- **Status**: ✅ Matches original

## 🔧 Fixed Issue

### Train/Test Split Inconsistency

**Problem Found**:
- Data generation script created separate `train.pth` and `test.pth` files
- Training script (`train_mocu_predictor.py`) expects a single file and splits at 96/4 internally
- This caused double-splitting or incorrect train/test sizes

**Fix Applied**:
- ✅ Data generation now outputs a **single combined file**
- ✅ File naming: `{total_samples}_{N}o_train.pth` (e.g., `75000_5o_train.pth`)
- ✅ Training script handles the 96/4 split automatically
- ✅ Matches original repository pattern

**Before**:
```python
# Generated separate files
train_data = pyg_data_list[:train_size]
test_data = pyg_data_list[train_size:]
torch.save(train_data, train_file)
torch.save(test_data, test_file)
```

**After**:
```python
# Generate single file (training script splits at 96/4)
torch.save(pyg_data_list, output_file)
# Training script: data_train = data_list[0:int(0.96 * len(data_list))]
```

## ✅ Data Format Verification

### Data Structure
```python
Data(
    x: [N, 1]              # Node features (natural frequencies)
    edge_index: [2, E]     # Edge connectivity (fully connected, no self-loops)
    edge_attr: [E, 2]      # Edge features [a_lower, a_upper]
    y: [1, 1]              # Target MOCU value
)
```

### Normalization
- MOCU values are normalized to mean=0, std=1 during training
- Statistics saved to `models/{model_name}/statistics.pth`
- **Status**: ✅ Matches original

## 📊 Comparison Table

| Aspect | Original (2023) | Current Code | Status |
|--------|-----------------|--------------|--------|
| Natural frequencies range | [-6, 6] | [-6, 6] | ✅ Match |
| Time horizon T | 4.0 | 4.0 | ✅ Match |
| Time step h | 1/160 | 1/160 | ✅ Match |
| K_max (MC samples) | 20480 | 20480 | ✅ Match |
| Coupling Type 1 | Per-edge multiplier | Per-edge multiplier | ✅ Match |
| Coupling Type 2 | Per-oscillator multiplier | Per-oscillator multiplier | ✅ Match |
| Synchronization filter | Skip sync systems | Skip sync systems | ✅ Match |
| MOCU computation | Compute twice, average | Compute twice, average | ✅ Match |
| PyG format | Same structure | Same structure | ✅ Match |
| Train/test split | 96/4 in training script | 96/4 in training script | ✅ Fixed |
| Edge attribute format | [E, 2] with transpose | [E, 2] with transpose | ✅ Match |

## 🎯 Conclusion

**Data generation now matches the original 2023 paper implementation**, except for DAD policy training data (which is intentionally different as it's trajectory-based for imitation learning).

The only issue found and fixed was the train/test split inconsistency, which has been resolved to match the original repository's pattern.

