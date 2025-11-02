# Separate Usage Pattern (Paper 2023 Workflow)

This document describes how the code maintains **strict separation** between MPNN (PyTorch) and PyCUDA operations, following the original paper 2023 workflow.

## Core Principle

**MPNN and PyCUDA are NEVER used simultaneously. They run in separate phases with explicit synchronization.**

## Separation Strategy

### 1. **Lazy Imports**
- `iNN` and `NN` methods: **NO top-level PyCUDA import**
  - They only import PyTorch/MPNN modules
  - PyCUDA is imported lazily in `base.run_episode()` when needed

- `evaluation.py`: **NO top-level PyCUDA import**
  - MOCU is imported lazily only when computing initial MOCU
  - This happens BEFORE MPNN methods are instantiated

- `DAD_MOCU_Method`: **NO top-level PyCUDA import**
  - Uses policy network (PyTorch) for selection
  - PyCUDA is only used in `base.run_episode()` for MOCU computation

### 2. **Synchronization Points in `base.run_episode()`**

The `run_episode()` method enforces strict separation:

```python
# Phase 1: Initial MOCU computation (PyCUDA)
it_temp_val = MOCU(...)  # PyCUDA operation
torch.cuda.synchronize()  # Wait for PyCUDA to finish

# Phase 2: Sequential design loop
for iteration in range(update_cnt):
    torch.cuda.synchronize()  # Ensure PyCUDA is done
    
    # Select experiment (MPNN methods use PyTorch here)
    selected_i, selected_j = self.select_experiment(...)
    
    torch.cuda.synchronize()  # Ensure MPNN is done
    
    # Re-compute MOCU (PyCUDA operation)
    it_temp_val = MOCU(...)
```

### 3. **Workflow Isolation**

Each workflow step runs in a **separate process**:

1. **Step 1: Generate MPNN data** (PyCUDA only)
   - Uses `generate_mocu_data.py` which imports PyCUDA
   - Process exits → CUDA context destroyed

2. **Step 2: Train MPNN** (PyTorch only)
   - Uses `train_mocu_predictor.py` which only uses PyTorch
   - Fresh process → No PyCUDA context

3. **Step 3: Train DAD** (MPNN predictor only, no PyCUDA)
   - Uses `train_dad_policy.py` with MPNN predictor
   - PyCUDA is NOT imported when using MPNN predictor
   - Fresh process → No PyCUDA context conflict

4. **Step 4: Evaluation** (Alternating usage with sync)
   - Uses `evaluation.py` which imports MOCU lazily
   - For MPNN methods: Initial MOCU (PyCUDA) → Sync → Load MPNN → Sync → Alternate

## Verification Checklist

✅ **No concurrent GPU access**
- MPNN forward passes and PyCUDA kernels never run simultaneously
- Explicit `torch.cuda.synchronize()` between phases

✅ **Separate usage pattern**
- MPNN methods (iNN/NN) never import PyCUDA at top level
- PyCUDA is only imported when actually needed (lazy import)

✅ **No mixed usage**
- iNN/NN methods use MPNN for inference separately from PyCUDA operations
- DAD uses policy network separately from PyCUDA MOCU computation

✅ **Workflow isolation**
- Each step runs in separate process
- No PyCUDA context persists between steps

## Original Paper 2023 Compliance

This implementation matches the original paper's workflow:

1. **MPNN Training**: Pure PyTorch, no PyCUDA
2. **MPNN Inference (iNN/NN)**: Alternates with PyCUDA MOCU computation, never concurrent
3. **Evaluation**: Explicit synchronization ensures no conflicts

## Testing

To verify separation, run:
```bash
bash run.sh configs/N5_config.yaml  # N=5 experiment
bash run.sh configs/N7_config.yaml  # N=7 experiment
```

Both should run without CUDA context conflicts or segmentation faults.

