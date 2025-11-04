# How Greedy MPNN Works in Sequential Experiments

## Overview

This document explains:
1. How greedy MPNN (iNN/NN) works in sequential experiments
2. How MPNN is trained and tested
3. What kind of data it uses (real MOCU vs estimated MOCU)

---

## 1. How Greedy MPNN Works in Sequential Experiments

### Two Variants: iNN (Iterative) vs NN (Static)

#### **iNN (Iterative Neural Network)** - Adaptive
- **Re-computes R matrix at EVERY step** based on updated bounds
- More accurate, adapts to new observations
- Computationally expensive

#### **NN (Static Neural Network)** - Fast
- **Computes R matrix ONCE at the beginning** (initial bounds)
- Faster, but less adaptive
- Reuses the same R matrix for all selections

---

### Step-by-Step Process (iNN Method)

#### **Step 1: Initialize**
- Load trained MPNNPlusPredictor model
- Load normalization statistics (mean, std)

#### **Step 2: For Each Experiment Selection (Iteration)**

```python
# 1. Compute Expected MOCU Matrix (R matrix)
R_matrix = _compute_expected_mocu_matrix(w, a_lower_bounds, a_upper_bounds)
```

**What `_compute_expected_mocu_matrix` does:**

For each possible experiment pair `(i, j)`:

1. **Compute `f_inv`** (critical coupling threshold):
   ```python
   f_inv = 0.5 * |w[i] - w[j]|
   ```

2. **Scenario 1: Assume synchronized observation**
   - Update bounds: `a_lower[i,j] = max(a_lower[i,j], f_inv)`
   - Create graph data: `(w, a_lower_syn, a_upper_syn)`
   - Predict MOCU using MPNN: `MOCU_syn = model(graph_syn)`

3. **Scenario 2: Assume non-synchronized observation**
   - Update bounds: `a_upper[i,j] = min(a_upper[i,j], f_inv)`
   - Create graph data: `(w, a_lower_nonsyn, a_upper_nonsyn)`
   - Predict MOCU using MPNN: `MOCU_nonsyn = model(graph_nonsyn)`

4. **Compute probability of synchronization**:
   ```python
   a_tilde = clamp(f_inv, a_lower[i,j], a_upper[i,j])
   P_syn = (a_upper[i,j] - a_tilde) / (a_upper[i,j] - a_lower[i,j])
   P_nonsyn = 1 - P_syn
   ```

5. **Compute expected remaining MOCU**:
   ```python
   R[i,j] = P_syn * MOCU_syn + P_nonsyn * MOCU_nonsyn
   ```

#### **Step 3: Select Experiment**
```python
# Mask out already selected pairs
for (i, j) in history:
    R_matrix[i, j] = 0.0

# Select pair with minimum expected MOCU (greedy)
(i_selected, j_selected) = argmin(R_matrix)
```

#### **Step 4: Perform Experiment**
- Actually perform experiment on `(i_selected, j_selected)`
- Observe if synchronized (1) or not (0)
- Update bounds based on observation

#### **Step 5: Update MOCU Curve**
```python
# Re-compute MOCU for updated bounds using MPNN predictor
mocu_pred = predict_mocu(model, mean, std, w, a_lower_updated, a_upper_updated)
MOCUCurve[iteration + 1] = mocu_pred
```

#### **Step 6: Repeat**
- Go back to Step 2 for next iteration
- **iNN**: Re-computes R matrix with NEW bounds
- **NN**: Reuses same R matrix (computed once)

---

### Key Insight: Expected MOCU (R Matrix)

The R matrix represents **expected remaining MOCU** after performing each experiment:

```
R[i,j] = E[MOCU_remaining | experiment (i,j)]
       = P(syn) * MOCU(if_syn) + P(not_syn) * MOCU(if_not_syn)
```

Greedy selection: Choose `(i,j)` that minimizes `R[i,j]` (minimizes expected remaining MOCU).

---

## 2. How MPNN is Trained

### Training Data Generation (`generate_mocu_data.py`)

#### **Input**: Graph representation
- **Node features**: Natural frequencies `w` [N]
- **Edge features**: Coupling bounds `[a_lower, a_upper]` [num_edges, 2]
- **All samples are at iteration 0** (initial bounds, before any experiments)

#### **Output**: True MOCU value
- Computed using **PyCUDA** (real MOCU, not estimated)
- Computed twice and averaged for stability:
  ```python
  MOCU_val1 = MOCU_pycuda(K_max, w, N, h, M, T, a_lower, a_upper, 0)
  MOCU_val2 = MOCU_pycuda(K_max, w, N, h, M, T, a_lower, a_upper, 0)
  mean_MOCU = (MOCU_val1 + MOCU_val2) / 2
  ```

#### **Data Distribution**:
- **Type 1**: Per-edge random multiplier (50% of data)
- **Type 2**: Per-oscillator random multiplier (50% of data)
- Each sample: `(w, a_lower_init, a_upper_init, true_MOCU)`

#### **Training Process** (`train_predictor.py`):

1. **Load Data**:
   ```python
   data_list = torch.load(data_path)  # PyTorch Geometric Data objects
   ```

2. **Normalization**:
   ```python
   mean = np.mean([d.y for d in data_list])
   std = np.std([d.y for d in data_list])
   # Normalize targets: y_normalized = (y - mean) / std
   ```

3. **Train/Test Split**:
   - 96% training, 4% test

4. **Training Loop**:
   ```python
   for epoch in range(EPOCHS):
       for batch in train_loader:
           prediction = model(batch)  # [batch_size, 1]
           
           # MSE Loss
           mse_loss = F.mse_loss(prediction, batch.y)
           
           # Rank Loss (monotonicity constraint)
           rank_loss = computeRankLoss(prediction, batch.edge_attr)
           
           total_loss = mse_loss + constrain_weight * rank_loss
           total_loss.backward()
           optimizer.step()
   ```

5. **Save Model**:
   - Model: `models/{config}/{timestamp}/model.pth`
   - Statistics: `models/{config}/{timestamp}/statistics.pth` (mean, std)

---

## 3. How MPNN is Tested

### During Training (Validation)
- Uses **test set** (4% of data)
- Computes MSE on normalized predictions
- Reports test MSE every 10 epochs

### During Evaluation (`evaluate.py`)

#### **Initial MOCU**:
- Uses **PyCUDA** to compute true MOCU:
  ```python
  initial_mocu = MOCU_pycuda(K_max, w, N, h, M, T, a_lower_init, a_upper_init, 0)
  ```

#### **Iterative MOCU** (for iNN/NN methods):
- Uses **MPNN predictor** to estimate MOCU:
  ```python
  mocu_pred = predict_mocu(model, mean, std, w, a_lower_updated, a_upper_updated)
  MOCUCurve[iteration + 1] = mocu_pred
  ```

---

## 4. What Kind of Data Does MPNN Use?

### **Training Data**: 
- ✅ **Real MOCU** (computed via PyCUDA)
- Input: Initial bounds (before experiments)
- Output: True MOCU value (ground truth)

### **Testing/Evaluation Data**:
- **Initial MOCU**: ✅ **Real MOCU** (PyCUDA)
- **Iterative MOCU**: ⚠️ **Estimated MOCU** (MPNN predictor)
  - Uses MPNN to predict MOCU for **updated bounds** (after experiments)
  - This is where distribution shift occurs!

---

## 5. Distribution Shift Problem

### **Training Distribution**:
- MPNN sees: `(w, a_lower_init, a_upper_init)` → `true_MOCU`
- All samples are at **iteration 0** (initial bounds)

### **Evaluation Distribution**:
- MPNN sees: `(w, a_lower_updated, a_upper_updated)` → `estimated_MOCU`
- Samples are at **iteration 1, 2, 3, ...** (updated bounds after experiments)

### **Problem**:
- MPNN was trained on initial bounds, but used on updated bounds
- Updated bounds have different distribution (tighter, more informative)
- This causes **distribution shift** → potential accuracy degradation

### **Solution** (from previous recommendations):
- Augment MPNN training data with intermediate states (after 1-6 experiments)
- Mix initial bounds (50%) + intermediate bounds (50%)
- This reduces distribution shift during evaluation

---

## 6. Summary Table

| Aspect | Training | Testing/Evaluation |
|--------|-----------|-------------------|
| **Input Data** | Initial bounds `(w, a_lower_init, a_upper_init)` | Updated bounds `(w, a_lower_updated, a_upper_updated)` |
| **MOCU Source** | ✅ **Real MOCU** (PyCUDA) | Initial: ✅ **Real MOCU** (PyCUDA)<br>Iterative: ⚠️ **Estimated MOCU** (MPNN) |
| **Data Distribution** | Initial bounds only (iteration 0) | Updated bounds (iterations 1, 2, 3, ...) |
| **Purpose** | Learn to predict MOCU from graph structure | Use learned predictor to estimate MOCU during experiments |
| **Greedy Selection** | N/A | Uses R matrix (expected MOCU) to select experiments |

---

## 7. Code Flow Diagram

```
MPNN Training:
┌─────────────────────────────────────┐
│ generate_mocu_data.py              │
│ - Generate random systems           │
│ - Compute TRUE MOCU (PyCUDA)       │
│ - Output: (w, a_lower, a_upper,    │
│           true_MOCU)               │
└──────────────┬────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│ train_predictor.py                  │
│ - Load data (PyG format)             │
│ - Normalize targets                 │
│ - Train MPNNPlusPredictor           │
│ - Loss: MSE + Rank Loss             │
│ - Save model + statistics           │
└─────────────────────────────────────┘

MPNN Evaluation (iNN/NN):
┌─────────────────────────────────────┐
│ evaluate.py                        │
│ - Compute initial MOCU (PyCUDA)    │
│ - Load MPNN model                  │
└──────────────┬────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│ iNN_Method.select_experiment()      │
│ For each iteration:                 │
│ 1. Compute R matrix:                │
│    - For each (i,j):                │
│      - Scenario 1: syn → bounds    │
│      - Scenario 2: nonsyn → bounds │
│      - Predict MOCU (MPNN)         │
│      - R[i,j] = E[MOCU]            │
│ 2. Select (i,j) = argmin(R)        │
└──────────────┬────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│ base.py: run_episode()              │
│ - Perform experiment                 │
│ - Update bounds                      │
│ - Predict MOCU (MPNN) for updated  │
│   bounds → MOCUCurve[iter+1]       │
└─────────────────────────────────────┘
```

---

## Key Takeaways

1. **MPNN Training**: Uses **real MOCU** (PyCUDA) computed from initial bounds
2. **MPNN Testing**: Uses **estimated MOCU** (MPNN predictor) for updated bounds
3. **Greedy Selection**: Computes expected MOCU (R matrix) for all possible experiments, selects minimum
4. **iNN vs NN**: iNN re-computes R matrix at each step (adaptive), NN computes once (static)
5. **Distribution Shift**: MPNN trained on initial bounds, but used on updated bounds → potential accuracy issues
6. **Solution**: Augment training data with intermediate states (after 1-6 experiments)

