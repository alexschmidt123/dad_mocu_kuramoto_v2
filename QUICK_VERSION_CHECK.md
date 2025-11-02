# Quick Version Check Commands

## Python/PyTorch/CUDA Versions

Run the comprehensive diagnostic script:
```bash
python3 check_versions.py
```

Or check individually:

### Python Version
```bash
python3 --version
```

### PyTorch Version
```python
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'cuDNN: {torch.backends.cudnn.version()}')"
```

### PyCUDA Version
```python
python3 -c "import pycuda; print(f'PyCUDA: {pycuda.VERSION}')"
```

### System CUDA/Driver (from NVIDIA)
```bash
nvidia-smi
```

### GPU Memory Status
```bash
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
```

## Common Compatibility Issues

### 1. PyTorch CUDA Version vs System CUDA
- PyTorch is usually built against a specific CUDA version
- Your system driver must support at least that CUDA version
- Check: `torch.version.cuda` vs `nvidia-smi` output

### 2. Memory Issues
- **Out of Memory (OOM)**: Can cause segfaults
- Check with: `nvidia-smi` during training
- Solution: Reduce batch size, clear GPU cache

### 3. PyCUDA/PyTorch Conflict
- Both try to create CUDA contexts
- Solution: Use `MOCU_BACKEND=torch` when PyTorch is active
- Our code now blocks PyCUDA if PyTorch CUDA is detected

## Troubleshooting Segfaults

1. **Check GPU Memory**:
   ```bash
   nvidia-smi  # Check if GPU memory is full
   ```

2. **Clear GPU Memory**:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

3. **Check if PyCUDA is trying to initialize**:
   - Set `MOCU_BACKEND=torch` in your environment
   - Our code now blocks PyCUDA initialization if PyTorch is active

4. **Verify PyTorch can access GPU**:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0))
   x = torch.randn(10, 10).cuda()
   print(x.device)
   ```

5. **Test PyCUDA separately** (when PyTorch is NOT active):
   ```python
   import pycuda.autoinit
   import pycuda.driver as drv
   print(drv.get_version())
   ```

## Expected Output Example

From `check_versions.py`:
```
PyTorch version: 2.0.1+cu118
CUDA available: YES
CUDA version (PyTorch): 11.8
cuDNN version (PyTorch): 8.7.0
Number of GPUs: 1
GPU 0: NVIDIA GeForce RTX 3090
  Total Memory: 24.00 GB
  Allocated: 0.00 GB
  Free: 24.00 GB
```

If versions don't match or memory is low, you may experience segfaults.

