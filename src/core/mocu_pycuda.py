"""
PyCUDA-based MOCU computation for high-performance data generation.

This implementation uses raw CUDA kernels via PyCUDA for maximum performance.
Based on the reference implementation from paper 2023 codes.

NOTE: This is separate from mocu.py (PyTorch version) to avoid conflicts.
PyCUDA can cause segfaults when used alongside PyTorch, so only use this
for standalone data generation (not during training).

IMPORTANT: N_global must be manually adjusted in the CUDA kernel based on 
your system size. For N oscillators, set N_global = N + 1.
"""

import time
import numpy as np
from typing import Union, Optional

# Try to import PyCUDA, but don't fail if not available
# CRITICAL: We use pycuda.autoinit for compatibility with reference code,
# but this creates a global CUDA context that conflicts with PyTorch.
# This is why data generation MUST run in separate processes.
try:
    # Use autoinit as in reference code for compatibility
    # WARNING: This creates a global CUDA context!
    import pycuda.autoinit
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False
    print("[WARNING] PyCUDA not available. Install with: pip install pycuda")

# Store compiled module per N_global value to support multiple system sizes
_compiled_modules = {}


def _get_compiled_module(N_global):
    """
    Get or compile CUDA module for given N_global.
    
    Args:
        N_global: Maximum system size (typically N+1 for N oscillators)
    
    Returns:
        Compiled SourceModule
    """
    if not PYCUDA_AVAILABLE:
        raise RuntimeError("PyCUDA not available")
    
    if N_global in _compiled_modules:
        return _compiled_modules[N_global]
    
    # CUDA kernel source - based on reference implementation
    # Note: N_global must match the system size (N+1)
    cuda_source = f"""
#include <stdio.h>

#define N_global {N_global}
#define NUMBER_FEATURES (N_global * N_global)

__device__ int mocu_comp(double *w, double h, int N, int M, double* a)
{{
    int D = 0;
    double tol, max_temp, min_temp;
    max_temp = -100.0;
    min_temp = 100.0;
    double pi_n = 3.14159265358979323846;

    double theta[N_global];
    double theta_old[N_global];
    double F[N_global], k1[N_global], k2[N_global], k3[N_global], k4[N_global];
    double diff_t[N_global];
    int i, j, k;
    double t = 0.0;
    double sum_temp;

    for (i = 0; i < N; i++) {{
        theta[i] = 0.0;
        theta_old[i] = 0.0;
        F[i] = 0.0;
        k1[i] = 0.0;
        k2[i] = 0.0;
        k3[i] = 0.0;
        k4[i] = 0.0;
        diff_t[i] = 0.0;
    }}

    for (k = 0; k < M; k++) {{
        for (i = 0; i < N; i++) {{
            sum_temp = 0.0;
            for (j = 0; j < N; j++) {{
                sum_temp += a[j*N+i] * sin(theta[j] - theta[i]);
            }}
            F[i] = w[i] + sum_temp;
        }}

        for (i = 0; i < N; i++) {{
            k1[i] = h * F[i];
            theta[i] = theta_old[i] + k1[i] / 2.0;
        }}

        for (i = 0; i < N; i++) {{
            sum_temp = 0.0;
            for (j = 0; j < N; j++) {{
                sum_temp += a[j*N+i] * sin(theta[j] - theta[i]);
            }}
            F[i] = w[i] + sum_temp;
        }}

        for (i = 0; i < N; i++) {{
            k2[i] = h * F[i];
            theta[i] = theta_old[i] + k2[i] / 2.0;
        }}

        for (i = 0; i < N; i++) {{
            sum_temp = 0.0;
            for (j = 0; j < N; j++) {{
                sum_temp += a[j*N+i] * sin(theta[j] - theta[i]);
            }}
            F[i] = w[i] + sum_temp;
        }}

        for (i = 0; i < N; i++) {{
            k3[i] = h * F[i];
            theta[i] = theta_old[i] + k3[i];
        }}

        for (i = 0; i < N; i++) {{
            sum_temp = 0.0;
            for (j = 0; j < N; j++) {{
                sum_temp += a[j*N+i] * sin(theta[j] - theta[i]);
            }}
            F[i] = w[i] + sum_temp;
        }}

        for (i = 0; i < N; i++) {{
            k4[i] = h * F[i];
            theta[i] = theta_old[i] + 1.0/6.0 * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
        }}

        for (i = 0; i < N; i++) {{
            if ((M/2) < k) {{
                diff_t[i] = (theta[i] - theta_old[i]);
            }}

            if (theta[i] > 2.0 * pi_n) {{
                theta[i] = theta[i] - 2.0 * pi_n;
            }}

            theta_old[i] = theta[i];
        }}

        if ((M/2) < k) {{
            for (i = 0; i < N; i++) {{
                if (diff_t[i] > max_temp) {{
                    max_temp = diff_t[i];
                }}
                if (diff_t[i] < min_temp) {{
                    min_temp = diff_t[i];
                }}
            }}
        }}

        t = t + h;
    }}

    tol = max_temp - min_temp;
    if (tol <= 0.001) {{
        D = 1;
    }}

    return D;
}}

__global__ void task(double *a, double *random_data, double *a_save, double *w, 
                     double h, int N, int M, double *a_lower_bound_update, 
                     double *a_upper_bound_update)
{{
    const int i_c = blockDim.x * blockIdx.x + threadIdx.x;
    int i, j;
    int rand_ind, cnt0, cnt1;

    double a_new[N_global * N_global];
    for (i = 0; i < N_global * N_global; i++) {{
        a_new[i] = 0.0;
    }}

    cnt0 = (i_c * (N - 1) * N / 2);
    cnt1 = 0;

    for (i = 0; i < N; i++) {{
        for (j = i + 1; j < N; j++) {{
            rand_ind = cnt0 + cnt1;
            a_new[j * (N + 1) + i] = a_lower_bound_update[(j * N) + i] + 
                                      (a_upper_bound_update[(j * N) + i] - 
                                       a_lower_bound_update[(j * N) + i]) * 
                                      random_data[rand_ind];
            a_new[i * (N + 1) + j] = a_new[j * (N + 1) + i];
            cnt1++;
        }}
    }}

    bool isFound = 0;
    int D;
    int iteration;
    double initialC = 0;

    for (iteration = 1; iteration < 20; iteration++) {{
        initialC = 2 * iteration;
        for (i = 0; i < N; i++) {{
            a_new[(i * (N + 1)) + N] = initialC;
            a_new[(N * (N + 1)) + i] = initialC;
        }}
        D = mocu_comp(w, h, N + 1, M, a_new);

        if (D > 0) {{
            isFound = 1;
            break;
        }}
    }}

    double c_lower = 0.0;
    double c_upper = initialC;
    double midPoint = 0;
    int iterationOffset = iteration - 1;

    if (isFound > 0) {{
        for (iteration = 0; iteration < (14 + iterationOffset); iteration++) {{
            midPoint = (c_upper + c_lower) / 2.0;

            for (i = 0; i < N; i++) {{
                a_new[(i * (N + 1)) + N] = midPoint;
                a_new[(N * (N + 1)) + i] = midPoint;
            }}
            D = mocu_comp(w, h, N + 1, M, a_new);

            if (D > 0) {{
                c_upper = midPoint;
            }}
            else {{
                c_lower = midPoint;
            }}

            if ((c_upper - c_lower) < 0.00025) {{
                break;
            }}
        }}
        a_save[i_c] = c_upper;
    }}
    else {{
        a_save[i_c] = 10000000;
    }}
}}
"""
    
    # Compile module
    mod = SourceModule(cuda_source)
    _compiled_modules[N_global] = mod
    return mod


def MOCU_pycuda(K_max: int, w: np.ndarray, N: int, h: float, M: int, T: float,
                aLowerBoundIn: np.ndarray, aUpperBoundIn: np.ndarray,
                seed: int = 0) -> float:
    """
    Compute MOCU using PyCUDA acceleration (optimized for data generation).
    
    This is a high-performance version using raw CUDA kernels that processes
    all K_max samples in parallel on GPU. Based on reference implementation.
    
    WARNING: This function creates a PyCUDA CUDA context. If PyTorch CUDA
    is used afterwards in the same process, segfaults may occur.
    Use this ONLY for standalone data generation in separate processes.
    
    Args:
        K_max: Number of Monte Carlo samples
        w: Natural frequencies [N]
        N: Number of oscillators
        h: Time step
        M: Number of time steps
        T: Time horizon (kept for compatibility, not used)
        aLowerBoundIn: Lower bounds [N, N]
        aUpperBoundIn: Upper bounds [N, N]
        seed: Random seed (0 = no seed)
    
    Returns:
        MOCU value (float)
    """
    if not PYCUDA_AVAILABLE:
        raise RuntimeError("PyCUDA not available. Install with: pip install pycuda")
    
    # Determine N_global (typically N+1 for the extended system)
    N_global = N + 1
    
    # Get or compile module for this N_global
    mod = _get_compiled_module(N_global)
    task = mod.get_function("task")
    
    # Set random seed
    if seed != 0:
        np.random.seed(seed)
    
    # Prepare extended w (add mean oscillator)
    w_extended = np.append(w, 0.5 * np.mean(w)).astype(np.float64)
    
    # Prepare bounds
    vec_a_lower = np.reshape(aLowerBoundIn.copy(), N * N).astype(np.float64)
    vec_a_upper = np.reshape(aUpperBoundIn.copy(), N * N).astype(np.float64)
    
    # Generate random coupling matrices
    if seed == 0:
        rand_data = np.random.random(int((N - 1) * N / 2.0 * K_max)).astype(np.float64)
    else:
        rng = np.random.RandomState(int(seed))
        rand_data = rng.uniform(size=int((N - 1) * N / 2.0 * K_max)).astype(np.float64)
    
    # Pre-allocate result array
    a_save = np.zeros(K_max).astype(np.float64)
    
    # Pre-allocate coupling matrix (for extended N+1 system)
    a = np.zeros((N + 1) * (N + 1)).astype(np.float64)
    
    # Launch parallel CUDA kernel
    # Use 128 blocks as in reference implementation
    blocks = 128
    block_size = int(np.ceil(K_max / blocks))
    
    try:
        task(drv.In(a), drv.In(rand_data), drv.Out(a_save), drv.In(w_extended),
             np.float64(h), np.intc(N), np.intc(M), drv.In(vec_a_lower),
             drv.In(vec_a_upper), grid=(blocks, 1), block=(block_size, 1, 1))
        
        # Synchronize to ensure computation completes
        ctx = drv.Context.get_current()
        if ctx is not None:
            ctx.synchronize()
    except Exception as e:
        raise RuntimeError(f"PyCUDA kernel execution failed: {e}. "
                          f"This may be due to CUDA context conflicts with PyTorch.")
    
    if np.min(a_save) == 0:
        print("Non sync case exists")
    
    # Compute MOCU value (same logic as reference)
    if K_max >= 1000:
        temp = np.sort(a_save)
        ll = int(K_max * 0.005)
        uu = int(K_max * 0.995)
        a_save_filtered = temp[ll - 1:uu]
        a_star = np.max(a_save_filtered)
        MOCU_val = np.sum(a_star - a_save_filtered) / (K_max * 0.99)
    else:
        a_star = np.max(a_save)
        MOCU_val = np.sum(a_star - a_save) / K_max
    
    return float(MOCU_val)
