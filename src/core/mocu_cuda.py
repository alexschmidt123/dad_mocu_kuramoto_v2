"""
CUDA-accelerated MOCU computation

This module is part of the MOCU-OED project for optimal experimental design
in coupled oscillator systems.

IMPORTANT: PyCUDA initialization is LAZY to avoid conflicts with PyTorch.
PyCUDA context is only created when MOCU() is actually called, not on import.
This ensures clean separation when using MPNN predictor (which uses PyTorch).
"""

import time
import numpy as np

# LAZY INITIALIZATION: Only import and initialize PyCUDA when MOCU() is called
# This prevents CUDA context conflicts with PyTorch when using MPNN predictor
_pycuda_initialized = False
_mod = None
_task = None
_drv = None

def _init_pycuda():
    """
    Lazily initialize PyCUDA context and compile CUDA kernel.
    This is called only when MOCU() is actually used, not on module import.
    """
    global _pycuda_initialized, _mod, _task, _drv
    
    if _pycuda_initialized:
        return
    
    # Now import PyCUDA (this will initialize CUDA context)
    import pycuda.autoinit
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule
    
    _drv = drv
    
    # Compile CUDA kernel
    _mod = SourceModule("""

// This should be manually changed due to the technical issue in the PyCUDA.
// Well, yes, I am lazy...
#include <stdio.h>

#define N_global 6
#define NUMBER_FEATURES (N_global * N_global)

__device__ int mocu_comp(double *w, double h, int N, int M, double* a)
{
    int D = 0;
    double tol,max_temp,min_temp;
    max_temp = -100.0;
    min_temp = 100.0;
    double pi_n = 3.14159265358979323846;

    double theta[N_global];
    double theta_old[N_global];
    double F[N_global],k1[N_global],k2[N_global],k3[N_global],k4[N_global];
    double diff_t[N_global];
    int i,j,k;
    double t = 0.0;
    double sum_temp;


    for (i=0;i<N;i++){
        theta[i] = 0.0;
        theta_old[i] = 0.0;
        F[i] = 0.0;
        k1[i] = 0.0;
        k2[i] = 0.0;
        k3[i] = 0.0;
        k4[i] = 0.0;
        diff_t[i] = 0.0;
    }

    for (k=0;k<M;k++){


        for (i=0;i<N;i++){

            sum_temp = 0.0;
            for (j=0;j<N;j++){
              sum_temp += a[j*N+i]*sin(theta[j] - theta[i]);

            }
            F[i] = w[i] + sum_temp;
        }

        for(i=0;i<N;i++){
            k1[i] = h*F[i];
            theta[i] = theta_old[i] + k1[i]/2.0;
          }



        for (i=0;i<N;i++){
            sum_temp = 0.0;
            for (j=0;j<N;j++){
              sum_temp += a[j*N+i]*sin(theta[j] - theta[i]);
            }
            F[i] = w[i] + sum_temp;
        }

        for(i=0;i<N;i++){
            k2[i] = h*F[i];
            theta[i] = theta_old[i] + k2[i]/2.0;
          }


        for (i=0;i<N;i++){
            sum_temp = 0.0;
            for (j=0;j<N;j++){
              sum_temp += a[j*N+i]*sin(theta[j] - theta[i]);
            }
            F[i] = w[i] + sum_temp;
         }
        for(i=0;i<N;i++){
            k3[i] = h*F[i];
            theta[i] = theta_old[i] + k3[i];
          }



        for (i=0;i<N;i++){
            sum_temp = 0.0;
            for (j=0;j<N;j++){
              sum_temp += a[j*N+i]*sin(theta[j] - theta[i]);
            }
            F[i] = w[i] + sum_temp;
        }


        for(i=0;i<N;i++){        
            k4[i] = h*F[i];
            theta[i] = theta_old[i] + 1.0/6.0*(k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
          }


        for (i=0;i<N;i++){
            if ((M/2) < k)
            {
             diff_t[i] = (theta[i] - theta_old[i]);
            }

             if (theta[i] > 2.0*pi_n)
             {
          		theta[i] = theta[i] - 2.0*pi_n;
            }

             theta_old[i] = theta[i];  
        }

        if ((M/2) < k){
            for(i=0;i<N;i++){
                if (diff_t[i] > max_temp)
                {
                    max_temp  = diff_t[i];
                }

                if (diff_t[i] < min_temp)
                {
                    min_temp  = diff_t[i];
                }
            }

        }

        t = t+h;

    }


    tol = max_temp-min_temp;
    if (tol <= 0.001){
        D = 1;
    }

    return D;
}

__global__ void task(double *a, double *random_data, double *a_save, double *w, \
                     double h , int N, int M, double *a_lower_bound_update, \
                    double *a_upper_bound_update)
{
    const int i_c = blockDim.x*blockIdx.x + threadIdx.x;
    int i,j;
    int observeIndex = 10000000000;

    double a_new[N_global*N_global];
    for (i=0;i<N_global*N_global;i++){
            a_new[i] = 0.0;
    }
    if (i_c == observeIndex) {
        printf("find minimum cost %d", i_c);
            for (i=0;i<N_global*N_global;i++){
            if ((i%N_global) == 0) {
                printf("\\n");
            }
            printf("a_new[%d]=%.10f\\t", i, a_new[i]);
        }
        printf("\\n");
    }
    int rand_ind, cnt0, cnt1;

    cnt0 = (i_c*(N-1)*N/2);
    cnt1 = 0;

    for (i=0;i<N;i++){
        for (j=i+1;j<N;j++)
        {
            rand_ind = cnt0 + cnt1;
            a_new[j*(N+1)+i] = a_lower_bound_update[(j*N)+i]+ (a_upper_bound_update[(j*N)+i]-a_lower_bound_update[(j*N)+i])*random_data[rand_ind];
            a_new[i*(N+1)+j] = a_new[j*(N+1)+i];
            cnt1++;
        }
    }

    if (i_c == observeIndex) {
        printf("Initialization of a_new", i_c);
            for (i=0;i<N_global*N_global;i++){
                            if ((i%N_global) == 0) {
                printf("\\n");
            }
            printf("a_new[%d]=%.10f\\t", i, a_new[i]);
        }
        printf("\\n");
    }
    bool isFound = 0;
    int D;
    int iteration;
    double initialC = 0;

    for (iteration = 1; iteration < 20; iteration++) {
        initialC = 2 * iteration;
        for (i=0;i<N;i++){
            a_new[(i*(N+1))+N] = initialC;
            a_new[(N*(N+1))+i] = initialC;
        }

        if (i_c == observeIndex) {
        printf("Find upper bound, iteration: %d, upperbound: %.10f", iteration, initialC);
            for (i=0;i<N_global*N_global;i++){
                            if ((i%N_global) == 0) {
                printf("\\n");
            }
            printf("a_new[%d]=%.10f\\t", i, a_new[i]);
            }
            printf("\\n");
        }
        D = mocu_comp(w, h, N+1, M, a_new);

        if (D > 0) {
            isFound = 1;
            break;
        }
    }

    double c_lower = 0.0;
    double c_upper = initialC;
    double midPoint = 0;
    int iterationOffset = iteration - 1;

    if (isFound > 0) {
        for (iteration = 0; iteration < (14 + iterationOffset); iteration++) {
            midPoint = (c_upper + c_lower) / 2.0;

            for (i=0;i<N;i++){
                a_new[(i*(N+1))+N] = midPoint;
                a_new[(N*(N+1))+i] = midPoint;
            }
            if (i_c == observeIndex) {
            printf("binary serach, iteration: %d, upper bound: %.10f, lower bound: %.10f", iteration, c_upper, c_lower);
                for (i=0;i<N_global*N_global;i++){
                    if ((i%N_global) == 0) {
                        printf("\\n");
                    }
                    printf("a_new[%d]=%.10f\\t", i, a_new[i]);
                }
                printf("\\n");
            }
            D = mocu_comp(w, h, N+1, M, a_new);

            if (D > 0) {  
                c_upper = midPoint;
            }
            else {  
                c_lower = midPoint;
            }

            if ((c_upper - c_lower) < 0.00025) {
                //printf("Upper - Lower is less than 0.00025\\n");
                break;
            }
        }
        a_save[i_c] = c_upper; 
    }
    else {
        printf("Can't find a! i_c: %d\\n", i_c);
        a_save[i_c] = 10000000; 
    }    
    if (i_c == observeIndex) {
        printf("binary serach end, iteration: %d, upper bound: %.10f, lower bound: %.10f", iteration, c_upper, c_lower);
        for (i=0;i<N_global*N_global;i++){
            if ((i%N_global) == 0) {
                printf("\\n");
            }
            printf("a_new[%d]=%.10f\\t", i, a_new[i]);
        }
        printf("\\n");
    }
}

"""
                   )
    
    _task = _mod.get_function("task")
    _pycuda_initialized = True


# Note: The CUDA kernel code above is compiled lazily when _init_pycuda() is called


def MOCU(K_max, w, N, h, M, T, aLowerBoundIn, aUpperBoundIn, seed):
    """
    Compute MOCU using CUDA-accelerated Monte Carlo sampling.
    
    IMPORTANT: This function initializes PyCUDA context on first call.
    If you're using MPNN predictor, this function should NEVER be called
    (use predictor_utils.predict_mocu() instead).
    """
    # Lazy initialization: Only initialize PyCUDA when MOCU is actually called
    # This prevents CUDA context conflicts with PyTorch when using MPNN predictor
    _init_pycuda()
    
    # seed = 0
    # Validate and adjust block configuration
    blocks = 128
    block_size = int(K_max / blocks)
    
    # Ensure block_size is valid (must be > 0 and power of 2 is better)
    if block_size <= 0:
        # Fallback: use fewer blocks if K_max is too small
        blocks = max(1, K_max // 32)  # Ensure at least 32 threads per block
        block_size = K_max // blocks if blocks > 0 else K_max
    
    # Ensure total threads match K_max exactly
    total_threads = blocks * block_size
    if total_threads != K_max:
        # Adjust to match exactly
        blocks = (K_max + block_size - 1) // block_size  # Ceiling division
        block_size = K_max // blocks if blocks > 0 else K_max
        total_threads = blocks * block_size
    
    # Safety check: ensure valid CUDA grid/block configuration
    if blocks <= 0 or block_size <= 0:
        raise ValueError(f"Invalid CUDA configuration: blocks={blocks}, block_size={block_size}, K_max={K_max}")
    
    if block_size > 1024:  # CUDA limit
        blocks = (K_max + 1023) // 1024
        block_size = min(1024, K_max // blocks if blocks > 0 else 1024)

    w = np.append(w, 0.5 * np.mean(w))

    a_save = np.zeros(K_max).astype(np.float64)

    vec_a_lower = np.zeros(N * N).astype(np.float64)
    vec_a_upper = np.zeros(N * N).astype(np.float64)

    vec_a_lower = np.reshape(aLowerBoundIn.copy(), N * N)
    vec_a_upper = np.reshape(aUpperBoundIn.copy(), N * N)

    a = np.zeros((N + 1) * (N + 1)).astype(np.float64)

    if (int(seed) == 0):
        rand_data = np.random.random(int((N - 1) * N / 2.0 * K_max)).astype(np.float64)
    else:
        rand_data = np.random.RandomState(int(seed)).uniform(size=int((N - 1) * N / 2.0 * K_max))

    # Add error handling for CUDA kernel launch
    try:
        _task(_drv.In(a), _drv.In(rand_data), _drv.Out(a_save), _drv.In(w),
             np.float64(h), np.intc(N), np.intc(M), _drv.In(vec_a_lower),
             _drv.In(vec_a_upper), grid=(blocks, 1), block=(block_size, 1, 1))
    except _drv.LogicError as e:
        # CUDA context or resource error - try to recover
        error_msg = str(e)
        if "cuFuncSetBlockShape" in error_msg or "invalid resource handle" in error_msg:
            # Clear CUDA context and retry once
            try:
                # Try to get current context and reset if possible
                ctx = _drv.Context.get_current()
                if ctx is not None:
                    ctx.pop()
                    ctx.push()
                
                # Retry with same configuration
                _task(_drv.In(a), _drv.In(rand_data), _drv.Out(a_save), _drv.In(w),
                     np.float64(h), np.intc(N), np.intc(M), _drv.In(vec_a_lower),
                     _drv.In(vec_a_upper), grid=(blocks, 1), block=(block_size, 1, 1))
            except Exception as retry_error:
                raise RuntimeError(
                    f"CUDA kernel launch failed: {error_msg}. "
                    f"Retry also failed: {retry_error}. "
                    f"This may be due to CUDA context conflict with PyTorch. "
                    f"Try running MOCU computation in a separate process."
                ) from e
        else:
            raise

    # print("a_save")
    # print(a_save)

    if min(a_save) == 0:
        print("Non sync case exists")

    # if K_max >= 50000:
    if K_max >= 1000:
        temp = np.sort(a_save)
        ll = int(K_max * 0.005)
        uu = int(K_max * 0.995)
        a_save = temp[ll - 1:uu]
        a_star = max(a_save)
        MOCU_val = sum(a_star - a_save) / (K_max * 0.99)

    else:
        a_star = max(a_save)
        MOCU_val = sum(a_star - a_save) / (K_max)

    return MOCU_val