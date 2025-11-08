"""
ODE-based OED strategy (Original MOCU method)

This module implements the original ODE-based optimal experimental design
strategy for coupled oscillator systems.
"""

import time
import numpy as np
from src.core.mocu_cuda import *


def findMOCUSequence(syncThresholds, isSynchronized, MOCUInitial, K_max, w, N, h, MVirtual, MReal, 
                     TVirtual, TReal, aLowerBoundIn, aUpperBoundIn, it_idx, update_cnt, iterative=True):
    """
    Find optimal experiment sequence using ODE-based MOCU strategy.
    
    Args:
        syncThresholds: Critical coupling strength matrix (synchronization threshold)
        isSynchronized: Synchronization state matrix (0=not sync, 1=sync)
        MOCUInitial: Initial MOCU value
        K_max: Maximum Monte Carlo samples
        w: Natural frequencies
        N: Number of oscillators
        h: Time step (deltaT)
        MVirtual: Virtual time steps for prediction
        MReal: Real time steps for MOCU computation
        TVirtual: Virtual time horizon
        TReal: Real time horizon
        aLowerBoundIn: Lower bound matrix for coupling strengths
        aUpperBoundIn: Upper bound matrix for coupling strengths
        it_idx: Number of MOCU averaging iterations
        update_cnt: Number of sequential experiments
        iterative: If True, recompute MOCU at each step (iODE); if False, compute once (ODE)
    
    Returns:
        MOCUCurve: Array of MOCU values at each step
        optimalExperiments: List of selected experiment pairs (i, j)
        timeComplexity: Array of computation times for each iteration
    """
    
    pseudoRandomSequence = True
    
    MOCUCurve = np.ones(update_cnt + 1) * 50.0
    MOCUCurve[0] = MOCUInitial
    timeComplexity = np.ones(update_cnt)
    
    aUpperBoundUpdated = aUpperBoundIn.copy()
    aLowerBoundUpdated = aLowerBoundIn.copy()
    
    optimalExperiments = []
    isInitiallyComputed = False
    R = np.zeros((N, N))
    
    for iteration in range(1, update_cnt + 1):
        iterationStartTime = time.time()
        if (not isInitiallyComputed) or iterative:
            # Computing the expected remaining MOCU
            for i in range(N):
                for j in range(i + 1, N):
                    isInitiallyComputed = True
                    if (i, j) not in optimalExperiments:
                        aUpper = aUpperBoundUpdated.copy()
                        aLower = aLowerBoundUpdated.copy()

                        w_i = w[i]
                        w_j = w[j]
                        f_inv = 0.5 * np.abs(w_i - w_j)

                        aLower[j, i] = max(f_inv, aLower[i, j])
                        aLower[i, j] = max(f_inv, aLower[i, j])

                        a_tilde = min(max(f_inv, aLowerBoundUpdated[i, j]), aUpperBoundUpdated[i, j])
                        P_syn = (aUpperBoundUpdated[i, j] - a_tilde) / (aUpperBoundUpdated[i, j] - aLowerBoundUpdated[i, j])

                        it_temp_val = np.zeros(it_idx)
                        for l in range(it_idx):
                            it_temp_val[l] = MOCU(K_max, w, N, h, MVirtual, TVirtual, aLower, aUpper, 0)
                        MOCU_matrix_syn = np.mean(it_temp_val)

                        aUpper = aUpperBoundUpdated.copy()
                        aLower = aLowerBoundUpdated.copy()

                        aUpper[i, j] = min(f_inv, aUpper[i, j])
                        aUpper[j, i] = min(f_inv, aUpper[i, j])

                        P_nonsyn = (a_tilde - aLowerBoundUpdated[i, j]) / (aUpperBoundUpdated[i, j] - aLowerBoundUpdated[i, j])

                        it_temp_val = np.zeros(it_idx)
                        for l in range(it_idx):
                            it_temp_val[l] = MOCU(K_max, w, N, h, MVirtual, TVirtual, aLower, aUpper, 0)
                        MOCU_matrix_nonsyn = np.mean(it_temp_val)
                        R[i, j] = P_syn * MOCU_matrix_syn + P_nonsyn * MOCU_matrix_nonsyn

        print(R)

        min_ind = np.where(R == np.min(R[np.nonzero(R)]))

        if len(min_ind[0]) == 1:
            min_i_MOCU = int(min_ind[0])
            min_j_MOCU = int(min_ind[1])
        else:
            min_i_MOCU = int(min_ind[0][0])
            min_j_MOCU = int(min_ind[1][0])

        iterationTime = time.time() - iterationStartTime
        timeComplexity[iteration - 1] = iterationTime

        optimalExperiments.append((min_i_MOCU, min_j_MOCU))
        R[min_i_MOCU, min_j_MOCU] = 0.0
        f_inv = syncThresholds[min_i_MOCU, min_j_MOCU]
        
        if isSynchronized[min_i_MOCU, min_j_MOCU] == 0.0:
            aUpperBoundUpdated[min_i_MOCU, min_j_MOCU] \
                = min(aUpperBoundUpdated[min_i_MOCU, min_j_MOCU], f_inv)
            aUpperBoundUpdated[min_j_MOCU, min_i_MOCU] \
                = min(aUpperBoundUpdated[min_i_MOCU, min_j_MOCU], f_inv)
        else:
            aLowerBoundUpdated[min_i_MOCU, min_j_MOCU] \
                = max(aLowerBoundUpdated[min_i_MOCU, min_j_MOCU], f_inv)
            aLowerBoundUpdated[min_j_MOCU, min_i_MOCU] \
                = max(aLowerBoundUpdated[min_i_MOCU, min_j_MOCU], f_inv)

        print("Iteration: ", iteration, ", selected: (", min_i_MOCU, min_j_MOCU, ")", iterationTime, "seconds")

        it_temp_val = np.zeros(it_idx)
        for l in range(it_idx):
            it_temp_val[l] = MOCU(K_max, w, N, h, MReal, TReal, aLowerBoundUpdated, aUpperBoundUpdated, 0)
        MOCUCurve[iteration] = np.mean(it_temp_val)  
        print("before adjusting")
        print(MOCUCurve[iteration])
        if MOCUCurve[iteration] > MOCUCurve[iteration - 1]:
            MOCUCurve[iteration] = MOCUCurve[iteration - 1]
        print("The end of iteration: actual MOCU", MOCUCurve[iteration])
    print(optimalExperiments)
    return MOCUCurve, optimalExperiments, timeComplexity

