import sys
import os
import time
from pathlib import Path
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parse command line arguments
parser = argparse.ArgumentParser(description='Visualize MOCU-OED results')
parser.add_argument('--N', type=int, default=5, help='Number of oscillators')
parser.add_argument('--update_cnt', type=int, default=10, help='Number of updates')
parser.add_argument('--result_folder', type=str, default='../results/', help='Results directory')
args = parser.parse_args()

update_cnt = args.update_cnt
N = args.N
resultFolder = args.result_folder

# Detect which methods have results by checking which files exist
all_possible_methods = ['iMP', 'MP', 'iODE', 'ODE', 'ENTROPY', 'RANDOM']
available_methods = []
for method in all_possible_methods:
    mocu_file = resultFolder + method + '_MOCU.txt'
    if os.path.exists(mocu_file):
        available_methods.append(method)

if not available_methods:
    print(f"Error: No result files found in {resultFolder}")
    print(f"Expected files like: *_MOCU.txt and *_timeComplexity.txt")
    sys.exit(1)

print(f"Found results for methods: {available_methods}")

# Load data for available methods
method_data = {}
for method in available_methods:
    mocu_file = resultFolder + method + '_MOCU.txt'
    time_file = resultFolder + method + '_timeComplexity.txt'
    
    mocu_data = np.loadtxt(mocu_file, delimiter="\t")
    time_data = np.loadtxt(time_file, delimiter="\t")
    
    # Handle both single run and multiple runs
    if mocu_data.ndim == 1:
        mocu_mean = mocu_data
        time_mean = time_data
    else:
        mocu_mean = mocu_data.mean(0)
        time_mean = time_data.mean(0)
    
    method_data[method] = {
        'mocu': mocu_mean,
        'time': time_mean
    }

# Define colors and markers for each method
style_map = {
    'iMP': ('r*:', 'MP (iterative)'),
    'MP': ('rs--', 'MP'),
    'iODE': ('yp:', 'ODE (iterative)'),
    'ODE': ('yo--', 'ODE'),
    'ENTROPY': ('gd:', 'Entropy'),
    'RANDOM': ('b,:', 'Random')
}

# Plot MOCU curves
x_ax = np.arange(0, update_cnt + 1, 1)
plt.figure(figsize=(10, 6))

plot_args = []
legend_labels = []
for method in available_methods:
    if method in style_map:
        style, label = style_map[method]
        plot_args.extend([x_ax, method_data[method]['mocu'], style])
        legend_labels.append(label)

plt.plot(*plot_args)
plt.legend(legend_labels)
plt.xticks(np.arange(0, update_cnt + 1, 1)) 
plt.xlabel('Number of updates')
plt.ylabel('MOCU')
plt.title(f'Experimental design for N={N} oscillators')
plt.grid(True)
plt.savefig(resultFolder + f"MOCU_{N}.png", dpi=300)
plt.close()
print(f"✓ Saved MOCU plot: {resultFolder}MOCU_{N}.png")

# Create side-by-side subplots for time complexity
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Identify slow methods (ODE-based) and fast methods
slow_methods = [m for m in available_methods if 'ODE' in m]
fast_methods = [m for m in available_methods if 'ODE' not in m]

# Left plot: All methods (log scale)
plot_args_log = []
legend_labels_log = []
for method in available_methods:
    if method in style_map:
        style, label = style_map[method]
        cumtime = np.insert(np.cumsum(method_data[method]['time']), 0, 0.0000000001)
        plot_args_log.extend([x_ax, cumtime, style])
        legend_labels_log.append(label)

if plot_args_log:
    ax1.plot(*plot_args_log)
    ax1.legend(legend_labels_log)
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of updates')
    ax1.set_ylabel('Cumulative time (seconds, log scale)')
    ax1.set_xticks(np.arange(0, update_cnt + 1, 1))
    ax1.set_title('All methods comparison (log scale)')
    ax1.grid(True)

# Right plot: Fast methods only (linear scale)
if fast_methods:
    plot_args_lin = []
    legend_labels_lin = []
    for method in fast_methods:
        if method in style_map:
            style, label = style_map[method]
            cumtime = np.insert(np.cumsum(method_data[method]['time']), 0, 0.0)
            plot_args_lin.extend([x_ax, cumtime, style])
            legend_labels_lin.append(label)
    
    ax2.plot(*plot_args_lin)
    ax2.legend(legend_labels_lin)
    ax2.set_xlabel('Number of updates')
    ax2.set_ylabel('Cumulative time (seconds, linear scale)')
    ax2.set_xticks(np.arange(0, update_cnt + 1, 1))
    ax2.set_title('Fast methods detail (linear scale)')
    ax2.grid(True)
else:
    ax2.text(0.5, 0.5, 'No fast methods available', 
             ha='center', va='center', transform=ax2.transAxes)

plt.tight_layout()
fig.savefig(resultFolder + f'timeComplexity_{N}.png', dpi=300)
plt.close(fig)
print(f"✓ Saved time complexity plot: {resultFolder}timeComplexity_{N}.png")

print(f"\n✓ Visualization complete!")
print(f"  Methods plotted: {', '.join(available_methods)}")
print(f"  Output files: MOCU_{N}.png, timeComplexity_{N}.png")

