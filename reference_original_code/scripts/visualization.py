import sys
import os
import time
import argparse

sys.path.append("./src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parse command line arguments
parser = argparse.ArgumentParser(description='Visualize MOCU-OED results')
parser.add_argument('--N', type=int, default=5, help='Number of oscillators')
parser.add_argument('--update_cnt', type=int, default=10, help='Number of updates')
parser.add_argument('--result_folder', type=str, required=True, help='Results directory to visualize')
args = parser.parse_args()

update_cnt = args.update_cnt
N = args.N
resultFolder = args.result_folder

# Keep iODE in list for index consistency, even though it's not used in experiments
listMethods = ['iNN', 'NN', 'iODE', 'ODE', 'ENTROPY', 'RANDOM']

# Detect which methods have results by checking which files exist
available_methods = []
for method in listMethods:
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

# Plot MOCU curves
x_ax = np.arange(0, update_cnt + 1, 1)
plt.figure()

# Build plot dynamically based on available methods
plot_args = []
legend_labels = []

if 'iNN' in available_methods:
    plot_args.extend([x_ax, method_data['iNN']['mocu'], 'r*:'])
    legend_labels.append('Proposed (iterative)')
if 'NN' in available_methods:
    plot_args.extend([x_ax, method_data['NN']['mocu'], 'rs--'])
    legend_labels.append('Proposed')
if 'ODE' in available_methods:
    plot_args.extend([x_ax, method_data['ODE']['mocu'], 'yo--'])
    legend_labels.append('ODE')
if 'ENTROPY' in available_methods:
    plot_args.extend([x_ax, method_data['ENTROPY']['mocu'], 'gd:'])
    legend_labels.append('Entropy')
if 'RANDOM' in available_methods:
    plot_args.extend([x_ax, method_data['RANDOM']['mocu'], 'b,:'])
    legend_labels.append('Random')

plt.plot(*plot_args)
plt.legend(legend_labels)
plt.xticks(np.arange(0, update_cnt + 1, 1)) 
plt.xlabel('Number of updates')
plt.ylabel('MOCU')
plt.grid(True)
plt.savefig(resultFolder + f"MOCU_{N}.png", dpi=300)
plt.close()
print(f"✓ Saved MOCU plot: {resultFolder}MOCU_{N}.png")

# Plot time complexity
x_ax = np.arange(0, update_cnt + 1, 1)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

plot_args = []
legend_labels = []

if 'iNN' in available_methods:
    plot_args.extend([x_ax, np.insert(np.cumsum(method_data['iNN']['time']), 0, 0.0000000001), 'r*:'])
    legend_labels.append('Proposed (iterative)')
if 'NN' in available_methods:
    plot_args.extend([x_ax, np.insert(np.cumsum(method_data['NN']['time']), 0, 0.0000000001), 'rs--'])
    legend_labels.append('Proposed')
if 'ODE' in available_methods:
    plot_args.extend([x_ax, np.insert(np.cumsum(method_data['ODE']['time']), 0, 0.0000000001), 'yo--'])
    legend_labels.append('ODE')

plt.plot(*plot_args)
plt.legend(legend_labels)
plt.yscale('log')
plt.xlabel('Number of updates')
plt.ylabel('Cumulative time complexity (in seconds)')
plt.xticks(np.arange(0, update_cnt + 1, 1)) 
plt.ylim(1, 10000)
plt.grid(True)
fig.savefig(resultFolder + f'timeComplexity_{N}.png', dpi=300)
plt.close(fig)
print(f"✓ Saved time complexity plot: {resultFolder}timeComplexity_{N}.png")

print(f"\n✓ Visualization complete!")
print(f"  Methods plotted: {', '.join(available_methods)}")
print(f"  Output folder: {resultFolder}")
print(f"  Files: MOCU_{N}.png, timeComplexity_{N}.png")

