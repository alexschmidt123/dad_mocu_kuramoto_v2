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
parser.add_argument('--update_cnt', type=int, default=None, help='Number of updates (auto-detect from data if not provided)')
parser.add_argument('--result_folder', type=str, required=True, help='Results directory to visualize')
parser.add_argument('--baseline_only', action='store_true', help='Only visualize baseline methods (exclude DAD)')
args = parser.parse_args()

N = args.N
resultFolder = args.result_folder

# Ensure resultFolder ends with path separator for proper path joining
if not resultFolder.endswith(os.sep):
    resultFolder = resultFolder + os.sep

# Keep iODE in list for index consistency, even though it's not used in experiments
# DAD: Deep Adaptive Design (new method)
# REGRESSION_SCORER: Regression Scorer (baseline method)
if args.baseline_only:
    listMethods = ['iNN', 'NN', 'iODE', 'ODE', 'ENTROPY', 'RANDOM', 'REGRESSION_SCORER']
else:
    listMethods = ['iNN', 'NN', 'iODE', 'ODE', 'ENTROPY', 'RANDOM', 'REGRESSION_SCORER', 'DAD']

# Detect which methods have results by checking which files exist
available_methods = []
for method in listMethods:
    mocu_file = os.path.join(resultFolder, f'{method}_MOCU.txt')
    if os.path.exists(mocu_file):
        available_methods.append(method)

if not available_methods:
    print(f"Error: No result files found in {resultFolder}")
    print(f"Expected files like: *_MOCU.txt and *_timeComplexity.txt")
    sys.exit(1)

print(f"Found results for methods: {available_methods}")

# Load data for available methods
method_data = {}
update_cnt_detected = None

for method in available_methods:
    mocu_file = os.path.join(resultFolder, f'{method}_MOCU.txt')
    time_file = os.path.join(resultFolder, f'{method}_timeComplexity.txt')
    
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

# Auto-detect update_cnt from maximum data length across all methods
if update_cnt_detected is None:
    if available_methods:
        max_len = max(len(method_data[method]['mocu']) for method in available_methods)
        update_cnt_detected = max_len - 1
        print(f"Auto-detected update_cnt={update_cnt_detected} from data (max length={max_len} across all methods)")
    else:
        update_cnt_detected = None

# Use provided update_cnt or auto-detected value
update_cnt = args.update_cnt if args.update_cnt is not None else update_cnt_detected
if update_cnt is None:
    print("Error: Could not determine update_cnt. Please provide --update_cnt argument.")
    sys.exit(1)

print(f"Using update_cnt={update_cnt} (will show steps 0-{update_cnt}, total {update_cnt+1} steps)")

# Verify data dimensions match
# Check all methods and use the maximum length to ensure all steps are shown
max_data_length = max(len(method_data[method]['mocu']) for method in available_methods)
if max_data_length != update_cnt + 1:
    print(f"Warning: Data has {max_data_length} values, expected {update_cnt + 1}")
    print(f"  Some methods may have different lengths. Using max length: {max_data_length}")
    # Use the maximum data length to ensure all steps are shown
    update_cnt = max_data_length - 1
    print(f"  Adjusted update_cnt to {update_cnt} (will show steps 0-{update_cnt})")
    
# Verify each method's data length
for method in available_methods:
    data_len = len(method_data[method]['mocu'])
    if data_len != update_cnt + 1:
        print(f"  Note: {method} has {data_len} values (expected {update_cnt + 1})")
        # Pad or truncate to match expected length
        if data_len < update_cnt + 1:
            # Pad with last value if data is shorter
            last_val = method_data[method]['mocu'][-1]
            padding = np.full(update_cnt + 1 - data_len, last_val)
            method_data[method]['mocu'] = np.concatenate([method_data[method]['mocu'], padding])
            print(f"    Padded {method} data to {update_cnt + 1} values")
        elif data_len > update_cnt + 1:
            # Truncate if data is longer
            method_data[method]['mocu'] = method_data[method]['mocu'][:update_cnt + 1]
            print(f"    Truncated {method} data to {update_cnt + 1} values")

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
if 'REGRESSION_SCORER' in available_methods:
    plot_args.extend([x_ax, method_data['REGRESSION_SCORER']['mocu'], 'c^--'])
    legend_labels.append('Regression Scorer')
if 'DAD' in available_methods:
    plot_args.extend([x_ax, method_data['DAD']['mocu'], 'm^-'])
    legend_labels.append('DAD (Proposed)')

plt.plot(*plot_args)
plt.legend(legend_labels)
plt.xticks(np.arange(0, update_cnt + 1, 1)) 
plt.xlabel('Number of updates')
plt.ylabel('MOCU')
plt.grid(True)
if args.baseline_only:
    mocu_plot_path = os.path.join(resultFolder, f'mocu_baseline_{N}.png')
else:
    mocu_plot_path = os.path.join(resultFolder, f'mocu_all_{N}.png')
plt.savefig(mocu_plot_path, dpi=300)
plt.close()
print(f"✓ Saved MOCU plot: {mocu_plot_path}")

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
if 'ENTROPY' in available_methods:
    plot_args.extend([x_ax, np.insert(np.cumsum(method_data['ENTROPY']['time']), 0, 0.0000000001), 'gd:'])
    legend_labels.append('Entropy')
if 'RANDOM' in available_methods:
    plot_args.extend([x_ax, np.insert(np.cumsum(method_data['RANDOM']['time']), 0, 0.0000000001), 'b,:'])
    legend_labels.append('Random')
if 'REGRESSION_SCORER' in available_methods:
    plot_args.extend([x_ax, np.insert(np.cumsum(method_data['REGRESSION_SCORER']['time']), 0, 0.0000000001), 'c^--'])
    legend_labels.append('Regression Scorer')
if 'DAD' in available_methods:
    plot_args.extend([x_ax, np.insert(np.cumsum(method_data['DAD']['time']), 0, 0.0000000001), 'm^-'])
    legend_labels.append('DAD (Proposed)')

plt.plot(*plot_args)
plt.legend(legend_labels)
plt.yscale('log')
plt.xlabel('Number of updates')
plt.ylabel('Cumulative time complexity (in seconds)')
plt.xticks(np.arange(0, update_cnt + 1, 1)) 
plt.ylim(1, 10000)
plt.grid(True)
if args.baseline_only:
    time_plot_path = os.path.join(resultFolder, f'timecomplexity_baseline_{N}.png')
else:
    time_plot_path = os.path.join(resultFolder, f'timecomplexity_all_{N}.png')
fig.savefig(time_plot_path, dpi=300)
plt.close(fig)
print(f"✓ Saved time complexity plot: {time_plot_path}")

print(f"\n✓ Visualization complete!")
print(f"  Methods plotted: {', '.join(available_methods)}")
print(f"  Output folder: {resultFolder}")
if args.baseline_only:
    print(f"  Files: mocu_baseline_{N}.png, timecomplexity_baseline_{N}.png")
else:
    print(f"  Files: mocu_all_{N}.png, timecomplexity_all_{N}.png")

