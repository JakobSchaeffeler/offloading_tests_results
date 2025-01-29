import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import math
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.cm import hsv

def plot(results_dir):
    import matplotlib.pyplot as plt
    import pandas as pd
    import os

    # Markers for omp and sycl
    markers = {'omp': 'o', 'sycl': 'x'}
    csv_file = results_dir 
    # Generate a color map for consistent coloring across CSV files
    #color_map = plt.cm.get_cmap('tab10')
    color_map = generate_colormap(100)

    # Keep track of unique legend entries
    legend_entries = []
    
    implementations = ['omp', 'sycl', 'hip']
    metric_totals = {impl: {} for impl in implementations}
    metric_counts = {impl: {} for impl in implementations}
    runtime_omp = {}
    runtime_hip = {}
    runtime_sycl = {}
    if csv_file.endswith('.csv'):
        file_path = csv_file
        print(csv_file)
        # Read CSV into a DataFrame
        df = pd.read_csv(file_path, index_col=0)
        # Iterate over rows and columns to accumulate metrics
        for metric, row in df.iterrows():
            for i, impl in enumerate(implementations):
                value = row[i]  # Select the appropriate column by index
                if "per wave" in metric:
                    continue
                if "DtoH" in metric or "HtoD" in metric:
                    continue
                if metric not in metric_totals[impl]:
                    metric_totals[impl][metric] = 0
                    metric_counts[impl][metric] = 0
                metric_totals[impl][metric] += value
                metric_counts[impl][metric] += 1
                if metric == "Runtime":
                    if impl == "omp":
                        runtime_omp[csv_file] = value
                    if impl == "sycl":
                        runtime_sycl[csv_file] = value
                    if impl == "hip":
                        runtime_hip[csv_file] = value


    relative_runtime_omp = {
        key: runtime_omp[key] / runtime_hip[key]
        for key in runtime_omp if key in runtime_hip and runtime_hip[key] != 0
    }
    relative_runtime_sycl = {
        key: runtime_sycl[key] / runtime_hip[key]
        for key in runtime_sycl if key in runtime_hip and runtime_hip[key] != 0
    }


    # Calculate averages
    average_metrics = {impl: {} for impl in implementations}
    for impl in implementations:
        for metric in metric_totals[impl]:
            total = metric_totals[impl][metric]
            count = metric_counts[impl][metric]
            average_metrics[impl][metric] = total / count if count > 0 else 0
    
    remove_metrics = []
    # Print averages
    print("Average Metrics Across All CSV Files:")
    for impl in implementations:
        print(f"\nImplementation: {impl}")
        for metric, avg in average_metrics[impl].items():
            print(f"  {metric}: {avg:.2f}")
    relative_metrics = {'omp': {}, 'sycl': {}}
    for metric in average_metrics['hip']:
        hip_value = average_metrics['hip'][metric]
        if hip_value > 0:  # Avoid division by zero
            for impl in ['omp', 'sycl']:
                relative_metrics[impl][metric] = average_metrics[impl][metric] / hip_value
        elif average_metrics['omp'][metric] > 0 or average_metrics['sycl'][metric] > 0:
            for impl in ['omp', 'sycl']:
                relative_metrics[impl][metric] = 0
        else:
            remove_metrics.append(metric)

    average_metrics['hip'] = {key: value for key, value in average_metrics['hip'].items() if key not in remove_metrics}

    # Plotting relative metrics
    metrics = list(a for a in average_metrics['hip'].keys())  # Metric names
    x = range(len(metrics))  # X-axis positions

    plt.figure(figsize=(14, 8))
    bar_width = 0.35
    plt.bar(x, relative_metrics['omp'].values(), bar_width, label='OMP', color='blue', alpha=0.7)
    plt.bar([i + bar_width for i in x], relative_metrics['sycl'].values(), bar_width, label='SYCL', color='orange', alpha=0.7)


    # Add labels and titles
    plt.axhline(y=1, color='red', linestyle='--', linewidth=1, label='Baseline (HIP)')
    plt.xticks([i + bar_width / 2 for i in x], metrics, rotation=45, ha='right')
    plt.ylabel('Factor Relative to HIP')
    plt.title('Metrics Relative to HIP')
    plt.legend()

    # Show/Save the plot
    plt.tight_layout()
    #plt.show()
    plt.savefig("relative_hip.pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot relative runtimes  (OMP & SYCL relative to HIP/CUDA).")
    parser.add_argument(
        "csv_file",
        type=str,
        help="csv file to generate plot for"
    )

    args = parser.parse_args()

    plot(args.results_dir)
