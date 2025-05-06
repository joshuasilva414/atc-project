import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import glob
import argparse

def safe_load_array(file_path):
    """Load a numpy array and ensure it's 1D for plotting."""
    if not os.path.exists(file_path):
        return None
    
    try:
        data = np.load(file_path, allow_pickle=True)
        
        # If data is empty, return None
        if data.size == 0:
            return None
            
        # Handle different array shapes
        if len(data.shape) == 0:  # Single scalar value
            return np.array([data])
        elif len(data.shape) == 1:  # Already 1D
            return data
        elif len(data.shape) == 2:
            if data.shape[0] == 1:  # Shape like (1, N)
                return data[0]
            elif data.shape[1] == 1:  # Shape like (N, 1)
                return data.flatten()
            else:
                # For multi-dimensional data, use the first column/feature
                return data[:, 0]
        else:  # Higher dimensional data
            # Flatten to 1D - this might not make sense for all data
            return data.flatten()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def plot_results(log_dir="logs/", save_dir="figures/", show_plots=False):
    """
    Generate visualizations from the training data collected during model training.
    
    Args:
        log_dir: Directory where logs are stored
        save_dir: Directory where to save the generated figures
        show_plots: Whether to display plots interactively
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Find all model directories
    model_dirs = []
    for model_type in ["DDPG", "PPO", "SAC"]:
        model_path = os.path.join(log_dir, model_type + "_observations")
        if os.path.exists(model_path):
            model_dirs.append(model_type)
    
    if not model_dirs:
        print(f"No model data found in {log_dir}")
        return
    
    print(f"Found data for models: {', '.join(model_dirs)}")
    
    # Plot comparison of rewards
    plt.figure(figsize=(12, 7))
    for model_name in model_dirs:
        rewards_file = os.path.join(log_dir, f"{model_name}_observations/rewards.npy")
        timesteps_file = os.path.join(log_dir, f"{model_name}_observations/timesteps.npy")
        
        timesteps = safe_load_array(timesteps_file)
        rewards = safe_load_array(rewards_file)
        
        if timesteps is not None and rewards is not None:
            # Ensure arrays have the same length for plotting
            min_len = min(len(timesteps), len(rewards))
            plt.plot(timesteps[:min_len], rewards[:min_len], label=model_name, marker='o', linestyle='-', markersize=4)
    
    plt.title("Training Reward Comparison", fontsize=14)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Episode Reward", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "reward_comparison.png"), dpi=300)
    
    # Create a dashboard of all metrics
    metrics = {
        'rewards.npy': 'Episode Reward',
        'faf_reach.npy': 'FAF Reach',
        'average_drift.npy': 'Average Drift',
        'total_intrusions.npy': 'Total Intrusions'
    }
    
    # Plot individual metrics
    for metric_file, metric_name in metrics.items():
        plt.figure(figsize=(12, 7))
        for model_name in model_dirs:
            metric_path = os.path.join(log_dir, f"{model_name}_observations/{metric_file}")
            timesteps_file = os.path.join(log_dir, f"{model_name}_observations/timesteps.npy")
            
            timesteps = safe_load_array(timesteps_file)
            metric_values = safe_load_array(metric_path)
            
            if timesteps is not None and metric_values is not None:
                # Ensure arrays have the same length for plotting
                min_len = min(len(timesteps), len(metric_values))
                if min_len > 0:  # Only plot if we have data
                    plt.plot(timesteps[:min_len], metric_values[:min_len], 
                           label=model_name, marker='o', linestyle='-', markersize=4)
        
        plt.title(f"{metric_name} Comparison", fontsize=14)
        plt.xlabel("Training Steps", fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        # Create a clean filename from the metric name
        clean_name = metric_name.lower().replace(' ', '_')
        plt.savefig(os.path.join(save_dir, f"{clean_name}_comparison.png"), dpi=300)
    
    # Create combined dashboard with all metrics in one figure
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig)
    
    for i, (metric_file, metric_name) in enumerate(metrics.items()):
        row, col = divmod(i, 2)
        ax = fig.add_subplot(gs[row, col])
        
        for model_name in model_dirs:
            metric_path = os.path.join(log_dir, f"{model_name}_observations/{metric_file}")
            timesteps_file = os.path.join(log_dir, f"{model_name}_observations/timesteps.npy")
            
            timesteps = safe_load_array(timesteps_file)
            metric_values = safe_load_array(metric_path)
            
            if timesteps is not None and metric_values is not None:
                # Ensure arrays have the same length for plotting
                min_len = min(len(timesteps), len(metric_values))
                if min_len > 0:  # Only plot if we have data
                    ax.plot(timesteps[:min_len], metric_values[:min_len], 
                          label=model_name, marker='o', linestyle='-', markersize=4)
        
        ax.set_title(metric_name, fontsize=12)
        ax.set_xlabel("Training Steps")
        ax.set_ylabel(metric_name)
        ax.grid(alpha=0.3)
        
        # Only add legend to the first subplot to avoid redundancy
        if i == 0:
            ax.legend(fontsize=10)
    
    plt.suptitle("Performance Metrics Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    plt.savefig(os.path.join(save_dir, "metrics_dashboard.png"), dpi=300)
    
    # Plot normalized metrics to compare scale-independent trends
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig)
    
    for i, (metric_file, metric_name) in enumerate(metrics.items()):
        row, col = divmod(i, 2)
        ax = fig.add_subplot(gs[row, col])
        
        for model_name in model_dirs:
            metric_path = os.path.join(log_dir, f"{model_name}_observations/{metric_file}")
            timesteps_file = os.path.join(log_dir, f"{model_name}_observations/timesteps.npy")
            
            timesteps = safe_load_array(timesteps_file)
            metric_values = safe_load_array(metric_path)
            
            if timesteps is not None and metric_values is not None:
                # Ensure arrays have the same length for plotting
                min_len = min(len(timesteps), len(metric_values))
                if min_len > 0:  # Only plot if we have data
                    # Normalize values to [0,1] for comparison
                    values_to_plot = metric_values[:min_len]
                    min_val = np.min(values_to_plot)
                    max_val = np.max(values_to_plot)
                    if max_val > min_val:  # Avoid division by zero
                        normalized = (values_to_plot - min_val) / (max_val - min_val)
                        ax.plot(timesteps[:min_len], normalized, 
                              label=model_name, marker='o', linestyle='-', markersize=4)
        
        ax.set_title(f"Normalized {metric_name}", fontsize=12)
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Normalized Value (0-1)")
        ax.set_ylim(-0.1, 1.1)  # Add some padding
        ax.grid(alpha=0.3)
        
        # Only add legend to the first subplot
        if i == 0:
            ax.legend(fontsize=10)
    
    plt.suptitle("Normalized Metrics Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    plt.savefig(os.path.join(save_dir, "normalized_metrics.png"), dpi=300)
    
    # Create progression visualization (how metrics change over training)
    if len(model_dirs) > 0:
        for model_name in model_dirs:
            timesteps_file = os.path.join(log_dir, f"{model_name}_observations/timesteps.npy")
            timesteps = safe_load_array(timesteps_file)
            
            if timesteps is None or len(timesteps) == 0:
                continue
                
            # Plot all metrics on the same graph with different y-axes
            fig, ax1 = plt.subplots(figsize=(14, 8))
            
            # Colors for different metrics
            colors = ['b', 'r', 'g', 'c']
            axes = [ax1]
            
            # Create additional y-axes
            for i in range(len(metrics) - 1):
                axes.append(ax1.twinx())
                axes[-1].spines['right'].set_position(('outward', 60 * i))
            
            for i, (metric_file, metric_name) in enumerate(metrics.items()):
                metric_path = os.path.join(log_dir, f"{model_name}_observations/{metric_file}")
                metric_values = safe_load_array(metric_path)
                
                if metric_values is not None:
                    # Ensure arrays have the same length
                    min_len = min(len(timesteps), len(metric_values))
                    if min_len > 0:  # Only plot if we have data
                        axes[i].plot(timesteps[:min_len], metric_values[:min_len], 
                                   color=colors[i], marker='o', linestyle='-', 
                                   label=metric_name, markersize=4)
                        axes[i].set_ylabel(metric_name, color=colors[i])
                        axes[i].tick_params(axis='y', labelcolor=colors[i])
            
            ax1.set_xlabel("Training Steps")
            ax1.grid(alpha=0.3)
            
            # Create a combined legend
            lines = []
            labels = []
            for ax in axes:
                axlines, axlabels = ax.get_legend_handles_labels()
                lines.extend(axlines)
                labels.extend(axlabels)
                
            ax1.legend(lines, labels, loc='best', fontsize=10)
            
            plt.title(f"{model_name} - Training Progression", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{model_name}_progression.png"), dpi=300)
            plt.close()
    
    if show_plots:
        plt.show()
    else:
        plt.close('all')
    
    print(f"All plots saved to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots from training data")
    parser.add_argument("--log_dir", type=str, default="logs/", help="Directory containing training logs")
    parser.add_argument("--save_dir", type=str, default="figures/", help="Directory to save figures")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    
    args = parser.parse_args()
    plot_results(args.log_dir, args.save_dir, args.show) 