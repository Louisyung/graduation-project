"""
Visualization script for PPO training metrics
----------------------------------------------
Reads metrics.json and generates line charts for analysis

Usage:
    python plot_metrics.py --metrics runs/pico64/metrics.json --output runs/pico64/plots
"""

import json
import argparse
import os
from typing import List, Dict

import numpy as np


def load_metrics(filepath: str) -> List[Dict]:
    """Load metrics from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_metrics(metrics: List[Dict], output_dir: str = "plots"):
    """
    Create line charts for training metrics using matplotlib
    
    Args:
        metrics: List of metric entries from JSON
        output_dir: Directory to save plots
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib not installed. Install with: pip install matplotlib")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not metrics:
        print("No metrics to plot")
        return
    
    # Extract data
    iterations = [m['iteration'] for m in metrics]
    mean_returns = [m['mean_return'] for m in metrics]
    std_returns = [m['std_return'] for m in metrics]
    policy_losses = [m['policy_loss'] for m in metrics]
    value_losses = [m['value_loss'] for m in metrics]
    elapsed_times = [m['elapsed_time'] for m in metrics]
    levels = [m['level'] for m in metrics]
    
    # Define colors for levels
    level_colors = {
        "EASY": "#1f77b4",
        "C0": "#ff7f0e",
        "C1": "#2ca02c",
        "C2": "#d62728",
        "C3": "#9467bd"
    }
    
    # Plot 1: Mean Evaluation Return over iterations
    fig, ax = plt.subplots(figsize=(12, 6))
    for level in sorted(set(levels)):
        mask = [l == level for l in levels]
        x = [it for it, m in zip(iterations, mask) if m]
        y = [ret for ret, m in zip(mean_returns, mask) if m]
        color = level_colors.get(level, "#000000")
        ax.plot(x, y, marker='o', label=level, color=color, linewidth=2, markersize=4)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Mean Evaluation Return', fontsize=12)
    ax.set_title('Training Progress: Mean Evaluation Return vs Iteration', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_mean_return.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {os.path.join(output_dir, '01_mean_return.png')}")
    
    # Plot 2: Return with std dev as error bars
    fig, ax = plt.subplots(figsize=(12, 6))
    for level in sorted(set(levels)):
        mask = [l == level for l in levels]
        x = [it for it, m in zip(iterations, mask) if m]
        y = [ret for ret, m in zip(mean_returns, mask) if m]
        err = [std for std, m in zip(std_returns, mask) if m]
        color = level_colors.get(level, "#000000")
        ax.errorbar(x, y, yerr=err, marker='o', label=level, color=color, 
                   linewidth=2, markersize=4, capsize=3, alpha=0.7)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Mean Evaluation Return', fontsize=12)
    ax.set_title('Training Progress: Mean Return ± Std Dev', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_mean_return_with_std.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {os.path.join(output_dir, '02_mean_return_with_std.png')}")
    
    # Plot 3: Policy Loss over iterations
    fig, ax = plt.subplots(figsize=(12, 6))
    for level in sorted(set(levels)):
        mask = [l == level for l in levels]
        x = [it for it, m in zip(iterations, mask) if m]
        y = [loss for loss, m in zip(policy_losses, mask) if m]
        color = level_colors.get(level, "#000000")
        ax.plot(x, y, marker='s', label=level, color=color, linewidth=2, markersize=4)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Policy Loss', fontsize=12)
    ax.set_title('Training Stability: Policy Loss vs Iteration', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_policy_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {os.path.join(output_dir, '03_policy_loss.png')}")
    
    # Plot 4: Value Loss over iterations
    fig, ax = plt.subplots(figsize=(12, 6))
    for level in sorted(set(levels)):
        mask = [l == level for l in levels]
        x = [it for it, m in zip(iterations, mask) if m]
        y = [loss for loss, m in zip(value_losses, mask) if m]
        color = level_colors.get(level, "#000000")
        ax.plot(x, y, marker='^', label=level, color=color, linewidth=2, markersize=4)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Value Loss', fontsize=12)
    ax.set_title('Training Stability: Value Loss vs Iteration', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_value_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {os.path.join(output_dir, '04_value_loss.png')}")
    
    # Plot 5: Both losses on same plot (dual axis)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    ax1.plot(iterations, policy_losses, marker='s', label='Policy Loss', 
            color='#1f77b4', linewidth=2, markersize=4)
    ax2.plot(iterations, value_losses, marker='^', label='Value Loss', 
            color='#ff7f0e', linewidth=2, markersize=4)
    
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Policy Loss', fontsize=12, color='#1f77b4')
    ax2.set_ylabel('Value Loss', fontsize=12, color='#ff7f0e')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    ax1.set_title('Training Stability: Policy Loss vs Value Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_both_losses.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {os.path.join(output_dir, '05_both_losses.png')}")
    
    # Plot 6: Return vs Elapsed Time (wall-clock time)
    fig, ax = plt.subplots(figsize=(12, 6))
    elapsed_minutes = [t / 60.0 for t in elapsed_times]
    for level in sorted(set(levels)):
        mask = [l == level for l in levels]
        x = [t for t, m in zip(elapsed_minutes, mask) if m]
        y = [ret for ret, m in zip(mean_returns, mask) if m]
        color = level_colors.get(level, "#000000")
        ax.plot(x, y, marker='o', label=level, color=color, linewidth=2, markersize=4)
    
    ax.set_xlabel('Elapsed Time (minutes)', fontsize=12)
    ax.set_ylabel('Mean Evaluation Return', fontsize=12)
    ax.set_title('Training Progress: Return vs Wall-Clock Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_return_vs_time.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {os.path.join(output_dir, '06_return_vs_time.png')}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("TRAINING METRICS SUMMARY")
    print("="*60)
    for level in sorted(set(levels)):
        mask = [l == level for l in levels]
        level_returns = [ret for ret, m in zip(mean_returns, mask) if m]
        level_policy_losses = [loss for loss, m in zip(policy_losses, mask) if m]
        level_value_losses = [loss for loss, m in zip(value_losses, mask) if m]
        
        if level_returns:
            print(f"\n{level}:")
            print(f"  Returns: min={min(level_returns):.2f}, max={max(level_returns):.2f}, "
                  f"mean={np.mean(level_returns):.2f}")
            print(f"  Policy Loss: min={min(level_policy_losses):.6f}, max={max(level_policy_losses):.6f}, "
                  f"mean={np.mean(level_policy_losses):.6f}")
            print(f"  Value Loss: min={min(level_value_losses):.6f}, max={max(level_value_losses):.6f}, "
                  f"mean={np.mean(level_value_losses):.6f}")
    
    print(f"\nTotal iterations: {len(metrics)}")
    print(f"Total training time: {elapsed_times[-1] / 60:.1f} minutes")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot PPO training metrics")
    parser.add_argument("--metrics", type=str, default="runs/pico64/metrics.json",
                       help="Path to metrics.json file")
    parser.add_argument("--output", type=str, default="runs/pico64/plots",
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.metrics):
        print(f"Error: Metrics file not found: {args.metrics}")
        exit(1)
    
    print(f"Loading metrics from: {args.metrics}")
    metrics = load_metrics(args.metrics)
    print(f"Loaded {len(metrics)} metric entries")
    
    print(f"\nGenerating plots to: {args.output}")
    plot_metrics(metrics, args.output)
    print("\nDone!")
