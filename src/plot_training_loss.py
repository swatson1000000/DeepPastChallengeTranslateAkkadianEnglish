#!/usr/bin/env python3
"""
Training Loss Visualization Script
Reads training logs and creates a plot of training and validation loss.
"""

import os
import re
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_training_log(log_file):
    """Parse training log and extract loss values."""
    folds_data = {}
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                # Look for lines with epoch information
                # Pattern: "Fold X | Epoch YYY/ZZZ | Train Loss: X.XXXX | Val Loss: X.XXXX | Ratio: XXx"
                match = re.search(
                    r'Fold (\d+) \| Epoch\s+(\d+)/(\d+) \| Train Loss: ([\d.]+|\s*nan) \| Val Loss: ([\d.]+|\s*nan)',
                    line
                )
                
                if match:
                    fold_num = int(match.group(1))
                    epoch = int(match.group(2))
                    train_loss_str = match.group(4).strip()
                    val_loss_str = match.group(5).strip()
                    
                    # Convert to float, handle NaN
                    train_loss = float(train_loss_str) if train_loss_str.lower() != 'nan' else np.nan
                    val_loss = float(val_loss_str) if val_loss_str.lower() != 'nan' else np.nan
                    
                    if fold_num not in folds_data:
                        folds_data[fold_num] = {'epochs': [], 'train_loss': [], 'val_loss': []}
                    
                    folds_data[fold_num]['epochs'].append(epoch)
                    folds_data[fold_num]['train_loss'].append(train_loss)
                    folds_data[fold_num]['val_loss'].append(val_loss)
    
    except FileNotFoundError:
        print(f"Error: Log file '{log_file}' not found")
        return None
    
    return folds_data if folds_data else None

def plot_losses(folds_data, output_file=None):
    """Create and display loss plots."""
    if not folds_data:
        print("No training data found in log file")
        return
    
    num_folds = len(folds_data)
    
    # Create figure with subplots
    fig, axes = plt.subplots(
        num_folds, 1, 
        figsize=(12, 4 * num_folds),
        sharex=False
    )
    
    # Handle single fold case (axes is not an array)
    if num_folds == 1:
        axes = [axes]
    
    # Plot each fold
    for fold_num in sorted(folds_data.keys()):
        data = folds_data[fold_num]
        epochs = data['epochs']
        train_loss = data['train_loss']
        val_loss = data['val_loss']
        
        ax = axes[fold_num - 1]
        
        # Filter out NaN values for plotting
        valid_train = [(e, l) for e, l in zip(epochs, train_loss) if not np.isnan(l)]
        valid_val = [(e, l) for e, l in zip(epochs, val_loss) if not np.isnan(l)]
        
        if valid_train:
            train_epochs, train_losses = zip(*valid_train)
            ax.plot(train_epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=4)
        
        if valid_val:
            val_epochs, val_losses = zip(*valid_val)
            ax.plot(val_epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=4)
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title(f'Fold {fold_num} - Training & Validation Loss', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or display
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to {output_file}")
    else:
        plt.show()

def main():
    # Find the most recent training log
    log_dir = Path('/home/swatson/work/MachineLearning/kaggle/DeepPastChallengeTranslateAkkadianEnglish/log')
    
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        # Find most recent train_*.log file
        log_files = sorted(log_dir.glob('train_*.log'), reverse=True)
        if not log_files:
            print("Error: No training log files found in log directory")
            print(f"Log directory: {log_dir}")
            return
        log_file = log_files[0]
    
    print(f"Reading log file: {log_file}")
    
    # Parse log
    folds_data = parse_training_log(log_file)
    
    if not folds_data:
        print("No training data found in log file")
        return
    
    # Print summary
    print(f"\n✓ Parsed {len(folds_data)} folds")
    for fold_num in sorted(folds_data.keys()):
        data = folds_data[fold_num]
        num_epochs = len(data['epochs'])
        print(f"  Fold {fold_num}: {num_epochs} epochs")
    
    # Determine output file
    output_file = None
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = str(log_dir / 'training_loss_plot.png')
    
    # Create plot
    print(f"\nGenerating plot...")
    plot_losses(folds_data, output_file)
    
    if output_file:
        print(f"\n✓ Training loss visualization complete!")
        print(f"  Output: {output_file}")
    print()

if __name__ == "__main__":
    main()
