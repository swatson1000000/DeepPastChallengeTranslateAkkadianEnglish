#!/usr/bin/env python3
"""
Quick GPU memory checker using PyTorch.
Run this in a separate terminal while training to monitor memory usage.
"""

import torch
import time
import sys

def check_gpu_memory():
    """Check and display GPU memory usage."""
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return
    
    device = torch.device('cuda:0')
    
    # Get memory info
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    free = total_memory - reserved
    
    # Convert to GB
    total_gb = total_memory / 1024**3
    allocated_gb = allocated / 1024**3
    reserved_gb = reserved / 1024**3
    free_gb = free / 1024**3
    
    # Calculate percentages
    allocated_pct = (allocated / total_memory) * 100
    reserved_pct = (reserved / total_memory) * 100
    
    print(f"\n{'='*60}")
    print(f"GPU Memory Status (NVIDIA GB10)")
    print(f"{'='*60}")
    print(f"Total Memory:      {total_gb:7.1f} GB")
    print(f"Allocated:         {allocated_gb:7.1f} GB ({allocated_pct:5.1f}%)")
    print(f"Reserved:          {reserved_gb:7.1f} GB ({reserved_pct:5.1f}%)")
    print(f"Free:              {free_gb:7.1f} GB")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # Run once or continuously
    if len(sys.argv) > 1 and sys.argv[1] == '--continuous':
        print("Monitoring GPU memory (Ctrl+C to stop)...")
        try:
            while True:
                check_gpu_memory()
                time.sleep(5)  # Check every 5 seconds
        except KeyboardInterrupt:
            print("Stopped.")
    else:
        check_gpu_memory()
