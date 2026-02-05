#!/usr/bin/env python3
"""
GPU memory monitor using nvidia-smi parsing.
Monitors NVIDIA GB10 GPU memory usage in real-time.
"""

import subprocess
import time
import sys

def get_gpu_memory_via_smi():
    """Try to get GPU memory from nvidia-smi using various methods."""
    try:
        # Method 1: Try with --query-gpu
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,memory.free',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(',')
            if len(parts) >= 3:
                try:
                    gpu_idx, used, total = parts[0], int(parts[1]), int(parts[2])
                    return float(used), float(total)
                except (ValueError, IndexError):
                    pass
    except Exception as e:
        pass
    
    # Method 2: Parse full nvidia-smi output
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if 'Memory-Usage' in result.stdout:
            # Look for pattern like "12345 / 128000 MB"
            import re
            match = re.search(r'(\d+).*?/\s*(\d+)\s*MB', result.stdout)
            if match:
                used = int(match.group(1))
                total = int(match.group(2))
                return float(used), float(total)
    except Exception as e:
        pass
    
    return None, None

def check_gpu_memory_smi():
    """Check and display GPU memory usage via nvidia-smi."""
    used_mb, total_mb = get_gpu_memory_via_smi()
    
    if used_mb is None or total_mb is None:
        print("Unable to retrieve GPU memory info from nvidia-smi")
        print("Fallback: Use 'python check_gpu_memory.py' or check training logs")
        return
    
    used_gb = used_mb / 1024
    total_gb = total_mb / 1024
    free_gb = (total_mb - used_mb) / 1024
    pct = (used_mb / total_mb) * 100
    
    print(f"\n{'='*60}")
    print(f"GPU Memory Status (NVIDIA GB10)")
    print(f"{'='*60}")
    print(f"Total Memory:      {total_gb:7.1f} GB")
    print(f"Used:              {used_gb:7.1f} GB ({pct:5.1f}%)")
    print(f"Free:              {free_gb:7.1f} GB")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # Run once or continuously
    if len(sys.argv) > 1 and sys.argv[1] == '--continuous':
        print("Monitoring GPU memory via nvidia-smi (Ctrl+C to stop)...")
        try:
            while True:
                check_gpu_memory_smi()
                time.sleep(5)  # Check every 5 seconds
        except KeyboardInterrupt:
            print("Stopped.")
    else:
        check_gpu_memory_smi()
