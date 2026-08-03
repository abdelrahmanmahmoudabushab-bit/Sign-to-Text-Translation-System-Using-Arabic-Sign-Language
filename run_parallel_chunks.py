"""
Orchestrator for parallel keypoint extraction from KArSL-502 dataset.

Splits the total sequence space into N chunks and spawns separate
extract_keypoints_parallel.py processes for each, then performs a
final export to produce the merged .npy cache files.

Usage:
    python run_parallel_chunks.py
"""

import subprocess
import sys
import time
import os

def main():
    total_sequences = 75515
    num_processes = 6
    chunk_size = total_sequences // num_processes
    
    python_exe = r".venv\Scripts\python.exe"
    script_path = "extract_keypoints_parallel.py"
    cache_dir = r"D:\signo v6\datasets\keypoints_cache"
    
    # 1. Initialize the memory-mapped files first by running a quick dummy setup command
    print("Initializing memory mapped files...")
    init_cmd = [
        python_exe, "-u", script_path,
        "--cache_dir", cache_dir,
        "--clean",
        "--workers", "1",
        "--start_idx", "0",
        "--end_idx", "1",
        "--no_export"
    ]
    subprocess.run(init_cmd)
    
    # 2. Spawn the parallel chunk extraction processes
    processes = []
    for i in range(num_processes):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_processes - 1 else total_sequences
        
        cmd = [
            python_exe, "-u", script_path,
            "--cache_dir", cache_dir,
            "--workers", "1",
            "--start_idx", str(start),
            "--end_idx", str(end),
            "--no_export"
        ]
        
        print(f"Spawning Process {i+1}/{num_processes} for indices [{start}:{end}]...")
        # Open individual log files for each process so we can track them
        log_file = open(f"extract_chunk_{i+1}.log", "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((proc, log_file))
        
    print(f"\nAll {num_processes} processes successfully spawned. Monitoring progress...")
    
    # 3. Monitor execution status
    try:
        while True:
            alive = [p.poll() is None for p, _ in processes]
            if not any(alive):
                break
            
            # Periodically check progress.dat to report overall percentage
            try:
                import numpy as np
                progress_path = os.path.join(cache_dir, "progress.dat")
                if os.path.exists(progress_path):
                    p_map = np.memmap(progress_path, dtype='uint8', mode='r')
                    completed = np.sum(p_map > 0)
                    pct = completed / len(p_map) * 100
                    print(f"Overall Progress: {completed}/{len(p_map)} completed ({pct:.2f}%)")
            except Exception as e:
                print(f"Error reading progress: {e}")
                
            time.sleep(30)
    finally:
        # Clean up logs
        for _, log_file in processes:
            log_file.close()
        
    print("\nAll chunk processes finished. Performing final numpy export...")
    
    # 4. Final export run (resume mode, no start/end filters, exports final npy/mapping files)
    export_cmd = [
        python_exe, "-u", script_path,
        "--cache_dir", cache_dir,
        "--workers", "1",
        "--start_idx", "0",
        "--end_idx", "0"
    ]
    subprocess.run(export_cmd)
    print("\nPre-extraction successfully finalized!")

if __name__ == "__main__":
    main()
