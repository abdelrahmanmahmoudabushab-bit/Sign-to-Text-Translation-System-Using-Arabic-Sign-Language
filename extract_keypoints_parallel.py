import os
import sys
import json
import argparse
import numpy as np
from PIL import Image
import mediapipe as mp
import multiprocessing as mp_pool
from tqdm import tqdm

# Add repository root to python search path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.DataLoader import DataLoader, actions, arabic_labels, N_FRAMES, N_KEYPOINTS
from app.shared import discover_sequences

# discover_sequences() is now imported from app.shared (single source of truth)

# MediaPipe wrapper function for multiprocessing with Lock to prevent collisions
def init_worker(shared_lock):
    global holistic_detector, worker_lock
    worker_lock = shared_lock
    with worker_lock:
        try:
            mp_holistic = mp.solutions.holistic
            holistic_detector = mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception as e:
            print(f"Warning: Worker failed to initialize MediaPipe Holistic at startup: {e}")
            holistic_detector = None

# Track the first inference per-process to serialize delegate setup
is_first_inference = True

def process_sequence(task_info):
    """
    Worker task: processes a single sequence folder.
    Returns (idx, keypoints_array, label) or (idx, None, None) on failure.
    """
    global holistic_detector, worker_lock, is_first_inference
    idx, (frames_dir, sign_id) = task_info
    label = actions[sign_id]
    
    try:
        # Lazy initialization fallback
        if holistic_detector is None:
            with worker_lock:
                mp_holistic = mp.solutions.holistic
                holistic_detector = mp_holistic.Holistic(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            
        # Optimization: Pre-slice frame_files to N_FRAMES to avoid wasting work on extra frames
        frame_files = sorted([
            f for f in os.listdir(frames_dir)
            if f.lower().endswith(('.jpg', '.png'))
        ])[:N_FRAMES]
        
        results_list = []
        for i, fname in enumerate(frame_files):
            image_path = os.path.join(frames_dir, fname)
            image = np.array(Image.open(image_path).convert('RGB'))
            
            # Serialize the very first inference of this process to avoid XNNPACK collisions
            if is_first_inference and i == 0:
                with worker_lock:
                    results = holistic_detector.process(image)
                    is_first_inference = False
            else:
                results = holistic_detector.process(image)
                
            keypoints = DataLoader.extract_keypoints(results)
            results_list.append(keypoints)
            
        # Pad if too few frames
        while len(results_list) < N_FRAMES:
            results_list.append(np.zeros(N_KEYPOINTS))
            
        return idx, np.array(results_list), label
        
    except Exception as e:
        return idx, None, None

def main():
    parser = argparse.ArgumentParser(description="Arabic Sign Language - Parallelized Feature Extraction")
    parser.add_argument("--data_dir", type=str, default=r"D:\signo v6\datasets\karsl-502",
                        help="Path to the KArSL-502 dataset")
    parser.add_argument("--cache_dir", type=str, default=r"D:\signo v6\datasets\keypoints_cache",
                        help="Path to save keypoints cache")
    parser.add_argument("--clean", action="store_true",
                        help="Clean existing temporary checkpoint files and start fresh")
    parser.add_argument("--max_sequences", type=int, default=None,
                        help="Limit the number of sequences to process (for quick testing)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of workers (set to 1 for stable single-process execution)")
    parser.add_argument("--start_idx", type=int, default=None,
                        help="Start index for processing sequences")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="End index for processing sequences")
    parser.add_argument("--no_export", action="store_true",
                        help="Skip exporting the final numpy arrays and mapping files")
    
    args = parser.parse_args()
    
    data_dir = args.data_dir
    cache_dir = args.cache_dir
    
    print("=" * 60)
    print("Arabic Sign Language - Parallelized Feature Extraction (Optimized)")
    print("=" * 60)
    print(f"Dataset: {data_dir}")
    print(f"Cache Output: {cache_dir}")
    print()
    
    # 1. Discover and sort sequences for deterministic index mapping
    sequences = sorted(discover_sequences(data_dir), key=lambda x: x[0])
    if not sequences:
        print("ERROR: No sequences discovered. Check dataset structure.")
        sys.exit(1)
        
    if args.max_sequences:
        sequences = sequences[:args.max_sequences]
        print(f"Limiting to first {args.max_sequences} sequences for testing.")
        
    os.makedirs(cache_dir, exist_ok=True)
    
    # 2. Checkpoint files setup using np.memmap
    cache_x_dat = os.path.join(cache_dir, "X_keypoints.dat")
    cache_y_dat = os.path.join(cache_dir, "y_labels.dat")
    progress_dat = os.path.join(cache_dir, "progress.dat")
    
    expected_x_size = len(sequences) * N_FRAMES * N_KEYPOINTS * 4 # 4 bytes for float32
    
    resume = False
    if not args.clean and os.path.exists(progress_dat) and os.path.exists(cache_x_dat) and os.path.exists(cache_y_dat):
        if os.path.getsize(cache_x_dat) == expected_x_size:
            resume = True
            print("Existing valid checkpoint found. Resuming extraction...")
        else:
            print("Checkpoint shape/size mismatch. Starting fresh.")
            
    if resume:
        X_mmap = np.memmap(cache_x_dat, dtype='float32', mode='r+', shape=(len(sequences), N_FRAMES, N_KEYPOINTS))
        y_mmap = np.memmap(cache_y_dat, dtype='int32', mode='r+', shape=(len(sequences),))
        progress_mmap = np.memmap(progress_dat, dtype='uint8', mode='r+', shape=(len(sequences),))
    else:
        # Create fresh memory-mapped files
        for p in [cache_x_dat, cache_y_dat, progress_dat]:
            if os.path.exists(p):
                os.remove(p)
        X_mmap = np.memmap(cache_x_dat, dtype='float32', mode='w+', shape=(len(sequences), N_FRAMES, N_KEYPOINTS))
        y_mmap = np.memmap(cache_y_dat, dtype='int32', mode='w+', shape=(len(sequences),))
        progress_mmap = np.memmap(progress_dat, dtype='uint8', mode='w+', shape=(len(sequences),))
        progress_mmap[:] = 0 # Mark all as unprocessed (0)
        X_mmap[:] = 0
        y_mmap[:] = -1
        progress_mmap.flush()
        X_mmap.flush()
        y_mmap.flush()

    # 3. Filter remaining tasks
    tasks = []
    for idx, seq in enumerate(sequences):
        if args.start_idx is not None and idx < args.start_idx:
            continue
        if args.end_idx is not None and idx >= args.end_idx:
            continue
        if resume and progress_mmap[idx] != 0:
            continue
        tasks.append((idx, seq))
        
    print(f"Total sequences in dataset: {len(sequences)}")
    print(f"Remaining tasks to process: {len(tasks)}")
    
    if len(tasks) == 0:
        print("All sequences already processed. Proceeding to finalize files.")
    else:
        # 4. Configure Multiprocessing with Lock to prevent collisions
        if args.workers is not None:
            num_workers = max(1, args.workers)
        else:
            num_workers = max(1, mp_pool.cpu_count() - 1)

        write_counter = 0
        if num_workers == 1:
            print("Starting extraction in single-process mode...")
            global holistic_detector, worker_lock
            mp_holistic = mp.solutions.holistic
            holistic_detector = mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            class MockLock:
                def __enter__(self): pass
                def __exit__(self, exc_type, exc_val, exc_tb): pass
            worker_lock = MockLock()
            
            pbar = tqdm(total=len(tasks), desc="Extracting Keypoints")
            for task_info in tasks:
                idx, res_arr, label = process_sequence(task_info)
                if res_arr is not None:
                    X_mmap[idx] = res_arr
                    y_mmap[idx] = label
                    progress_mmap[idx] = 1 # Mark completed (1)
                else:
                    progress_mmap[idx] = 2 # Mark failed/skipped (2)
                    
                write_counter += 1
                pbar.update(1)
                
                # Flush updates periodically to disk
                if write_counter % 10 == 0:
                    X_mmap.flush()
                    y_mmap.flush()
                    progress_mmap.flush()
            pbar.close()
        else:
            print(f"Starting extraction pool with {num_workers} parallel workers...")
            manager = mp_pool.Manager()
            lock = manager.Lock()
            
            pool = mp_pool.Pool(
                processes=num_workers,
                initializer=init_worker,
                initargs=(lock,),
                maxtasksperchild=100
            )
            
            try:
                # Use tqdm to track progress of imap_unordered
                pbar = tqdm(total=len(tasks), desc="Extracting Keypoints")
                
                for idx, res_arr, label in pool.imap_unordered(process_sequence, tasks):
                    if res_arr is not None:
                        X_mmap[idx] = res_arr
                        y_mmap[idx] = label
                        progress_mmap[idx] = 1 # Mark completed (1)
                    else:
                        progress_mmap[idx] = 2 # Mark failed/skipped (2)
                        
                    write_counter += 1
                    pbar.update(1)
                    
                    # Flush updates periodically to disk
                    if write_counter % 10 == 0:
                        X_mmap.flush()
                        y_mmap.flush()
                        progress_mmap.flush()
                        
                pbar.close()
            finally:
                pool.close()
                pool.join()
            
        X_mmap.flush()
        y_mmap.flush()
        progress_mmap.flush()
        
    if args.no_export:
        print("\nExtraction chunk complete. Skipping final export (--no_export).")
        return
        
    # Save final cache data
    print("\nExporting final numpy cache data (memory-mapped chunks)...")
    cache_x_path = os.path.join(cache_dir, "X_keypoints.npy")
    cache_y_path = os.path.join(cache_dir, "y_labels.npy")
    
    completed_indices = np.where(progress_mmap == 1)[0]
    
    # Export X_keypoints.npy in chunks to prevent memory errors
    print(f"Exporting X keypoints to {cache_x_path}...")
    X_npy = np.lib.format.open_memmap(
        cache_x_path, mode='w+', dtype='float32', 
        shape=(len(completed_indices), N_FRAMES, N_KEYPOINTS)
    )
    
    chunk_size = 5000
    for i in range(0, len(completed_indices), chunk_size):
        chunk_idx = completed_indices[i:i+chunk_size]
        X_npy[i:i+len(chunk_idx)] = X_mmap[chunk_idx]
    X_npy.flush()
    del X_npy
    
    # Export y_labels.npy
    print(f"Exporting y labels to {cache_y_path}...")
    y_npy = np.lib.format.open_memmap(
        cache_y_path, mode='w+', dtype='int32', 
        shape=(len(completed_indices),)
    )
    y_npy[:] = y_mmap[completed_indices]
    y_npy.flush()
    del y_npy
    
    # Save label mappings
    y = y_mmap[completed_indices]
    unique_labels = sorted(np.unique(y))
    n_classes = len(unique_labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    reverse_label_map = {new: old for old, new in label_map.items()}
    
    mapping_path = os.path.join(cache_dir, "label_mapping.json")
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump({
            "label_to_idx": {str(int(k)): int(v) for k, v in label_map.items()},
            "idx_to_label": {str(int(k)): int(v) for k, v in reverse_label_map.items()},
            "idx_to_arabic": {str(int(v)): arabic_labels.get(int(k), "?") for k, v in label_map.items()}
        }, f, ensure_ascii=False, indent=2)
        
    print(f"Label mapping saved to {mapping_path}")
    print("Pre-extraction complete and verified!")

if __name__ == "__main__":
    mp_pool.freeze_support()
    main()
