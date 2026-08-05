"""
Pipeline monitor: waits for a keypoint extraction process to finish,
then automatically launches CNN-LSTM training.

Usage:
    python monitor_and_train.py --pid 12920
    python monitor_and_train.py --pid 12920 --epochs 50
"""

import argparse
import os
import sys
import time
import subprocess
import psutil


def main():
    parser = argparse.ArgumentParser(description="Monitor extraction and auto-launch training")
    parser.add_argument("--pid", type=int, required=True,
                       help="PID of the extraction process to monitor")
    parser.add_argument("--venv_python", type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv", "Scripts", "python.exe"),
                       help="Path to virtualenv python executable")
    parser.add_argument("--cache_dir", type=str,
                       default=r"D:\signo v6\datasets\keypoints_cache",
                       help="Path to keypoints cache directory")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Training epochs")
    parser.add_argument("--check_interval", type=int, default=300,
                       help="Seconds between status checks (default: 300)")

    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    train_script = os.path.join(project_root, "scripts", "train.py")
    log_file = os.path.join(project_root, "monitor_and_train.log")

    def log(message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        msg_str = f"[{timestamp}] {message}\n"
        print(msg_str.strip())
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg_str)

    log("="*60)
    log(f"Pipeline Monitor Started. Watching PID: {args.pid}")
    log("="*60)

    # 1. Wait for parallel keypoint extraction to complete
    while True:
        try:
            process = psutil.Process(args.pid)
            if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                log("Extraction process finished or zombie.")
                break
        except psutil.NoSuchProcess:
            log("Extraction process completed (PID no longer exists).")
            break

        log("Extraction is still running in background...")
        time.sleep(args.check_interval)

    # 2. Check if cache files were successfully created
    cache_x = os.path.join(args.cache_dir, "X_keypoints.npy")
    cache_y = os.path.join(args.cache_dir, "y_labels.npy")

    if not (os.path.exists(cache_x) and os.path.exists(cache_y)):
        log(f"ERROR: Keypoint cache files not found in {args.cache_dir}. Aborting training.")
        sys.exit(1)

    log("Keypoint cache files verified successfully.")

    # 3. Launch training
    log(f"Launching CNN-LSTM model training ({args.epochs} epochs)...")
    cmd = [
        args.venv_python,
        train_script,
        "--epochs", str(args.epochs),
        "--data_dir", r"D:\signo v6\datasets\karsl-502",
        "--cache_dir", args.cache_dir
    ]

    try:
        training_log = os.path.join(project_root, "training_run.log")
        with open(training_log, "w", encoding="utf-8") as out:
            process = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, text=True)
            log(f"Training started with PID: {process.pid}")
            process.wait()

        if process.returncode == 0:
            log("TRAINING COMPLETED SUCCESSFULLY!")
        else:
            log(f"Training failed with exit code: {process.returncode}")

    except Exception as e:
        log(f"Error executing training: {e}")

if __name__ == "__main__":
    main()
