import os
import sys
import zipfile
import requests
import time
from tqdm import tqdm
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset():
    print("=" * 60)
    print("KArSL-502 Dataset Downloader (Robust Resume Edition)")
    print("=" * 60)
    
    # 1. Authenticate
    print("Authenticating with Kaggle API...")
    api = KaggleApi()
    api.authenticate()
    username = api.config_values.get('username')
    key = api.config_values.get('key')
    if not username or not key:
        print("Error: Kaggle credentials not found.")
        sys.exit(1)
    
    dest_dir = r"D:\signo v6\datasets"
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "karsl-502.zip")
    url = "https://www.kaggle.com/api/v1/datasets/download/yousefdotpy/karsl-502"
    
    # Target size of the dataset
    expected_size = 25472557893
    chunk_size = 1024 * 1024  # 1MB chunks
    max_retries = 30
    retry_delay = 5
    
    # Check if download is already fully complete
    if os.path.exists(zip_path) and os.path.getsize(zip_path) == expected_size:
        print("Dataset zip already fully downloaded.")
        return zip_path
        
    print(f"Target size: {expected_size / (1024**3):.2f} GB")
    
    headers = {"User-Agent": "Kaggle-CLI/1.0"}
    
    for attempt in range(1, max_retries + 1):
        try:
            existing_size = os.path.getsize(zip_path) if os.path.exists(zip_path) else 0
            
            # If the file somehow got bigger than expected, delete it and start over
            if existing_size > expected_size:
                print(f"Existing file size ({existing_size}) is larger than expected ({expected_size}). Resetting...")
                os.remove(zip_path)
                existing_size = 0
                
            if existing_size == expected_size:
                print("Download completed successfully!")
                break
                
            if existing_size > 0:
                print(f"Resuming download from {existing_size / (1024**2):.2f} MB...")
                headers["Range"] = f"bytes={existing_size}-"
                open_mode = 'ab'
            else:
                print("Starting download from scratch...")
                open_mode = 'wb'
                
            response = requests.get(url, auth=(username, key), headers=headers, stream=True, timeout=30)
            
            # 206 is Partial Content (if Range is used), 200 is OK (if server ignores Range or downloading from scratch)
            if response.status_code not in (200, 206):
                print(f"HTTP Error {response.status_code}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                continue
                
            # If server ignored range header, reset local write pointer
            if response.status_code == 200 and existing_size > 0:
                print("Server does not support resume or ignored range header. Restarting download...")
                existing_size = 0
                open_mode = 'wb'
                
            with open(zip_path, open_mode) as f:
                with tqdm(total=expected_size, initial=existing_size, unit='B', unit_scale=True, desc="karsl-502.zip") as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            f.flush()
                            pbar.update(len(chunk))
                            
            # Verify size
            if os.path.getsize(zip_path) == expected_size:
                print("Download complete and verified!")
                break
                
        except (requests.exceptions.RequestException, Exception) as e:
            print(f"\n[Attempt {attempt}/{max_retries}] Connection lost or error: {e}")
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    else:
        print("Error: Max retries exceeded. Download failed.")
        sys.exit(1)
        
    return zip_path

def unzip_dataset(zip_path):
    extract_path = r"D:\signo v6\datasets\karsl-502"
    print(f"Extracting dataset to {extract_path}...")
    os.makedirs(extract_path, exist_ok=True)
    
    # We will use zipfile but keep track of progress and gracefully handle errors
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            files = zip_ref.namelist()
            print(f"Found {len(files)} files to extract.")
            
            # To speed up extraction of millions of small files, we can extract in blocks
            # and verify each block
            skipped_files = 0
            with tqdm(total=len(files), unit='file', desc="Extracting") as pbar:
                for file in files:
                    try:
                        zip_ref.extract(file, extract_path)
                    except Exception as e:
                        skipped_files += 1
                        # Print first few failures to debug
                        if skipped_files <= 10:
                            print(f"\nWarning: Failed to extract {file}: {e}")
                    pbar.update(1)
            if skipped_files > 0:
                print(f"\nExtraction completed with {skipped_files} files skipped due to corruption.")
            else:
                print("\nExtraction successfully completed!")
        print("Cleaning up temporary zip file...")
        os.remove(zip_path)
        print("Cleanup done.")
        
    except zipfile.BadZipFile:
        print("Error: The downloaded zip file is corrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during extraction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    zip_p = download_dataset()
    unzip_dataset(zip_p)
