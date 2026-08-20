#!/usr/bin/env python3
"""
JSL Video Downloader

Parses datasets/jsl_manifest.json, downloads all sign language videos from web.soosl.net to 
datasets/jsl/videos/, and provides throttling, retries, and resumable logic.
"""

import os
import json
import time
import random
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MANIFEST_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "datasets",
    "jsl_manifest.json"
))
DOWNLOAD_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "datasets",
    "jsl",
    "videos"
))

# Custom headers to bypass AWS S3 CDN referrer restrictions
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://web.soosl.net/',
    'Origin': 'https://web.soosl.net'
}

def download_video(item):
    video_url = item.get("videoUrl")
    filename = item.get("videoFilename")
    
    if not video_url or not filename:
        return {"status": "skipped", "file": None, "reason": "No URL or filename"}

    dest_path = os.path.join(DOWNLOAD_DIR, filename)

    # Resume/skip if file already exists with non-zero size
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024:
        return {"status": "exists", "file": filename}

    # Be nice to SooSL servers by adding randomized delays
    time.sleep(random.uniform(0.1, 0.4))

    retries = 3
    for attempt in range(retries):
        try:
            r = requests.get(video_url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            with open(dest_path, 'wb') as out_file:
                out_file.write(r.content)
            return {"status": "success", "file": filename}
        except Exception as e:
            if attempt == retries - 1:
                # Cleanup partial files on failure
                if os.path.exists(dest_path):
                    try:
                        os.remove(dest_path)
                    except OSError:
                        pass
                return {"status": "failed", "file": filename, "error": str(e)}
            # Wait longer on retry
            time.sleep(2.0 * (attempt + 1))

def main():
    if not os.path.exists(MANIFEST_PATH):
        logger.error("JSL Manifest file not found at %s. Run crawl_soosl.py first.", MANIFEST_PATH)
        return

    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    # Filter out signs that don't have video URLs
    download_queue = [item for item in manifest if item.get("videoUrl")]
    logger.info("Found %d signs with video URLs in manifest.", len(download_queue))

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    success_count = 0
    skipped_count = 0
    failed_count = 0

    # Process downloads in a thread pool (max 5 concurrent threads to respect server bounds)
    logger.info("Starting throttled parallel video downloads...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(download_video, item): item for item in download_queue}
        
        total = len(futures)
        completed = 0
        
        for future in as_completed(futures):
            res = future.result()
            completed += 1
            status = res["status"]
            
            if status == "success":
                success_count += 1
            elif status == "exists":
                skipped_count += 1
            else:
                failed_count += 1
                logger.warning("Failed to download %s: %s", res["file"], res.get("error", "unknown error"))

            if completed % 100 == 0 or completed == total:
                logger.info("Progress: %d/%d (Success: %d, Skipped: %d, Failed: %d)",
                            completed, total, success_count, skipped_count, failed_count)

    logger.info("JSL Download session complete.")
    logger.info("Total: %d, Downloaded: %d, Existing: %d, Failed: %d",
                total, success_count, skipped_count, failed_count)

if __name__ == "__main__":
    main()
