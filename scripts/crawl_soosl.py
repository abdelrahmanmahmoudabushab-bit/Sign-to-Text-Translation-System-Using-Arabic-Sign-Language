#!/usr/bin/env python3
"""
SooSL Jordanian Sign Language Scraper

Automates a browser session via Selenium to load the Dictionary of Jordanian Sign Language on SooSL Web,
extracts the complete gloss database directly from the React client-side application state,
and populates video filenames and S3 URLs in parallel using requests ThreadPoolExecutor.
"""

import os
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SOOSL_JSL_URL = "https://web.soosl.net/projects/dictionary-of-jordanian-sign-language"
MANIFEST_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "datasets",
    "jsl_manifest.json"
)

def populate_details(gloss_item, session):
    url = "https://api-web.soosl.net/projects/get"
    url_key = gloss_item.get("url")
    if not url_key:
        return gloss_item
        
    payload = {
        "requestData": "sign",
        "reqProjectUrlKey": "dictionary-of-jordanian-sign-language",
        "reqLangId": 1,
        "reqGlossUrlKey": url_key,
        "modifiedDateTime": None
    }
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/plain, */*",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        r = session.post(url, json=payload, headers=headers, timeout=15)
        if r.status_code == 200:
            data = r.json()
            current_sign = data.get("currentSign")
            if current_sign:
                path = current_sign.get("path")
                if path:
                    gloss_item["videoFilename"] = os.path.basename(path)
                    gloss_item["videoUrl"] = f"https://websoosl-projects-prod.s3.amazonaws.com/dictionary-of-jordanian-sign-language/public{path}"
                
                # Extract English term
                senses = current_sign.get("senses", [])
                if senses:
                    gloss_texts = senses[0].get("glossTexts", [])
                    for gt in gloss_texts:
                        if gt.get("langId") == 2:
                            gloss_item["englishTerm"] = gt.get("text", "")
                            break
    except Exception as e:
        logger.warning("Failed to fetch details for %s: %s", url_key, e)
        
    return gloss_item

def crawl_jsl():
    logger.info("Initializing Selenium Web Driver...")
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
    except ImportError:
        logger.error("Selenium not installed. Install via: pip install selenium")
        return False

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        logger.info("Navigating to SooSL JSL project: %s", SOOSL_JSL_URL)
        driver.get(SOOSL_JSL_URL)

        logger.info("Waiting for React application to mount and render dictionary list...")
        wait = WebDriverWait(driver, 25)
        wait.until(EC.presence_of_element_located((By.ID, "glossesWrapper")))
        time.sleep(3.0)

        logger.info("Extracting dictionary metadata array from React Fiber state stores...")
        js_extract_script = """
        const root = document.querySelector('#root');
        if (!root) return { error: 'No React #root element found.' };
        
        let glosses = null;
        const elements = [root, ...Array.from(root.querySelectorAll('*'))];
        
        for (const el of elements) {
            const reactPropKey = Object.keys(el).find(k => k.startsWith('__reactProps') || k.startsWith('__reactEventHandlers'));
            if (reactPropKey) {
                const props = el[reactPropKey];
                if (props && props.children) {
                    const childrenArray = Array.isArray(props.children) ? props.children : [props.children];
                    for (const child of childrenArray) {
                        if (child && child.props && child.props.root && child.props.root.project) {
                            const rootStore = child.props.root;
                            if (rootStore.project && rootStore.project.glosses) {
                                glosses = rootStore.project.glosses;
                                break;
                            }
                        }
                    }
                }
            }
            if (glosses) break;
        }

        if (!glosses && window.rootStore && window.rootStore.project && window.rootStore.project.glosses) {
            glosses = window.rootStore.project.glosses;
        }

        if (!glosses) return { error: 'Glosses store not found in React fiber properties.' };

        return glosses.map(g => ({
            id: g.sortKey || null,
            glossText: g.glossText || '',
            signId: g.signId || '',
            url: g.urlKey || '',
            videoFilename: '',
            videoUrl: '',
            englishTerm: ''
        }));
        """

        extracted_data = driver.execute_script(js_extract_script)

        if isinstance(extracted_data, dict) and "error" in extracted_data:
            logger.error("Extraction script failed: %s", extracted_data["error"])
            return False

        if not extracted_data:
            logger.error("No data extracted. The glosses array is empty or undefined.")
            return False

        logger.info("Successfully extracted %d JSL signs from memory store.", len(extracted_data))

        # Respect developer slice limit if set
        limit = int(os.environ.get("SIGNO_CRAWL_LIMIT", "0"))
        if limit > 0:
            logger.info("Applying SIGNO_CRAWL_LIMIT = %d", limit)
            extracted_data = extracted_data[:limit]

        # Parallelize API queries using requests Session
        import requests
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
        session.mount("https://", adapter)
        
        logger.info("Starting threaded population of video URLs and English terms for %d signs...", len(extracted_data))
        populated_data = []
        
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = {executor.submit(populate_details, item, session): item for item in extracted_data}
            completed_count = 0
            for future in as_completed(futures):
                res = future.result()
                populated_data.append(res)
                completed_count += 1
                if completed_count % 50 == 0 or completed_count == len(extracted_data):
                    logger.info("Populated details for %d/%d signs...", completed_count, len(extracted_data))

        # Sort back to original index sorting
        populated_data.sort(key=lambda x: x.get("id") or 0)

        # Write manifest file
        os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(populated_data, f, ensure_ascii=False, indent=2)

        logger.info("Saved complete manifest to: %s", MANIFEST_PATH)
        return True

    except Exception as e:
        logger.exception("An error occurred during scraping: %s", e)
        return False
    finally:
        driver.quit()

if __name__ == "__main__":
    success = crawl_jsl()
    if not success:
        logger.warning("Scraper failed.")
