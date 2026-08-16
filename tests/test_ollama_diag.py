#!/usr/bin/env python3
import sys, os, requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.llm_util import smooth_sign_sentence, OLLAMA_URL, TRANSLATOR_MODEL

print("Ollama URL:", OLLAMA_URL)
print("Model:", TRANSLATOR_MODEL)

try:
    print("Testing direct generate API...")
    resp = requests.post(f"{OLLAMA_URL}/api/generate", json={"model": TRANSLATOR_MODEL, "prompt": "Say hi", "stream": False}, timeout=10)
    print("Status code:", resp.status_code)
    if resp.status_code != 200:
        print("Response text:", resp.text)
    else:
        print("Response:", resp.json().get("response"))
except Exception as e:
    print("Direct generate failed:", e)

try:
    print("\nCalling smooth_sign_sentence...")
    res = smooth_sign_sentence(["مدرسة", "كبير"], "Saudi Arabic Sign Language")
    print("Result:", res)
except Exception as e:
    print("Failed:", e)
