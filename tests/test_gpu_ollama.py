#!/usr/bin/env python3
import requests, sys

# Test GPU Ollama on port 11435 with llama3.2:3b (larger model)
url = "http://localhost:11435"
model = "llama3.2:3b"

print(f"Testing GPU Ollama at {url} with {model}...")
try:
    resp = requests.post(f"{url}/api/generate", json={"model": model, "prompt": "Say hello!", "stream": False}, timeout=30)
    print("Status Code:", resp.status_code)
    if resp.status_code == 200:
        print("Success! Response:")
        print(resp.json().get("response"))
    else:
        print("Error response:")
        print(resp.text)
except Exception as e:
    print("Failed to query GPU Ollama:", e)
