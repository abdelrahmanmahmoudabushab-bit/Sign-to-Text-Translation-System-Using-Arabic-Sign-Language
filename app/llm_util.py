import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:3b" # Or gemma2:2b

def smooth_sign_sentence(words_list, model=DEFAULT_MODEL):
    """
    Takes a list of raw sign language words, queries the local Ollama LLM, 
    and returns a grammatically correct Arabic sentence and its English translation.
    """
    if not words_list:
        return {"arabic": "", "english": ""}
        
    keywords_str = " ".join(words_list)
    
    # Prompt instructing the LLM to output a clean JSON structure
    prompt = f"""
Act as an expert sign language translator.
You will be given a sequence of raw Arabic sign language keywords. 
Your task is to:
1. Reconstruct these keywords into a grammatically correct, natural, and polite spoken Arabic sentence.
2. Translate that reconstructed sentence into a natural English sentence.

Input Keywords: {keywords_str}

Output format MUST be a valid JSON object matching this structure EXACTLY (do not include markdown formatting or backticks):
{{
  "arabic": "reconstructed Arabic sentence here",
  "english": "translated English sentence here"
}}
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3 # Low temperature for consistent formatting
                }
            },
            timeout=8 # Reasonable timeout for local SLMs on CPU/GPU
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get("response", "").strip()
            
            # Clean up potential markdown formatting block wrappers
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            response_text = response_text.strip()
            
            # Parse the JSON output from the LLM
            data = json.loads(response_text)
            return {
                "arabic": data.get("arabic", "").strip(),
                "english": data.get("english", "").strip()
            }
            
    except Exception as e:
        print(f"Ollama integration warning: {e}")
        
    # Fallback if Ollama is not running or fails
    fallback_arabic = " ".join(words_list)
    return {
        "arabic": fallback_arabic,
        "english": "Ollama offline. Start Ollama and run 'ollama run qwen2.5:3b' to enable translations."
    }
