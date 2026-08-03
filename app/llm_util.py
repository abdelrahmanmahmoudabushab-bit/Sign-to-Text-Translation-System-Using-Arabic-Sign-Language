import requests
import json
import os

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/generate"
DEFAULT_MODEL = "qwen2.5:3b" # Or gemma2:2b

def smooth_sign_sentence(words_list, dialect="Saudi Arabic Sign Language", model=DEFAULT_MODEL):
    """
    Takes a list of raw sign language words, queries the local Ollama LLM, 
    and returns a grammatically correct sentence in the selected dialect and its English translation.
    """
    if not words_list:
        return {"arabic": "", "english": ""}
        
    keywords_str = " ".join(words_list)
    
    # Prompt instructing the LLM to output a clean JSON structure adapted to regional dialect grammar
    prompt = f"""
Act as an expert sign language translator.
You will be given a sequence of raw Arabic sign language keywords. 
Your task is to:
1. Reconstruct these keywords into a grammatically correct, natural, and polite spoken Arabic sentence in the specified regional dialect/language context: {dialect}.
2. Translate that reconstructed sentence into a natural English sentence.

Input Keywords: {keywords_str}
Selected Dialect/Sign Language Context: {dialect}

Output format MUST be a valid JSON object matching this structure EXACTLY (do not include markdown formatting or backticks):
{{
  "arabic": "reconstructed sentence in the target dialect here",
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


def predict_sign_with_vlm(video_path: str, candidates: list, dialect: str = "Saudi Arabic Sign Language", model: str = "qwen2-vl:2b") -> str:
    """
    Extracts storyboard frames from a video, feeds them to the local VLM (Qwen2-VL),
    and predicts which gesture candidate is performed in the video.
    """
    import base64
    from app.vlm_util import generate_storyboard

    grid_path = os.path.join(os.path.dirname(video_path), "temp_grid.jpg")
    storyboard = generate_storyboard(video_path, grid_path)
    
    if not storyboard or not os.path.exists(storyboard):
        # Fallback to top LSTM candidate if storyboard generation fails
        return candidates[0]

    try:
        with open(storyboard, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")

        prompt = f"""
Act as an expert sign language translator specializing in the regional dialect: {dialect}.
The attached image is a 2x3 storyboard containing sequential frames of a person signing.
Identify which gesture word from the candidates list is being performed in the sequence.

Candidates: {candidates}
Selected Dialect: {dialect}

Pay close attention to the signer's:
1. Facial expressions (eyebrow furrowing, mouth open).
2. Head and neck movements (head tilts, nods, shakes, neck posture).
3. Complex hand shapes and movements, including finger twists, palm orientations, wrist rotations, and hand-to-body contact in {dialect}.

Response MUST be ONLY the correct word from the candidates list. Do not output anything else.
"""

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "images": [img_base64],
                "stream": False,
                "options": {
                    "temperature": 0.1
                }
            },
            timeout=15
        )

        if response.status_code == 200:
            vlm_response = response.json().get("response", "").strip()
            # Try to match one of the candidate words in the response
            for candidate in candidates:
                if candidate.strip() in vlm_response:
                    return candidate.strip()
    except Exception as e:
        print(f"VLM inference warning: {e}")

    return candidates[0]


def debate_and_decide(pred_lstm: str, lstm_confidence: float, pred_vlm: str, history: list, dialect: str = "Saudi Arabic Sign Language", model: str = DEFAULT_MODEL) -> str:
    """
    Spawns the local LLM as a Judge agent to evaluate a disagreement
    between the LSTM model and the VLM model.
    """
    prompt = f"""
Act as an expert Arabic Sign Language Translation Judge specializing in: {dialect}.
Two AI model pipelines disagree on a gesture prediction and need you to resolve the debate based on the semantic rules of {dialect}.

Previous Translated Words Context: {history}
Selected Dialect/Sign Language Context: {dialect}

- LSTM Model (coordinate tracking) predicted: "{pred_lstm}" with {int(lstm_confidence * 100)}% confidence
- VLM Model (direct storyboard visual context) predicted: "{pred_vlm}"

Determine which prediction is more likely correct based on the grammatical and semantic flow in the specified regional context ({dialect}).
If both fit equally well, default to the VLM's prediction because its visual analysis is more robust.

Output your decision strictly as a valid JSON object matching this structure (no backticks, no markdown):
{{
  "decision": "correct predicted word here",
  "reasoning": "one-line reasoning here"
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
                    "temperature": 0.2
                }
            },
            timeout=8
        )
        if response.status_code == 200:
            response_text = response.json().get("response", "").strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            response_text = response_text.strip()
            
            data = json.loads(response_text)
            decided_word = data.get("decision", "").strip()
            
            # Ensure the decided word is one of the two choices
            if decided_word in (pred_lstm, pred_vlm):
                print(f"Judge Agent Decision: {decided_word} (Reason: {data.get('reasoning')})")
                return decided_word
    except Exception as e:
        print(f"Judge Agent debate failed: {e}")

    # Fallback if debate fails: choose the VLM prediction
    return pred_vlm

