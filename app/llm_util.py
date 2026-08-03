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

        formatted_candidates = ", ".join([f"'{c}'" for c in candidates])
        prompt = f"""
Act as an expert Arabic Sign Language Master Translator ({dialect}).
The attached image is a 2x3 sequential storyboard grid of a person performing a sign gesture.
Your objective is to pick the EXACT correct sign from the candidate list below.

Ranked Candidates: [{formatted_candidates}]
Selected Dialect: {dialect}

Visual Inspection Checklist:
1. Contact location: Is the sign performed at the forehead, chest, chin, or open air?
2. Handshape & Orientation: Are fingers extended, curved, or closed? Are palms facing inward, outward, or downward?
3. Facial Action: Is there an eyebrow furrow, nod, or mouth movement corresponding to {dialect}?

Response MUST contain ONLY the exact chosen word string from the Candidates list.
"""

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "images": [img_base64],
                "stream": False,
                "options": {
                    "temperature": 0.0,  # Zero temperature for 100% deterministic, zero-hallucination decoding
                    "top_p": 0.1
                }
            },
            timeout=15
        )

        if response.status_code == 200:
            vlm_response = response.json().get("response", "").strip()
            # Strict candidate matching — accept ONLY words explicitly present in candidates
            for candidate in candidates:
                cand_clean = candidate.strip()
                if cand_clean and cand_clean in vlm_response:
                    return cand_clean
    except Exception as e:
        print(f"VLM inference warning: {e}")

    # If VLM output didn't contain any candidate, fall back to top LSTM candidate
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

Candidate 1 (LSTM coordinate tracking): "{pred_lstm}" (Confidence: {int(lstm_confidence * 100)}%)
Candidate 2 (VLM storyboard visual): "{pred_vlm}"

CRITICAL RULE: You MUST choose EXACTLY ONE of the two candidate words above ("{pred_lstm}" OR "{pred_vlm}"). DO NOT invent or generate any other word.

Output your decision strictly as a valid JSON object matching this structure (no backticks, no markdown):
{{
  "decision": "{pred_vlm}",
  "reasoning": "one-line explanation here"
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
                    "temperature": 0.0,  # Deterministic decoding
                    "top_p": 0.1
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
            
            # Strict verification: MUST be one of the two input candidates
            if decided_word == pred_lstm:
                print(f"Judge Agent Decision: {pred_lstm} (Reason: {data.get('reasoning')})")
                return pred_lstm
            elif decided_word == pred_vlm:
                print(f"Judge Agent Decision: {pred_vlm} (Reason: {data.get('reasoning')})")
                return pred_vlm
    except Exception as e:
        print(f"Judge Agent debate failed: {e}")

    # Default to VLM if high confidence or debate parse fails
    return pred_vlm if lstm_confidence < 0.6 else pred_lstm

