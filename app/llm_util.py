import requests
import json
import os

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/generate"

VLM_MODEL = os.environ.get("VLM_MODEL", "qwen2.5vl:3b")          # Vision Classifier
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "llama3.2:3b")        # Arbitration Judge
TRANSLATOR_MODEL = os.environ.get("TRANSLATOR_MODEL", "llama3.2:3b") # Sentence Grammar Reconstructor

def smooth_sign_sentence(words_list, dialect="Saudi Arabic Sign Language", model=TRANSLATOR_MODEL):
    """
    Takes a list of raw sign language words, queries the local Ollama LLM, 
    and returns a grammatically correct sentence in the selected dialect and its English translation.
    """
    if not words_list:
        return {"arabic": "", "english": ""}
        
    keywords_str = " -> ".join(words_list)
    
    prompt = f"""
[SYSTEM ROLE: Master Arabic Sign Language Translator & Computational Linguist]
Your task is to convert a sequence of disjointed Arabic Sign Language (ArSL) sign glosses into a natural, grammatically correct spoken sentence in the targeted dialect: {dialect}.

Input Sign Sequence: [{keywords_str}]
Target Dialect: {dialect}

[TRANSLATION GUIDELINES]
1. Reconstruct the raw keywords into a fluent, natural, and polite spoken sentence matching the grammatical syntax of {dialect}.
2. Provide an accurate, idiomatic English translation of the reconstructed sentence.
3. Do not add fictitious details not implied by the sign sequence.

[OUTPUT SPECIFICATION]
Return ONLY a valid JSON object matching this schema (no markdown, no backticks, no extra commentary):
{{
  "arabic": "Reconstructed fluent sentence in {dialect}",
  "english": "Fluent English translation"
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
            result = response.json()
            response_text = result.get("response", "").strip()
            
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            response_text = response_text.strip()
            
            data = json.loads(response_text)
            return {
                "arabic": data.get("arabic", "").strip(),
                "english": data.get("english", "").strip()
            }
            
    except Exception as e:
        print(f"Ollama integration warning: {e}")
        
    fallback_arabic = " ".join(words_list)
    return {
        "arabic": fallback_arabic,
        "english": "Ollama offline. Start Ollama and run 'ollama run qwen2.5:3b' to enable translations."
    }


def predict_sign_with_vlm(video_path: str, candidates: list, dialect: str = "Saudi Arabic Sign Language", model: str = VLM_MODEL) -> str:
    """
    Extracts storyboard frames from a video, feeds them to the local VLM (Qwen2-VL),
    and predicts which gesture candidate is performed in the video.
    """
    import base64
    from app.vlm_util import generate_storyboard

    grid_path = os.path.join(os.path.dirname(video_path), "temp_grid.jpg")
    storyboard = generate_storyboard(video_path, grid_path)
    
    if not storyboard or not os.path.exists(storyboard):
        return candidates[0]

    try:
        with open(storyboard, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")

        formatted_candidates = ", ".join([f"'{c}'" for c in candidates])
        prompt = f"""
[SYSTEM ROLE: Senior Computer Vision Analyst & Master Sign Language Expert ({dialect})]
The attached image is a 2x3 sequential temporal grid (storyboard) capturing a person signing in real-time.
Your goal is to inspect the visual evidence and select the SINGLE EXACT sign performed from the candidate list below.

Candidate Pool: [{formatted_candidates}]
Selected Regional Dialect: {dialect}

[4-STEP VISUAL EVALUATION PROTOCOL]
Step 1. Body Contact Zone: Identify if the hand contacts or moves near the Forehead, Eye level, Mouth/Chin, Chest, or Open Space.
Step 2. Hand Configuration: Observe finger extension, wrist tilt, and palm orientation across the 6 sequence frames.
Step 3. Movement Trajectory: Determine if the motion is Circular, Linear (Up/Down/Side), Pulsing, or Stationary.
Step 4. Facial Grammar: Check for eyebrow furrows, head tilts, mouth shapes, or eye movements unique to {dialect}.

[STRICT OUTPUT CONTRACT]
Output ONLY the exact string of the winning candidate word from the Candidate Pool. Do NOT include punctuation, explanations, or backticks.
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
            for candidate in candidates:
                cand_clean = candidate.strip()
                if cand_clean and cand_clean in vlm_response:
                    return cand_clean
    except Exception as e:
        print(f"VLM inference warning: {e}")

    return candidates[0]


def debate_and_decide(pred_lstm: str, lstm_confidence: float, pred_vlm: str, history: list, dialect: str = "Saudi Arabic Sign Language", model: str = JUDGE_MODEL) -> str:
    """
    Spawns the local LLM as a Judge agent to evaluate a disagreement
    between the LSTM model and the VLM model.
    """
    prompt = f"""
[SYSTEM ROLE: Chief Sign Language AI Arbitrator ({dialect})]
Two specialized AI perception models have generated conflicting predictions for a gesture. Your task is to resolve the debate and select the correct sign based on statistical confidence, visual evidence, and semantic context in {dialect}.

Contextual History (Previous Words): {history}
Regional Dialect: {dialect}

Disagreeing Models:
- Model A (Coordinate Tracking LSTM): "{pred_lstm}" (Confidence: {int(lstm_confidence * 100)}%)
- Model B (Storyboard Vision VLM): "{pred_vlm}"

[ARBITRATION PROTOCOL]
1. Evaluate semantic continuity: Which candidate makes natural linguistic sense following the sequence {history}?
2. If Model A's coordinate confidence is low (< 60%), prioritize Model B's direct visual evidence.
3. You MUST pick strictly Candidate 1 ("{pred_lstm}") OR Candidate 2 ("{pred_vlm}"). DO NOT invent any third option.

[OUTPUT FORMAT CONTRACT]
Return ONLY a valid JSON object matching this exact structure:
{{
  "decision": "insert chosen word strictly matching {pred_lstm} or {pred_vlm}",
  "reasoning": "one-line concise justification"
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
                    "temperature": 0.0,  # Zero-hallucination deterministic decoding
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
            
            if decided_word == pred_lstm:
                print(f"Judge Agent Decision: {pred_lstm} (Reason: {data.get('reasoning')})")
                return pred_lstm
            elif decided_word == pred_vlm:
                print(f"Judge Agent Decision: {pred_vlm} (Reason: {data.get('reasoning')})")
                return pred_vlm
    except Exception as e:
        print(f"Judge Agent debate failed: {e}")

    return pred_vlm if lstm_confidence < 0.6 else pred_lstm

