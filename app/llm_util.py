import requests
import json
import logging
import os
import re

# Detect if running on NVIDIA Jetson target
is_jetson = os.path.exists("/etc/nv_tegra_release")
default_host = "http://localhost:11435" if is_jetson else "http://localhost:11434"
OLLAMA_URL = os.environ.get("OLLAMA_HOST", default_host)

VLM_MODEL = os.environ.get("VLM_MODEL", "qwen2.5vl:3b")            # Vision Classifier
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "llama3.2:3b")         # Arbitration Judge
TRANSLATOR_MODEL = os.environ.get("TRANSLATOR_MODEL", "llama3.2:3b") # Grammar Reconstructor

logger = logging.getLogger(__name__)

def smooth_sign_sentence(words_list, dialect="Saudi Arabic Sign Language", model=TRANSLATOR_MODEL):
    """
    Takes a list of raw sign language words, queries the local Ollama LLM, 
    and returns a grammatically correct sentence in the selected dialect and its English translation.
    """
    if not words_list:
        return {"arabic": "", "english": ""}
        
    is_options_format = isinstance(words_list[0], list)
    
    if is_options_format:
        options_lines = []
        for idx, opts in enumerate(words_list):
            options_lines.append(f"Position {idx + 1} Options: [{', '.join(opts)}]")
        keywords_str = "\n".join(options_lines)
        
        prompt = f"""
[SYSTEM ROLE: Master Arabic Sign Language Translator & Computational Linguist]
Your task is to review a sequence of positions, where each position contains candidate signs (ordered by model likelihood). 
Choose exactly one word from each position list to construct the most logical, natural, and grammatically correct spoken sentence in the targeted dialect: {dialect}.

Input Sign Candidate Sequences:
{keywords_str}

Target Dialect: {dialect}

[TRANSLATION GUIDELINES]
1. Choose the combination of words that forms the most contextually and grammatically correct spoken sentence in {dialect}.
2. Do not choose options that produce nonsense or illogical phrases (e.g. choice combinations that make no semantic sense).
3. Reconstruct the selected keywords into a fluent, natural, and polite spoken sentence matching the grammatical syntax of {dialect}.
4. Provide an accurate, idiomatic English translation of the reconstructed sentence.

[OUTPUT SPECIFICATION]
Return ONLY a valid JSON object matching this schema (no markdown, no backticks, no extra commentary):
{{
  "arabic": "Reconstructed fluent sentence in {dialect}",
  "english": "Fluent English translation"
}}
"""
    else:
        keywords_str = " -> ".join(words_list)
        prompt = f"""
[SYSTEM ROLE: Master Arabic Sign Language Translator & Computational Linguist]
Your task is to convert a sequence of disjointed Arabic Sign Language (ArSL) sign glosses into a natural, grammatically correct spoken sentence in the targeted dialect: {dialect}.

Input Sign Sequence: [{keywords_str}]
Target Dialect: {dialect}

[TRANSLATION GUIDELINES]
1. Reconstruct the raw keywords into a fluent, natural, and polite spoken sentence matching the grammatical syntax of {dialect}.
2. If any sign gloss contains a grammatical annotation such as "(negation)", "(question)", or "(emphasis)" (e.g. "ذهب (negation)" or "مدرسة (question)"), apply that grammatical rule to the sentence reconstruction:
   - "(negation)" means the action or concept is negated (e.g., "لم يذهب" or "ليس مدرسة").
   - "(question)" means the sentence should be phrased as a question (e.g., "هل ذهب؟" or "هل هذه مدرسة؟").
   - "(emphasis)" means the action/adjective is intensified (e.g., "ذهب بسرعة" or "مدرسة ممتازة جداً").
3. Provide an accurate, idiomatic English translation of the reconstructed sentence.
4. Do not add fictitious details not implied by the sign sequence.

[OUTPUT SPECIFICATION]
Return ONLY a valid JSON object matching this schema (no markdown, no backticks, no extra commentary):
{{
  "arabic": "Reconstructed fluent sentence in {dialect}",
  "english": "Fluent English translation"
}}
"""

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2
                }
            },
            timeout=90
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get("response", "").strip()
            
            if response_text.startswith("```"):
                # Strip markdown code fences robustly
                response_text = re.sub(r'^```(?:json|JSON)?\s*', '', response_text)
                response_text = re.sub(r'\s*```\s*$', '', response_text)
            response_text = response_text.strip()
            
            data = json.loads(response_text)
            return {
                "arabic": data.get("arabic", "").strip(),
                "english": data.get("english", "").strip()
            }
            
    except Exception as e:
        logger.warning("Ollama integration warning: %s", e)
        
    fallback_arabic = " ".join(words_list)
    return {
        "arabic": fallback_arabic,
        "english": f"Ollama offline. Start Ollama and run 'ollama run {model}' to enable translations."
    }


def query_vlm_with_frames(base64_frames: list, candidates: list, dialect: str = "Saudi Arabic Sign Language", model: str = VLM_MODEL) -> tuple:
    """
    Feeds a sequence of base64 video frames to the local VLM (Qwen2-VL),
    and predicts which gesture candidate is performed, along with any grammar markers.
    Returns:
        (predicted_sign, grammar_annotation) - e.g., ("ذهب", " (negation)")
    """
    if not base64_frames:
        return (candidates[0] if candidates else "", "")

    try:
        formatted_candidates = ", ".join([f"'{c}'" for c in candidates])
        prompt = f"""
[SYSTEM ROLE: Senior Computer Vision Analyst & Master Arabic Sign Language Expert ({dialect})]
The attached images are a chronological sequence of frames from a video capturing a person signing in real-time.
Your goal is to:
1. Analyze the movement trajectory, timing, and transitions across these sequential images to select the SINGLE EXACT sign performed from the candidate list below.
2. Analyze the signer's facial expression, head movements, and body language to detect non-manual grammatical markers (negation, question, or emphasis).

Candidate Pool: [{formatted_candidates}]
Selected Regional Dialect: {dialect}

[4-STEP VISUAL EVALUATION PROTOCOL]
Step 1. Body Contact Zone: Identify if the hand contacts or moves near the Forehead, Eye level, Mouth/Chin, Chest, or Open Space.
Step 2. Hand Configuration: Observe finger extension, wrist tilt, and palm orientation across the sequence of frames.
Step 3. Movement Trajectory & Transitions: Analyze the motion sequence (Circular, Linear Up/Down/Side, Pulsing, or Stationary) across the chronological frames.
Step 4. Facial Grammar & Non-Manual Markers: 
  - Look for head shaking side-to-side (Negation).
  - Look for raised eyebrows, head tilted forward, or widened eyes (Question).
  - Look for intense facial expressions, tensed body posture, or sudden sharp movements (Emphasis).

[STRICT OUTPUT CONTRACT]
Output strictly in this format:
<winning_candidate_word_exactly_from_pool> [GRAMMAR_MARKER]

Where [GRAMMAR_MARKER] is:
- [NEG] if negation (head shake, etc.) is detected.
- [Q] if a question cue (eyebrows raised, head forward, etc.) is detected.
- [EMP] if intense emphasis is detected.
- [NONE] if no specific grammatical marker is detected.

Example Output:
ذهب [NEG]

Do NOT include any extra text, markdown, explanations, or backticks.
"""

        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "images": base64_frames,
                "stream": False,
                "options": {
                    "temperature": 0.0,  # Zero temperature for 100% deterministic, zero-hallucination decoding
                    "top_p": 0.1
                }
            },
            timeout=90
        )

        if response.status_code == 200:
            vlm_response = response.json().get("response", "").strip()
            
            # Extract winning candidate
            matched_candidate = None
            for candidate in candidates:
                cand_clean = candidate.strip()
                if cand_clean and cand_clean in vlm_response:
                    matched_candidate = cand_clean
                    break
            
            if matched_candidate:
                grammar_annotation = ""
                if "[NEG]" in vlm_response:
                    grammar_annotation = " (negation)"
                elif "[Q]" in vlm_response:
                    grammar_annotation = " (question)"
                elif "[EMP]" in vlm_response:
                    grammar_annotation = " (emphasis)"
                return matched_candidate, grammar_annotation
    except Exception as e:
        logger.warning("VLM inference warning: %s", e)

    return (candidates[0] if candidates else "", "")


def predict_sign_with_vlm(video_path: str, candidates: list, dialect: str = "Saudi Arabic Sign Language", model: str = VLM_MODEL) -> str:
    """
    Extracts sequential video frames, feeds them to the local VLM (Qwen2-VL),
    and predicts which gesture candidate is performed, returning it with any annotation.
    """
    from app.vlm_util import extract_video_frames

    base64_frames = extract_video_frames(video_path, count=6)
    if not base64_frames:
        return candidates[0] if candidates else ""

    matched_candidate, annotation = query_vlm_with_frames(base64_frames, candidates, dialect, model)
    return f"{matched_candidate}{annotation}"


def debate_and_decide(candidates: list, candidate_confidences: list, pred_vlm: str, history: list, dialect: str = "Saudi Arabic Sign Language", model: str = JUDGE_MODEL) -> str:
    """
    Spawns the local LLM as a Judge agent to evaluate all top candidates and recommendations
    to select the most accurate sign.
    """
    if not candidates:
        return pred_vlm

    pred_lstm = candidates[0]
    lstm_confidence = candidate_confidences[0] if candidate_confidences else 0.0

    # Format the candidate pool for the prompt
    pool_lines = []
    for idx, (cand, conf) in enumerate(zip(candidates, candidate_confidences)):
        pool_lines.append(f"- Candidate {idx+1}: \"{cand}\" (Coordinate confidence: {int(conf * 100)}%)")
    candidate_info = "\n".join(pool_lines)

    prompt = f"""
[SYSTEM ROLE: Chief Sign Language AI Arbitrator ({dialect})]
You are resolving a disagreement between two specialized sign language perception models. You have no independent ground truth. 
To eliminate single-model perspective bias, you will execute a multi-agent persona debate internally in a single pass.

Contextual History (Previous Words in current session): {history}
Regional Dialect: {dialect}

Candidate Pool (ordered by coordinate model confidence):
{candidate_info}

Model Recommendations:
- Coordinate Tracking LSTM recommends: "{pred_lstm}" (Confidence: {int(lstm_confidence * 100)}%)
- Storyboard Vision VLM recommends: "{pred_vlm}" (based on direct visual/facial grammar evidence)

[DEBATE PERSONAS]
1. Persona A (Linguistic Context Specialist): Advocates for the word that makes the most natural semantic and contextual sense following {history}.
2. Persona B (Coordinate LSTM Advocate): Focuses heavily on the Coordinate LSTM's exact confidences, trajectories, speed, and continuous movement.
3. Persona C (VLM Vision Advisor): Advocates for hand shapes, spatial contact zones, and facial/non-manual grammar markers (negation, questions, emphasis).

[ARBITRATION RULES]
- **Abstention Rule:** If BOTH the Coordinate LSTM confidence is low (< 50%) AND the VLM recommendation does not fit the semantic history context, the consensus decision must be to ABSTAIN and return "?".
- **Candidate Pool Constraint:** Any chosen word must match one of the candidates in the pool exactly, or "?".

[OUTPUT FORMAT CONTRACT]
Return ONLY a valid JSON object matching this exact structure:
{{
  "persona_a_argument": "brief analysis of semantic fit",
  "persona_b_argument": "brief analysis of coordinate/LSTM trajectories",
  "persona_c_argument": "brief analysis of facial/hand shapes visual cues",
  "consensus_debate_resolution": "brief summary of how the disagreement is resolved",
  "decision": "chosen candidate word strictly matching one in the pool, OR '?' to abstain",
  "reasoning": "one-line concise final reasoning"
}}
"""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,  # Zero-hallucination deterministic decoding
                    "top_p": 0.1
                }
            },
            timeout=90
        )
        if response.status_code == 200:
            response_text = response.json().get("response", "").strip()
            if response_text.startswith("```"):
                # Strip markdown code fences robustly
                response_text = re.sub(r'^```(?:json|JSON)?\s*', '', response_text)
                response_text = re.sub(r'\s*```\s*$', '', response_text)
            response_text = response_text.strip()
            
            data = json.loads(response_text)
            decided_word = data.get("decision", "").strip()
            
            if decided_word == "?":
                logger.info("Judge Agent Decision: ABSTAIN '?' (Reason: %s)", data.get('reasoning'))
                return "?"
            
            # Ensure the decided word is strictly in the candidates list
            for candidate in candidates:
                if decided_word == candidate.strip():
                    logger.info("Judge Agent Decision: %s (Reason: %s)", candidate, data.get('reasoning'))
                    return candidate
    except Exception as e:
        logger.warning("Judge Agent debate failed: %s", e)

    # Fallback if Ollama fails or Judge outputs an invalid word
    return pred_vlm if lstm_confidence < 0.6 else pred_lstm


# ─── Proactive GPU Model Preloading Background Thread ─────────────────────────
def _preload_gpu_models():
    """Sends empty queries to Ollama to preload VLM and Judge models in the GPU."""
    import time
    time.sleep(2.0)  # Wait for startup to settle down
    for model_name in (VLM_MODEL, JUDGE_MODEL):
        try:
            requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": model_name, "prompt": "", "keep_alive": -1},
                timeout=5
            )
        except Exception:
            pass

# Start the background thread only during actual server runtime (not during migrate, collectstatic, etc.)
import sys
import threading as _threading
_is_management_command = len(sys.argv) > 1 and sys.argv[1] in ('migrate', 'collectstatic', 'makemigrations', 'shell', 'test', 'check')
if not _is_management_command and (os.environ.get('RUN_MAIN') == 'true' or os.environ.get('DJANGO_DEBUG', 'True').lower() == 'false'):
    _threading.Thread(target=_preload_gpu_models, daemon=True).start()

