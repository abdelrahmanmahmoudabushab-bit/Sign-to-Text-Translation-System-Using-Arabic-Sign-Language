import requests
import json
import logging
import os
import re
import hashlib
import tempfile
import threading

# Detect if running on NVIDIA Jetson target
is_jetson = os.path.exists("/etc/nv_tegra_release")
default_host = "http://localhost:11435" if is_jetson else "http://localhost:11434"
OLLAMA_URL = os.environ.get("OLLAMA_HOST", default_host)

VLM_MODEL = os.environ.get("VLM_MODEL", "qwen2.5vl:3b")            # Vision Classifier
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "llama3.2:3b")         # Arbitration Judge
TRANSLATOR_MODEL = os.environ.get("TRANSLATOR_MODEL", "llama3.2:3b") # Grammar Reconstructor

logger = logging.getLogger(__name__)

# ─── 20x SWE Cache & Async Management ─────────────────────────────────────────
import queue

class BackgroundCacheSaver:
    """Thread-safe, non-blocking, coalesced disk cache writer daemon."""
    def __init__(self, filename_resolver, cache_dict_ref, lock):
        self.filename_resolver = filename_resolver
        self.cache_dict_ref = cache_dict_ref
        self.lock = lock
        self.queue = queue.Queue(maxsize=1)
        self.worker_thread = None
        self.active = False

    def start(self):
        if self.worker_thread and self.worker_thread.is_alive():
            return
        self.active = True
        self.worker_thread = threading.Thread(target=self._run, daemon=True)
        self.worker_thread.start()

    def trigger_save(self):
        self.start()
        try:
            self.queue.put_nowait(True)
        except queue.Full:
            pass

    def _run(self):
        while self.active:
            try:
                self.queue.get(timeout=2.0)
            except queue.Empty:
                continue
            
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break

            try:
                cache_file = self.filename_resolver()
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with self.lock:
                    snapshot = dict(self.cache_dict_ref)
                
                fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(cache_file), suffix='.tmp')
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        json.dump(snapshot, f, ensure_ascii=False, indent=2)
                    os.replace(tmp_path, cache_file)
                except Exception:
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                    raise
            except Exception as e:
                logger.warning("Failed to save cache in background: %s", e)


_TRANSLATION_CACHE = {}
_CACHE_LOCK = threading.Lock()
_TRANSLATION_CACHE_SAVER = None
_cache_loaded = False

def _init_cache():
    global _TRANSLATION_CACHE, _cache_loaded, _TRANSLATION_CACHE_SAVER
    if _cache_loaded:
        return
    with _CACHE_LOCK:
        if _cache_loaded:
            return
        cache_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "media", "translation_cache.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    _TRANSLATION_CACHE.update(json.load(f))
                logger.info("⚡ Loaded %d translations from persistent cache", len(_TRANSLATION_CACHE))
            except Exception as e:
                logger.warning("Failed to load translation cache: %s", e)
        
        _TRANSLATION_CACHE_SAVER = BackgroundCacheSaver(
            filename_resolver=lambda: os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "media", "translation_cache.json"),
            cache_dict_ref=_TRANSLATION_CACHE,
            lock=_CACHE_LOCK
        )
        _TRANSLATION_CACHE_SAVER.start()
        _cache_loaded = True

def _save_cache():
    if _TRANSLATION_CACHE_SAVER:
        _TRANSLATION_CACHE_SAVER.trigger_save()

def rule_based_fallback(words_list, dialect):
    """Zero-LLM fallback: maps top candidates to a basic grammatical structure."""
    if isinstance(words_list[0], list):
        words = [opts[0] for opts in words_list]
    else:
        words = words_list
    # Remove annotations like (negation) for readable output
    clean_words = [re.sub(r'\s*\([^)]*\)', '', w) for w in words]
    arabic = " ".join(clean_words)
    english = f"[Offline Fallback] {' '.join(words)}"
    return {
        "arabic": arabic,
        "english": english
    }

def smooth_sign_sentence(words_list, dialect="Saudi Arabic Sign Language", model=TRANSLATOR_MODEL, conversation_history=None, average_confidence=1.0):
    """
    Takes a list of raw sign language words/candidates, queries the local Ollama LLM, 
    and returns a grammatically correct sentence in the selected dialect and its English translation.
    Features: Semantic Caching, Stateful Conversation History, Dynamic Temp, and Rule-Based Fallback.
    """
    if not words_list:
        return {"arabic": "", "english": ""}
        
    _init_cache()
    
    # 1. Check Cache
    # Build unique hash key from candidate sequence, dialect, and conversation history
    history_serialized = json.dumps(conversation_history or [], sort_keys=True)
    words_serialized = json.dumps(words_list, sort_keys=True)
    cache_key = hashlib.sha256(f"{words_serialized}_{dialect}_{history_serialized}".encode("utf-8")).hexdigest()
    
    with _CACHE_LOCK:
        cached_res = _TRANSLATION_CACHE.get(cache_key)
    if cached_res:
        logger.info("⚡ Translation Cache HIT: %s", cached_res["arabic"])
        return cached_res

    # 2. Dynamic Temperature Control
    # Use strict temperature if confidence is high, allow more linguistic flexibility if low
    temp = 0.0 if average_confidence >= 0.8 else 0.3

    # Dialect-specific tips for Levantine/Jordanian Arabic
    dialect_guidelines = ""
    if "Levantine" in dialect:
        dialect_guidelines = "\n5. Target Dialect Specifics: For Levantine/Jordanian Arabic, favor colloquial regional verb conjugation and terms (e.g. use 'بدي' instead of 'أريد', 'هون' instead of 'هنا', 'شلونك' or 'كيفك' instead of 'كيف حالك', 'هلق' or 'هسا' instead of 'الآن') to make the output sound natural and friendly to local speakers."

    # 3. Format Prompt
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
"""
    else:
        keywords_str = " -> ".join(words_list)
        prompt = f"""
[SYSTEM ROLE: Master Arabic Sign Language Translator & Computational Linguist]
Your task is to convert a sequence of disjointed Arabic Sign Language (ArSL) sign glosses into a natural, grammatically correct spoken sentence in the targeted dialect: {dialect}.

Input Sign Sequence: [{keywords_str}]
Target Dialect: {dialect}
"""

    # Stateful Conversation History Injection
    if conversation_history:
        history_str = "\n".join([
            f"Prior Arabic sentence: {item.get('arabic')}"
            for item in conversation_history[-3:] # Pass last 3 sentences for local context window
        ])
        prompt += f"""
[CONVERSATION CONTEXT]
The signer has already established the following sentences in the current session:
{history_str}
Use this context to resolve pronouns (he, she, it) and preserve correct timeline/tenses.
"""

    prompt += f"""
[TRANSLATION GUIDELINES]
1. Choose the combination of words that forms the most contextually and grammatically correct spoken sentence in {dialect}.
2. Do not choose options that produce nonsense or illogical phrases (e.g. choice combinations that make no semantic sense).
3. Reconstruct the selected keywords into a fluent, natural, and polite spoken sentence matching the grammatical syntax of {dialect}.
4. Provide an accurate, idiomatic English translation of the reconstructed sentence.{dialect_guidelines}

[OUTPUT SPECIFICATION]
Return ONLY a valid JSON object matching this schema (no markdown, no backticks, no extra commentary):
{{
  "arabic": "Reconstructed fluent sentence in {dialect}",
  "english": "Fluent English translation"
}}
"""

    # 4. Execute LLM Query
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temp
                }
            },
            timeout=10 # Short timeout for responsive UI fallback
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get("response", "").strip()
            
            if response_text.startswith("```"):
                response_text = re.sub(r'^```(?:json|JSON)?\s*', '', response_text)
                response_text = re.sub(r'\s*```\s*$', '', response_text)
            response_text = response_text.strip()
            
            data = json.loads(response_text)
            smoothed_res = {
                "arabic": data.get("arabic", "").strip(),
                "english": data.get("english", "").strip()
            }
            
            # Save to persistent cache
            with _CACHE_LOCK:
                _TRANSLATION_CACHE[cache_key] = smoothed_res
            _save_cache()
            return smoothed_res
            
    except Exception as e:
        logger.warning("Ollama integration failed. Falling back to rule-based generation: %s", e)
        
    # 5. Rule-Based Fallback
    return rule_based_fallback(words_list, dialect)


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

