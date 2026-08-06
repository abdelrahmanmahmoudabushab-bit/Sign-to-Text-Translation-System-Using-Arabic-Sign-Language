import json
import logging
import os
import shutil
import time
import uuid

from django.conf import settings as django_settings
from django.http import JsonResponse
from django.middleware.csrf import get_token
from django.shortcuts import render
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt

from app import util
from app.DataLoader import DataLoader

logger = logging.getLogger(__name__)

# Constants for upload validation
MAX_VIDEO_SIZE = 30 * 1024 * 1024  # 30 MB
ALLOWED_VIDEO_MIMES = {'video/webm', 'video/mp4', 'video/ogg', 'video/x-matroska', 'application/octet-stream'}


@csrf_exempt
@require_POST
def upload_video(request):
    try:
        video_file = request.FILES.get('video')
        if video_file is None:
            return JsonResponse({'status': 'failed', 'message': 'No video file received.'}, status=400)

        # Validate file size
        if video_file.size > MAX_VIDEO_SIZE:
            return JsonResponse({
                'status': 'failed',
                'message': f'Video too large. Maximum size is {MAX_VIDEO_SIZE // (1024 * 1024)} MB.'
            }, status=400)

        # Validate MIME type
        content_type = getattr(video_file, 'content_type', '')
        if content_type and content_type not in ALLOWED_VIDEO_MIMES:
            logger.warning("Rejected upload with MIME type: %s", content_type)
            return JsonResponse({
                'status': 'failed',
                'message': 'Invalid file type. Please upload a video file.'
            }, status=400)

        # Save the uploaded video with unique filename and correct extension to prevent race conditions
        original_name = getattr(video_file, 'name', 'video.webm')
        ext = os.path.splitext(original_name)[1] or '.webm'
        upload_dir = os.path.join('media', 'uploaded_videos')
        os.makedirs(upload_dir, exist_ok=True)
        video_filename = f'video_{uuid.uuid4().hex[:8]}{ext}'
        video_path = os.path.join(upload_dir, video_filename)
        with open(video_path, 'wb+') as destination:
            for chunk in video_file.chunks():
                destination.write(chunk)

        # Fetch dialect and history from session/request
        dialect = request.POST.get('dialect', 'Saudi Arabic Sign Language')
        history = request.session.get('translation_history', [])

        # Predict with parallelized MediaPipe extraction and VLM prep
        prediction = util.predict(x=None, video_path=video_path, history=history, dialect=dialect)
        logger.info("Prediction: %s", prediction)

        # Update history
        history.append(prediction)
        if len(history) > 10:
            history.pop(0)
        request.session['translation_history'] = history

        # ── Save clip to database ────────────────────────────────────────────
        clip_id = None
        try:
            from app.models import TranslationClip
            from django.core.files import File

            # Move clip into MEDIA_ROOT/sessions/ via Django's FileField
            with open(video_path, 'rb') as f:
                clip = TranslationClip(
                    gesture=prediction or '',
                    dialect=dialect,
                    confidence=0.0,   # placeholder; update if model returns score
                )
                clip.video.save(os.path.basename(video_path), File(f), save=True)

            clip_id = clip.pk

            # Track clip IDs in session for sentence linking
            pending = request.session.get('pending_clip_ids', [])
            pending.append(clip.pk)
            if len(pending) > 50:            # guard against runaway sessions
                pending = pending[-50:]
            request.session['pending_clip_ids'] = pending
            logger.info("Saved TranslationClip pk=%s gesture='%s'", clip.pk, prediction)
        except Exception as db_err:
            logger.warning("Could not save clip to DB: %s", db_err)

        # On-Device Self-Learning Data Collector: Archive clips for batch retraining
        if not prediction or prediction == "?":
            try:
                new_signs_dir = os.path.join(django_settings.MEDIA_ROOT, 'new_signs')
                os.makedirs(new_signs_dir, exist_ok=True)
                sample_id = f"sample_{int(time.time()*1000)}"
                archived_clip = os.path.join(new_signs_dir, f"{sample_id}.webm")
                shutil.copyfile(video_path, archived_clip)

                meta_path = os.path.join(new_signs_dir, f"{sample_id}.json")
                with open(meta_path, 'w', encoding='utf-8') as mf:
                    json.dump({
                        'sample_id': sample_id,
                        'dialect': dialect,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'keypoints_shape': [60, 225]
                    }, mf, ensure_ascii=False, indent=2)
                logger.info("Saved un-matched clip to %s for self-learning batch retraining", archived_clip)
            except Exception as e:
                logger.warning("Failed to save self-learning clip: %s", e)

        # Clean up the temp upload file (the DB copy is already saved separately)
        try:
            os.remove(video_path)
        except OSError:
            pass

        return JsonResponse({
            'status': 'success',
            'message': prediction,
            'clip_id': clip_id,
            'dialect': dialect,
            'history_count': len(history),
            'timestamp': time.strftime('%H:%M:%S')
        })

    except Exception:
        logger.exception("Error during video prediction")
        return JsonResponse(
            {'status': 'failed', 'message': 'An internal error occurred. Please try again.'},
            status=500
        )


def index(request):
    # Pass CSRF token to the template for JavaScript fetch calls
    get_token(request)
    # Reset translation history on main page load/refresh
    request.session['translation_history'] = []
    return render(request, 'app/index.html')



@csrf_exempt
@require_POST
def smooth_sentence(request):
    """
    POST view accepting a list of raw sign words, returning
    smoothed Arabic and English translations via local Ollama.
    Also saves a TranslationSession linking the clips collected this session.
    """
    try:
        data = json.loads(request.body)
        words = data.get('words', [])
        dialect = data.get('dialect', 'Saudi Arabic Sign Language')
        if not words:
            return JsonResponse({'status': 'failed', 'message': 'No words provided.'}, status=400)

        from app.llm_util import smooth_sign_sentence
        translation = smooth_sign_sentence(words, dialect=dialect)

        # ── Save session to database ─────────────────────────────────────────
        session_id = None
        try:
            from app.models import TranslationClip, TranslationSession
            session = TranslationSession.objects.create(
                arabic_sentence=translation.get('arabic', ''),
                english_sentence=translation.get('english', ''),
                dialect=dialect,
            )
            # Link the clips collected during this browser session
            pending_ids = request.session.get('pending_clip_ids', [])
            if pending_ids:
                clips = TranslationClip.objects.filter(pk__in=pending_ids)
                session.clips.set(clips)
            # Clear pending list for the next sentence
            request.session['pending_clip_ids'] = []
            session_id = session.pk
            logger.info("Saved TranslationSession pk=%s with %d clips", session.pk, session.clips.count())
        except Exception as db_err:
            logger.warning("Could not save session to DB: %s", db_err)

        return JsonResponse({
            'status': 'success',
            'arabic': translation['arabic'],
            'english': translation['english'],
            'session_id': session_id,
        })
    except json.JSONDecodeError:
        return JsonResponse({'status': 'failed', 'message': 'Invalid JSON payload.'}, status=400)
    except Exception:
        logger.exception("Error during sentence smoothing")
        return JsonResponse(
            {'status': 'failed', 'message': 'An internal error occurred. Please try again.'},
            status=500
        )


def dashboard(request):
    """Render the live telemetry dashboard UI."""
    return render(request, 'app/dashboard.html')


def api_telemetry(request):
    """JSON API endpoint returning live system telemetry snapshot."""
    from app.telemetry import get_telemetry_snapshot
    return JsonResponse(get_telemetry_snapshot())


@csrf_exempt
@require_POST
def api_clear_logs(request):
    """Reset telemetry history logs."""
    from app.telemetry import clear_telemetry_logs
    clear_telemetry_logs()
    return JsonResponse({'status': 'success', 'message': 'Telemetry logs cleared.'})


def api_new_signs(request):
    """JSON API endpoint returning stats on self-learning clips collected."""
    from django.conf import settings
    new_signs_dir = os.path.join(settings.MEDIA_ROOT, 'new_signs')
    count = 0
    samples = []
    if os.path.exists(new_signs_dir):
        files = os.listdir(new_signs_dir)
        json_files = [f for f in files if f.endswith('.json')]
        count = len(json_files)
        for jf in sorted(json_files, reverse=True)[:10]:
            try:
                with open(os.path.join(new_signs_dir, jf), 'r', encoding='utf-8') as f:
                    samples.append(json.load(f))
            except Exception:
                pass
    return JsonResponse({'status': 'success', 'count': count, 'recent_samples': samples})


def api_developer_logs(request):
    """JSON API endpoint returning the latest Gunicorn/systemd logs for signo.service."""
    import subprocess
    try:
        res = subprocess.run(
            ["journalctl", "-u", "signo.service", "-n", "45", "--no-pager"],
            capture_output=True,
            text=True,
            timeout=2.0
        )
        if res.returncode == 0:
            lines = res.stdout.strip().split("\n")
            return JsonResponse({'status': 'success', 'logs': lines})
    except Exception as e:
        logger.warning("Failed to fetch journal logs: %s", e)
    
    return JsonResponse({'status': 'success', 'logs': ["Systemd journal logs not available on this host platform."]})


@csrf_exempt
@require_POST
def api_control_jetson(request):
    """Execute hardware and model management actions on the Jetson Nano."""
    import subprocess
    try:
        data = json.loads(request.body)
        action = data.get("action")
        
        if action == "restart_service":
            import os
            res = subprocess.run(["pgrep", "-o", "-f", "gunicorn"], capture_output=True, text=True)
            if res.returncode == 0:
                pid = res.stdout.strip()
                if pid:
                    os.kill(int(pid), 1)  # 1 is SIGHUP (HUP)
                    return JsonResponse({'status': 'success', 'message': f'Gunicorn master PID {pid} reloaded.'})
            return JsonResponse({'status': 'failed', 'message': 'Gunicorn master PID not found.'})
            
        elif action == "preload_vlm":
            from app.llm_util import OLLAMA_URL, VLM_MODEL
            import requests
            res = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": VLM_MODEL,
                    "prompt": "",
                    "keep_alive": -1
                },
                timeout=15
            )
            if res.status_code == 200:
                return JsonResponse({'status': 'success', 'message': f'Model {VLM_MODEL} loaded into GPU.'})
            return JsonResponse({'status': 'failed', 'message': f'Ollama returned status {res.status_code}'})
            
        elif action == "preload_judge":
            from app.llm_util import OLLAMA_URL, JUDGE_MODEL
            import requests
            res = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": JUDGE_MODEL,
                    "prompt": "",
                    "keep_alive": -1
                },
                timeout=15
            )
            if res.status_code == 200:
                return JsonResponse({'status': 'success', 'message': f'Model {JUDGE_MODEL} loaded into GPU.'})
            return JsonResponse({'status': 'failed', 'message': f'Ollama returned status {res.status_code}'})
            
        elif action == "keep_alive":
            # Set infinite keep-alive timeout for both models
            from app.llm_util import OLLAMA_URL, VLM_MODEL, JUDGE_MODEL
            import requests
            for m in [VLM_MODEL, JUDGE_MODEL]:
                try:
                    requests.post(
                        f"{OLLAMA_URL}/api/generate",
                        json={"model": m, "prompt": "", "keep_alive": -1},
                        timeout=5
                    )
                except Exception:
                    pass
            return JsonResponse({'status': 'success', 'message': 'Ollama infinite keep-alive set for active models.'})
            
        else:
            return JsonResponse({'status': 'failed', 'message': f'Action {action} not recognized.'}, status=400)
            
    except Exception as e:
        logger.exception("Error executing Jetson control action")
        return JsonResponse({'status': 'failed', 'message': str(e)}, status=500)


def api_ollama_status(request):
    """
    JSON API endpoint — polls the local Ollama server for loaded model status.
    Returns: loaded models, VRAM usage, and whether each AI role is ready.
    Used by the kiosk 'AI Status' widget.
    """
    import requests as req
    from app.llm_util import OLLAMA_URL, VLM_MODEL, JUDGE_MODEL, TRANSLATOR_MODEL

    try:
        resp = req.get(f"{OLLAMA_URL}/api/ps", timeout=3)
        if resp.status_code != 200:
            return JsonResponse({'status': 'offline', 'models': [], 'roles': {}})

        data = resp.json()
        loaded = data.get('models', [])
        loaded_names = [m.get('name', '') for m in loaded]

        roles = {
            'vlm':        {'model': VLM_MODEL,        'ready': VLM_MODEL in loaded_names},
            'judge':      {'model': JUDGE_MODEL,       'ready': JUDGE_MODEL in loaded_names},
            'translator': {'model': TRANSLATOR_MODEL,  'ready': TRANSLATOR_MODEL in loaded_names},
        }

        return JsonResponse({
            'status': 'online',
            'models': loaded,
            'roles': roles,
        })

    except Exception as e:
        logger.warning("Ollama status check failed: %s", e)
        return JsonResponse({'status': 'offline', 'models': [], 'roles': {}})


def history(request):
    """Render the translation history browser page."""
    return render(request, 'app/history.html')


def api_sessions(request):
    """
    GET  /api/sessions/       — list recent TranslationSessions (JSON)
    DELETE /api/sessions/<id>/ — delete a session and its clips
    """
    from app.models import TranslationSession

    if request.method == 'DELETE':
        # Extract session ID from URL path
        path_parts = request.path.rstrip('/').split('/')
        try:
            session_id = int(path_parts[-1])
        except (ValueError, IndexError):
            return JsonResponse({'status': 'failed', 'message': 'Invalid session ID.'}, status=400)
        try:
            sess = TranslationSession.objects.get(pk=session_id)
            # Also delete video files via clip.delete()
            for clip in sess.clips.all():
                clip.delete()
            sess.delete()
            return JsonResponse({'status': 'success', 'deleted': session_id})
        except TranslationSession.DoesNotExist:
            return JsonResponse({'status': 'failed', 'message': 'Session not found.'}, status=404)

    # GET — return paginated list
    page = int(request.GET.get('page', 1))
    per_page = int(request.GET.get('per_page', 20))
    offset = (page - 1) * per_page

    sessions = TranslationSession.objects.prefetch_related('clips').order_by('-created_at')[offset:offset + per_page]
    total = TranslationSession.objects.count()

    results = []
    for sess in sessions:
        clips_data = []
        for clip in sess.clips.all():
            clips_data.append({
                'id': clip.pk,
                'gesture': clip.gesture,
                'dialect': clip.dialect,
                'confidence': round(clip.confidence * 100, 1),
                'video_url': clip.video.url if clip.video else None,
                'created_at': clip.created_at.strftime('%Y-%m-%d %H:%M'),
            })
        results.append({
            'id': sess.pk,
            'arabic': sess.arabic_sentence,
            'english': sess.english_sentence,
            'dialect': sess.dialect,
            'clip_count': len(clips_data),
            'clips': clips_data,
            'created_at': sess.created_at.strftime('%Y-%m-%d %H:%M'),
        })

    return JsonResponse({
        'status': 'success',
        'total': total,
        'page': page,
        'per_page': per_page,
        'sessions': results,
    })
