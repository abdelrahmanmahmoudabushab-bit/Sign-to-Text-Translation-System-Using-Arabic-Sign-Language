import json
import logging
import os

from django.http import JsonResponse
from django.middleware.csrf import get_token
from django.shortcuts import render
from django.views.decorators.http import require_POST

from app import util

logger = logging.getLogger(__name__)

# Constants for upload validation
MAX_VIDEO_SIZE = 30 * 1024 * 1024  # 30 MB
ALLOWED_VIDEO_MIMES = {'video/webm', 'video/mp4', 'video/ogg', 'video/x-matroska'}


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

        # Save the uploaded video (browser records WebM, so keep correct extension)
        upload_dir = os.path.join('media', 'uploaded_videos')
        os.makedirs(upload_dir, exist_ok=True)
        video_path = os.path.join(upload_dir, 'video.webm')
        with open(video_path, 'wb+') as destination:
            for chunk in video_file.chunks():
                destination.write(chunk)

        # Extract keypoints and predict
        x = util.DataLoader.DataLoader.load_inference_data(video_path)
        if x is None:
            return JsonResponse({'status': 'failed', 'message': 'Could not process video frames.'})

        # Fetch dialect and history from session/request
        dialect = request.POST.get('dialect', 'Saudi Arabic Sign Language')
        history = request.session.get('translation_history', [])

        logger.info("Inference input shape: %s (Dialect: %s)", x.shape, dialect)
        prediction = util.predict(x, video_path=video_path, history=history, dialect=dialect)
        logger.info("Prediction: %s", prediction)

        # Update history
        history.append(prediction)
        if len(history) > 10:
            history.pop(0)
        request.session['translation_history'] = history

        import time
        return JsonResponse({
            'status': 'success',
            'message': prediction,
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



@require_POST
def smooth_sentence(request):
    """
    POST view accepting a list of raw sign words, returning
    smoothed Arabic and English translations via local Ollama.
    """
    try:
        data = json.loads(request.body)
        words = data.get('words', [])
        dialect = data.get('dialect', 'Saudi Arabic Sign Language')
        if not words:
            return JsonResponse({'status': 'failed', 'message': 'No words provided.'}, status=400)

        from app.llm_util import smooth_sign_sentence
        translation = smooth_sign_sentence(words, dialect=dialect)
        return JsonResponse({
            'status': 'success',
            'arabic': translation['arabic'],
            'english': translation['english']
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


@require_POST
def api_clear_logs(request):
    """Reset telemetry history logs."""
    from app.telemetry import clear_telemetry_logs
    clear_telemetry_logs()
    return JsonResponse({'status': 'success', 'message': 'Telemetry logs cleared.'})

