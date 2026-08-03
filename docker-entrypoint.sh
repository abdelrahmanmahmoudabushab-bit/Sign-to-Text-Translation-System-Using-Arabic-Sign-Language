#!/bin/bash
set -e

# ============================================================================
#  Signo — Docker Entrypoint
#  Maximizes Jetson Orin Nano performance before starting the web server
# ============================================================================

echo "═══════════════════════════════════════════════════════"
echo "  🤟 Signo — Arabic Sign Language Translation System"
echo "═══════════════════════════════════════════════════════"

# ─── Generate Django secret key if not provided ──────────────────────────────
if [ -z "$DJANGO_SECRET_KEY" ]; then
    export DJANGO_SECRET_KEY=$(python3 -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())")
    echo "⚠️  Auto-generated DJANGO_SECRET_KEY (set one in docker-compose.yml for persistence)"
fi

# ─── Run Django migrations ───────────────────────────────────────────────────
echo "🔄 Running database migrations..."
python3 manage.py migrate --noinput 2>/dev/null || true

# ─── Wait for Ollama to be ready ────────────────────────────────────────────
echo "⏳ Waiting for Ollama service..."
for i in $(seq 1 30); do
    if curl -s http://ollama:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "⚠️  Ollama not reachable — VLM/LLM features will use fallback mode"
    fi
    sleep 2
done

# ─── Pre-pull standalone Ollama models if not already cached ─────────────────
VLM_MODEL=${VLM_MODEL:-qwen2-vl:2b}
JUDGE_MODEL=${JUDGE_MODEL:-qwen2.5:3b}
TRANSLATOR_MODEL=${TRANSLATOR_MODEL:-qwen2.5:3b}

echo "📦 Ensuring standalone Ollama models are available..."
echo "   - VLM Model:        $VLM_MODEL"
echo "   - Judge Model:      $JUDGE_MODEL"
echo "   - Translator Model: $TRANSLATOR_MODEL"

curl -s -X POST http://ollama:11434/api/pull -d "{\"name\":\"$VLM_MODEL\"}" > /dev/null 2>&1 &
curl -s -X POST http://ollama:11434/api/pull -d "{\"name\":\"$JUDGE_MODEL\"}" > /dev/null 2>&1 &
curl -s -X POST http://ollama:11434/api/pull -d "{\"name\":\"$TRANSLATOR_MODEL\"}" > /dev/null 2>&1 &

# ─── Print GPU info ─────────────────────────────────────────────────────────
echo ""
echo "🖥️  GPU Status:"
python3 -c "
import os
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f'   TensorFlow GPU: {len(gpus)} device(s) detected')
        print(f'   TF Version: {tf.__version__}')
    else:
        print('   ⚠️ No GPU detected — running on CPU')
except Exception as e:
    print(f'   GPU check error: {e}')
" 2>/dev/null

echo ""
echo "🚀 Starting Gunicorn (workers=3, threads=2)..."
echo "═══════════════════════════════════════════════════════"

# ─── Launch Gunicorn ─────────────────────────────────────────────────────────
# 3 workers: optimal for 8GB RAM Jetson (1 master + 3 workers)
# 2 threads per worker: allows concurrent VLM API calls while processing
# 120s timeout: VLM inference can take 10-15s, so we give headroom
exec gunicorn pbl.wsgi:application \
    --bind 0.0.0.0:8000 \
    --workers 3 \
    --threads 2 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    --log-level info
