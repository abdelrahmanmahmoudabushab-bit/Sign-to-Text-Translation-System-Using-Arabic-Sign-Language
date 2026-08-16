#!/bin/bash
set -e

# ============================================================================
#  Signo — Automated Production Deployment Script
#  Target Platform: NVIDIA Jetson Orin Nano (JetPack 6.x / L4T R36)
# ============================================================================

echo "════════════════════════════════════════════════════════════════"
echo "  🤟 Signo — Arabic Sign Language Production Deployment"
echo "════════════════════════════════════════════════════════════════"

# 1. Check prerequisites: Docker & Docker Compose
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed. Please install Docker & NVIDIA Container Toolkit first."
    exit 1
fi

if ! docker compose version &> /dev/null; then
    echo "❌ Error: 'docker compose' plugin is missing."
    exit 1
fi

echo "✅ Docker and Docker Compose detected."

# 2. Check NVIDIA Container Runtime
echo "🔍 Checking NVIDIA GPU runtime availability..."
if docker info 2>/dev/null | grep -i "nvidia" > /dev/null; then
    echo "✅ NVIDIA Container Runtime is active!"
else
    echo "⚠️ Warning: NVIDIA Container Runtime may not be default runtime. Ensure nvidia-container-toolkit is configured."
fi

# 3. Create required persistent data directories
echo "📁 Preparing persistent host storage volumes..."
mkdir -p media/uploaded_videos media/new_signs db staticfiles
chmod -R 777 media db staticfiles

# 4. Generate production DJANGO_SECRET_KEY if missing in environment
if [ ! -f .env ]; then
    echo "📝 Generating production .env configuration file..."
    SECRET_KEY=$(python3 -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())" 2>/dev/null || echo "prod-signo-key-$(date +%s)")
    cat << EOF > .env
DJANGO_SECRET_KEY=$SECRET_KEY
DJANGO_DEBUG=False
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1,0.0.0.0,signo-web
OLLAMA_HOST=http://ollama:11434
VLM_MODEL=qwen2-vl:2b
JUDGE_MODEL=qwen2.5:3b
TRANSLATOR_MODEL=qwen2.5:3b
SIGNO_BYPASS_ARBITRATION=0
EOF
    echo "✅ Created .env file."
fi

# 5. Build and launch production Docker stack
echo "🚀 Building and starting Signo production containers..."
docker compose up -d --build

# 6. Monitor container launch & poll web app health check
echo "⏳ Waiting for Signo web application to report HEALTHY..."
HOST_URL="http://localhost:8000"

for i in $(seq 1 30); do
    STATUS_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$HOST_URL/" || true)
    if [ "$STATUS_CODE" -eq 200 ]; then
        echo ""
        echo "════════════════════════════════════════════════════════════════"
        echo "🎉 Signo Production System Deployed Successfully!"
        echo "════════════════════════════════════════════════════════════════"
        echo "  - Main Kiosk Interface:     http://localhost/"
        echo "  - Web App Direct (Django):  http://localhost:8000/"
        echo "  - Telemetry Dashboard:      http://localhost/dashboard/"
        echo "  - Translation History:      http://localhost/history/"
        echo "  - Ollama Engine:            http://localhost:11434/"
        echo "════════════════════════════════════════════════════════════════"
        exit 0
    fi
    sleep 3
    echo "   ... waiting for Gunicorn server ($i/30)"
done

echo "⚠️ Container started, but health check timed out. Inspect logs with: docker compose logs -f"
