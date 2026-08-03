# ============================================================================
#  Signo — Arabic Sign Language Translation System
#  Production Dockerfile for NVIDIA Jetson Orin Nano (JetPack 6.x / L4T R36)
#
#  Base: NVIDIA L4T TensorFlow container (includes CUDA, cuDNN, TensorRT)
#  This gives us GPU-accelerated TensorFlow out of the box on ARM64.
# ============================================================================

# ---------- Stage 1: Runtime Image ----------
FROM nvcr.io/nvidia/l4t-tensorflow:r36.4.0-tf2.18-py3 AS runtime

# Metadata
LABEL maintainer="Abdelrahman <signo-project>"
LABEL description="Signo ArSL Translation — Django + TensorFlow + MediaPipe on Jetson"

# ─── Environment: Maximum GPU Performance ────────────────────────────────────
# Force TensorFlow to use GPU and allow memory growth (no OOM crashes)
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
# Minimize TensorFlow logging noise
ENV TF_CPP_MIN_LOG_LEVEL=2
# Ensure CUDA libraries are found
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH}
ENV CUDA_HOME=/usr/local/cuda
# Python optimizations
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Django production settings
ENV DJANGO_DEBUG=False
ENV DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1,0.0.0.0,signo-web

WORKDIR /app

# ─── System Dependencies ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV system dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    # Video codec support
    ffmpeg \
    # Health check utility
    curl \
    && rm -rf /var/lib/apt/lists/*

# ─── Python Dependencies ────────────────────────────────────────────────────
# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install production Python packages
# Note: TensorFlow is already in the base image, so we skip it
# and install only the remaining dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
    django==5.2.16 \
    gunicorn==23.0.0 \
    mediapipe==1.0.0 \
    scikit-learn==1.9.0 \
    opencv-python-headless==4.11.0.86 \
    numpy==1.26.4 \
    Pillow==12.3.0 \
    requests==2.34.2 \
    psutil>=5.9.0

# ─── Application Code ───────────────────────────────────────────────────────
COPY . .

# Create media directories for uploads and storyboard grids
RUN mkdir -p media/uploaded_videos

# Collect Django static files (CSS, JS, images)
RUN python3 manage.py collectstatic --noinput 2>/dev/null || true

# ─── Entrypoint ─────────────────────────────────────────────────────────────
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

ENTRYPOINT ["/docker-entrypoint.sh"]
