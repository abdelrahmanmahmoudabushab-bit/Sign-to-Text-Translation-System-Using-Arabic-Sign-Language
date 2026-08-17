#!/bin/bash
# Start a secondary Ollama instance on port 11435 with GPU enabled but limited to 1 parallel context to minimize VRAM usage
export OLLAMA_HOST="127.0.0.1:11435"
export OLLAMA_MODELS="/usr/share/ollama/.ollama/models"
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1

# Kill any existing CPU/GPU Ollama serve instances running on port 11435
pkill -f "ollama serve.*11435" || true
sleep 1

echo "Starting GPU Ollama (Parallel=1) on port 11435..."
/usr/local/bin/ollama serve > /tmp/ollama_gpu_p1.log 2>&1 &

# Wait for server to become ready
for i in {1..10}; do
    if curl -s http://127.0.0.1:11435/api/tags > /dev/null; then
        echo "Ollama GPU (Parallel=1) server is ONLINE on port 11435!"
        exit 0
    fi
    sleep 1
done

echo "Failed to start. Logs:"
cat /tmp/ollama_gpu_p1.log
exit 1
