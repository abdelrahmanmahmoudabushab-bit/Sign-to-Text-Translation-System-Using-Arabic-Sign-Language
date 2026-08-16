#!/bin/bash
# Start a secondary CPU-only Ollama instance on port 11435 to bypass CUDA out-of-memory errors on Jetson Orin Nano
export CUDA_VISIBLE_DEVICES=""
export OLLAMA_HOST="127.0.0.1:11435"
export OLLAMA_MODELS="/usr/share/ollama/.ollama/models"

# Kill any existing CPU Ollama serve instance running on this user
pkill -f "ollama serve.*11435" || true

echo "Starting CPU-only Ollama on port 11435..."
/usr/local/bin/ollama serve > /tmp/ollama_cpu.log 2>&1 &

# Wait for server to become ready
for i in {1..10}; do
    if curl -s http://127.0.0.1:11435/api/tags > /dev/null; then
        echo "Ollama CPU server is ONLINE on port 11435!"
        curl -s http://127.0.0.1:11435/api/tags
        exit 0
    fi
    sleep 1
done

echo "Failed to start Ollama CPU server. Logs:"
cat /tmp/ollama_cpu.log
exit 1
