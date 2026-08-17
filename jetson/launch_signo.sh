#!/bin/bash
# Signo v6 — Kiosk Autostart Script
# Checks Ollama, updates environment, and runs the Pygame Kiosk

echo "=========================================================="
echo "⚡ Starting Signo v6 Translation Kiosk..."
echo "=========================================================="

# 1. Ensure GPU-accelerated Ollama on port 11435 is running
if ! curl -s http://127.0.0.1:11435/api/tags > /dev/null; then
    echo "📦 Starting GPU Ollama on port 11435..."
    /home/jetson/Sign-to-Text-Translation-System-Using-Arabic-Sign-Language/jetson/start_ollama_cpu.sh
else
    echo "✅ GPU Ollama is already running on port 11435."
fi

# 2. Stop any existing kiosk instances
echo "🔄 Clearing old kiosk sessions..."
pkill -9 -f run_native_kiosk.py || true
sleep 1

# 3. Launch the kiosk app
echo "🚀 Relaunching Pygame Kiosk..."
export DISPLAY=:0
python3 -u /home/jetson/Sign-to-Text-Translation-System-Using-Arabic-Sign-Language/scripts/run_native_kiosk.py
