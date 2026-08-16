#!/bin/bash
set -e

# ============================================================================
#  Signo — Jetson Hardware Performance Optimization Script
#  Target Platform: NVIDIA Jetson Orin Nano / Orin AGX (JetPack 5.x / 6.x)
# ============================================================================

echo "════════════════════════════════════════════════════════════════"
echo "  ⚡ Signo — Jetson Hardware Performance & Clock Locking"
echo "════════════════════════════════════════════════════════════════"

if [ "$EUID" -ne 0 ]; then
    echo "⚠️  This script requires root privileges to configure hardware clocks."
    echo "    Re-running with sudo..."
    exec sudo "$0" "$@"
fi

# 1. Set MAXN Power Mode (Maximum performance mode, no power limit throttling)
if command -v nvpmodel &> /dev/null; then
    echo "⚙️ Setting Jetson Power Mode to MAXN (Mode 0)..."
    nvpmodel -m 0 || true
    echo "✅ Power mode set to MAXN."
else
    echo "ℹ️ nvpmodel utility not found (non-Jetson system)."
fi

# 2. Lock CPU and GPU clocks to maximum frequency (eliminates dynamic frequency scaling latency)
if command -v jetson_clocks &> /dev/null; then
    echo "🔥 Locking CPU, GPU, and EMC memory clocks to MAX frequency..."
    jetson_clocks || true
    echo "✅ Jetson clocks locked to maximum."
else
    echo "ℹ️ jetson_clocks utility not found (non-Jetson system)."
fi

# 3. Read & Display Hardware Performance Summary
echo ""
echo "📊 Current Jetson Hardware Status:"
if [ -f /sys/devices/system/cpu/online ]; then
    echo "   - Online CPU Cores:  $(cat /sys/devices/system/cpu/online)"
fi

if [ -f /sys/class/devfreq/17000000.gpu/cur_freq ]; then
    GPU_FREQ_MHZ=$(($(cat /sys/class/devfreq/17000000.gpu/cur_freq) / 1000000))
    echo "   - GPU Frequency:     ${GPU_FREQ_MHZ} MHz"
fi

if [ -d /sys/class/thermal ]; then
    for zone in /sys/class/thermal/thermal_zone*; do
        if [ -f "$zone/type" ] && [ -f "$zone/temp" ]; then
            TYPE=$(cat "$zone/type")
            TEMP=$(($(cat "$zone/temp") / 1000))
            if [[ "$TYPE" == *"cpu"* || "$TYPE" == *"gpu"* || "$TYPE" == *"thermal"* ]]; then
                echo "   - Temp ($TYPE): ${TEMP}°C"
            fi
        fi
    done
fi

echo "════════════════════════════════════════════════════════════════"
echo "⚡ Hardware is optimized for minimum latency inference!"
echo "════════════════════════════════════════════════════════════════"
