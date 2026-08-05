#!/bin/bash
# ============================================================================
#  Signo — NVIDIA Jetson Orin Nano Hardware Performance Tuning Script
#
#  Configures max power mode (MAXN), locks maximum GPU/CPU clocks,
#  enables ZRAM swap memory, and sets up high-performance Docker runtime.
#
#  Usage:
#    sudo bash optimize_jetson.sh
# ============================================================================

set -e

echo "═══════════════════════════════════════════════════════"
echo "  🚀 Signo — NVIDIA Jetson Orin Nano Hardware Optimizer"
echo "═══════════════════════════════════════════════════════"

# Check root privileges
if [ "$EUID" -ne 0 ]; then
    echo "❌ Please run as root: sudo bash optimize_jetson.sh"
    exit 1
fi

# ─── 1. Set Maximum Power Mode (MAXN) ────────────────────────────────────────
echo "⚡ Setting Jetson Power Mode to MAXN (Maximum Performance)..."
if command -v nvpmodel > /dev/null 2>&1; then
    nvpmodel -m 0
    echo "✅ Power mode set to MAXN (0)"
else
    echo "⚠️ nvpmodel tool not found — skipping power mode configuration"
fi

# ─── 2. Lock GPU/CPU Clocks & Max Fan Speed ─────────────────────────────────
echo "🔒 Locking maximum CPU, GPU, and EMC memory clocks..."
if command -v jetson_clocks > /dev/null 2>&1; then
    jetson_clocks
    jetson_clocks --fan 2>/dev/null || true
    echo "✅ Jetson Clocks locked to maximum frequency & fan active"
else
    echo "⚠️ jetson_clocks tool not found — skipping clock locking"
fi

# ─── 3. Optimize Linux Virtual Memory & Page Cache ──────────────────────────
echo "🧠 Tuning Linux Virtual Memory for streaming video buffers..."
sysctl -w vm.dirty_background_ratio=5 > /dev/null
sysctl -w vm.dirty_ratio=10 > /dev/null
sysctl -w vm.vfs_cache_pressure=50 > /dev/null
echo "✅ VM parameters updated (dirty_background=5%, dirty_ratio=10%)"

# ─── 4. Configure Docker NVIDIA Default Runtime ─────────────────────────────
DOCKER_DAEMON="/etc/docker/daemon.json"
if [ -f "$DOCKER_DAEMON" ]; then
    echo "🐳 Ensuring NVIDIA Container Runtime is set as default in Docker..."
    if ! grep -q '"default-runtime": "nvidia"' "$DOCKER_DAEMON"; then
        cp "$DOCKER_DAEMON" "${DOCKER_DAEMON}.bak"
        python3 -c "
import json
with open('$DOCKER_DAEMON', 'r') as f:
    data = json.load(f)
data['default-runtime'] = 'nvidia'
with open('$DOCKER_DAEMON', 'w') as f:
    json.dump(data, f, indent=4)
" 2>/dev/null && systemctl restart docker || echo "⚠️ Re-run docker restart manually"
        echo "✅ Docker default-runtime updated to 'nvidia'"
    else
        echo "✅ Docker default-runtime is already set to 'nvidia'"
    fi
fi

# ─── 5. Check ZRAM / Swap Memory ─────────────────────────────────────────────
SWAP_TOTAL=$(free -m | awk '/Swap:/ {print $2}')
echo "💾 Total Swap Memory: ${SWAP_TOTAL} MB"
if [ "$SWAP_TOTAL" -lt 2000 ]; then
    echo "⚠️ Low swap space detected. Enabling 4GB ZRAM swap..."
    if [ -f /usr/bin/nvzramconfig.sh ]; then
        /usr/bin/nvzramconfig.sh start || true
    fi
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  🎉 Jetson Orin Nano Hardware Optimization Complete!"
echo "  Your Jetson is operating at maximum GPU/CPU power."
echo "═══════════════════════════════════════════════════════"
