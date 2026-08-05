# 🚀 Signo — NVIDIA Jetson Orin Nano Deployment Guide

This guide covers deploying Signo on NVIDIA Jetson Orin Nano edge hardware for production kiosk operation.

---

## 📋 Prerequisites

| Requirement | Version |
|-------------|---------|
| **Hardware** | NVIDIA Jetson Orin Nano (8GB recommended) |
| **JetPack** | 6.x (L4T R36) |
| **Docker** | Installed with `nvidia-container-toolkit` |
| **Storage** | 32GB+ microSD / NVMe (models + Docker images) |
| **Network** | Required for initial model download only |

---

## 1. Hardware Optimization

Run the hardware tuning script to maximize GPU/CPU performance:

```bash
sudo bash jetson/optimize_jetson.sh
```

This script performs:
- **MAXN Power Mode** (`nvpmodel -m 0`) — unlock all CPU/GPU cores at max frequency
- **Clock Locking** (`jetson_clocks`) — prevent dynamic frequency scaling
- **Fan Control** — force active cooling to prevent thermal throttling
- **VM Tuning** — optimize dirty page ratios for video buffer streaming
- **Docker NVIDIA Runtime** — set `nvidia` as the default Docker runtime
- **ZRAM Swap** — enable 4GB compressed swap if physical swap is low

---

## 2. Docker Deployment

### Full Stack Launch

```bash
cd /home/jetson/Sign-to-Text-Translation-System-Using-Arabic-Sign-Language
docker compose up -d
```

This starts three containers:

| Container | Service | Port |
|-----------|---------|------|
| `signo-web` | Django + TensorFlow + MediaPipe | 8000 |
| `signo-ollama` | Ollama LLM/VLM engine | 11434 |
| `signo-nginx` | Nginx reverse proxy | 80 |

### First Boot: TensorRT Engine Compilation

On the first run, compile the optimized TensorRT FP16 engine for your specific Jetson GPU:

```bash
docker compose exec signo-web python3 scripts/export_tensorrt.py
```

> **Note:** This takes 5-10 minutes on Orin Nano. The resulting `.engine` file is hardware-specific and cached for subsequent boots.

### Rebuild After Code Changes

```bash
docker compose build --no-cache signo-web
docker compose up -d
```

### View Logs

```bash
docker compose logs -f          # All containers
docker compose logs -f signo-web  # Web app only
```

---

## 3. Systemd Service (Non-Docker)

For direct deployment without Docker, use the systemd service unit:

### Install

```bash
sudo cp jetson/signo.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable signo.service
sudo systemctl start signo.service
```

### Service Configuration

The service file (`jetson/signo.service`) runs Gunicorn with:
- **3 workers** (optimal for 8GB RAM)
- **2 threads per worker** (concurrent VLM API calls)
- **120s timeout** (headroom for VLM inference)

### Customize

Edit the service file to update:
- `WorkingDirectory` — path to the project
- `DJANGO_ALLOWED_HOSTS` — add your Jetson's IP address
- `ExecStart` — adjust worker count based on available RAM

### Check Status

```bash
sudo systemctl status signo.service
journalctl -u signo.service -f    # Live logs
```

---

## 4. Kiosk Auto-Launch

For touchscreen kiosk deployments, the desktop entry auto-launches Chromium in app mode:

### Install

```bash
cp jetson/signo-app.desktop ~/.config/autostart/
```

### Behavior

On GNOME/Unity login:
1. Chromium opens `http://localhost:8000/` in fullscreen app mode
2. No address bar, no browser UI — pure kiosk experience
3. The web app's Wake Lock API prevents screen dimming

### Manual Launch

```bash
chromium-browser --app=http://localhost:8000/ --start-maximized --no-first-run
```

---

## 5. Ollama Model Management

### Pre-pull Models

```bash
ollama pull qwen2-vl:2b     # Vision Language Model (~1.5GB)
ollama pull qwen2.5:3b       # Judge + Translator (~2GB)
```

### Keep Models in GPU Memory

From the dashboard (`/dashboard/`), use the **Keep Alive** button to set infinite keep-alive timeout, preventing model unloading between requests.

Or via API:
```bash
curl -X POST http://localhost:8000/api/control_jetson/ \
  -H "Content-Type: application/json" \
  -d '{"action": "keep_alive"}'
```

### Swap Models

Change models via environment variables in `docker-compose.yml` or `.env`:

```yaml
environment:
  - VLM_MODEL=qwen2-vl:7b        # Larger VLM for higher accuracy
  - JUDGE_MODEL=qwen2.5:7b        # Larger judge for better arbitration
```

---

## 6. Performance Monitoring

### Live Telemetry Dashboard

Access at `http://<JETSON-IP>/dashboard/` — provides:

- **CPU/GPU utilization** and temperature
- **RAM and disk usage**
- **Ollama model status** (loaded/unloaded)
- **Inference event feed** (decision type, latency, reasoning)
- **Self-learning sample counter**

### Jetson-Specific Metrics

The telemetry engine reads Jetson sysfs paths for:
- GPU load: `/sys/class/devfreq/17000000.gpu/device/load`
- CPU temperature: `/sys/class/thermal/thermal_zone*/temp`
- GPU temperature: `/sys/class/thermal/thermal_zone*/temp`

---

## 7. Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **GPU not detected** | Ensure `nvidia-container-toolkit` is installed; check `docker info \| grep Runtime` |
| **TensorRT build fails** | Run on the Jetson itself (engines are hardware-specific) |
| **Ollama models not loading** | Check disk space; models need ~4GB free |
| **High latency (>5s)** | Run `optimize_jetson.sh`; check thermal throttling |
| **Screen dims during kiosk** | The app uses Wake Lock API; ensure Chromium has focus |
| **Out of memory** | Reduce Gunicorn workers to 2; enable ZRAM swap |

### Verify GPU Access

```bash
# Inside Docker container:
docker compose exec signo-web python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# On host:
nvidia-smi  # or tegrastats for Jetson
```

### Reset Everything

```bash
docker compose down -v          # Stop + remove volumes
docker compose up -d --build    # Rebuild and start fresh
```
