import os
import time
import threading
import requests
import psutil
from collections import deque
from typing import Dict, List, Any

_LOCK = threading.Lock()
_LOG_HISTORY: deque = deque(maxlen=100)
_STATS = {
    "total_requests": 0,
    "consensus_count": 0,
    "judge_count": 0,
    "failed_count": 0,
    "dialects": {},
    "latencies_ms": deque(maxlen=50),
    "start_time": time.time()
}


def log_inference_event(
    lstm_pred: str,
    lstm_conf: float,
    vlm_pred: str,
    final_pred: str,
    decision_type: str,  # "consensus", "judge", "lstm_only", "fallback"
    reasoning: str,
    dialect: str,
    latency_ms: float,
    has_video: bool = True
):
    """Record an inference event in the telemetry ring buffer."""
    with _LOCK:
        _STATS["total_requests"] += 1
        if decision_type == "consensus":
            _STATS["consensus_count"] += 1
        elif decision_type == "judge":
            _STATS["judge_count"] += 1
        elif decision_type == "failed":
            _STATS["failed_count"] += 1
        elif decision_type == "cache_hit":
            _STATS["cache_hit_count"] = _STATS.get("cache_hit_count", 0) + 1
        elif decision_type == "lstm_only":
            _STATS["lstm_only_count"] = _STATS.get("lstm_only_count", 0) + 1
        elif decision_type == "fallback":
            _STATS["fallback_count"] = _STATS.get("fallback_count", 0) + 1

        _STATS["dialects"][dialect] = _STATS["dialects"].get(dialect, 0) + 1
        _STATS["latencies_ms"].append(latency_ms)

        event = {
            "id": _STATS["total_requests"],
            "timestamp": time.strftime("%H:%M:%S"),
            "lstm_pred": lstm_pred,
            "lstm_conf": int(lstm_conf * 100),
            "vlm_pred": vlm_pred or "N/A",
            "final_pred": final_pred,
            "decision_type": decision_type,
            "reasoning": reasoning or ("Models agreed" if decision_type == "consensus" else "LSTM direct prediction"),
            "dialect": dialect,
            "latency_ms": round(latency_ms, 1),
            "has_video": has_video
        }
        _LOG_HISTORY.appendleft(event)


def clear_telemetry_logs():
    """Reset stored telemetry logs."""
    with _LOCK:
        _LOG_HISTORY.clear()
        _STATS["total_requests"] = 0
        _STATS["consensus_count"] = 0
        _STATS["judge_count"] = 0
        _STATS["failed_count"] = 0
        _STATS["dialects"] = {}
        _STATS["latencies_ms"].clear()


def check_ollama_status() -> Dict[str, Any]:
    """Check connection and list loaded models in Ollama."""
    ollama_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        res = requests.get(f"{ollama_url}/api/tags", timeout=1.5)
        if res.status_code == 200:
            models = [m.get("name") for m in res.json().get("models", [])]
            return {
                "online": True,
                "url": ollama_url,
                "models": models
            }
    except Exception:
        pass
    return {
        "online": False,
        "url": ollama_url,
        "models": []
    }


def check_gpu_status() -> Dict[str, Any]:
    """Inspect GPU status via TensorFlow physical devices."""
    gpu_available = False
    device_count = 0
    tf_version = "Unknown"
    try:
        import tensorflow as tf
        tf_version = tf.__version__
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            gpu_available = True
            device_count = len(gpus)
    except Exception:
        pass

    return {
        "gpu_available": gpu_available,
        "device_count": device_count,
        "tf_version": tf_version
    }


def get_telemetry_snapshot() -> Dict[str, Any]:
    """Generate full live telemetry payload for frontend dashboard."""
    with _LOCK:
        latencies = list(_STATS["latencies_ms"])
        avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0.0
        
        tot = _STATS["total_requests"]
        consensus_count = _STATS["consensus_count"]
        judge_count = _STATS["judge_count"]
        cache_hit_count = _STATS.get("cache_hit_count", 0)
        lstm_only_count = _STATS.get("lstm_only_count", 0)
        fallback_count = _STATS.get("fallback_count", 0)
        failed_count = _STATS["failed_count"]
        consensus_rate = round((consensus_count / tot) * 100, 1) if tot > 0 else 100.0
        judge_rate = round((judge_count / tot) * 100, 1) if tot > 0 else 0.0

        logs_list = list(_LOG_HISTORY)[:30]
        dialects_dict = dict(_STATS["dialects"])
        start_time = _STATS["start_time"]

    # CPU & RAM stats
    cpu_percent = psutil.cpu_percent(interval=None)
    cpu_count = psutil.cpu_count()
    mem = psutil.virtual_memory()
    try:
        disk = psutil.disk_usage(os.path.abspath(os.sep))
    except Exception:
        disk = psutil.disk_usage('.')

    uptime_sec = int(time.time() - start_time)
    hours, remainder = divmod(uptime_sec, 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime_str = f"{hours}h {minutes}m {seconds}s"

    ollama_info = check_ollama_status()
    gpu_info = check_gpu_status()

    # Read Jetson Hardware Specific Metrics
    gpu_load = get_jetson_gpu_load()
    cpu_temp = get_thermal_temperature("cpu-thermal")
    gpu_temp = get_thermal_temperature("gpu-thermal")

    return {
        "status": "online",
        "system": {
            "cpu_percent": cpu_percent,
            "cpu_count": cpu_count,
            "memory_used_mb": round(mem.used / (1024 * 1024), 1),
            "memory_total_mb": round(mem.total / (1024 * 1024), 1),
            "memory_percent": mem.percent,
            "disk_percent": disk.percent,
            "uptime": uptime_str,
            "gpu_load": gpu_load,
            "cpu_temp": cpu_temp,
            "gpu_temp": gpu_temp
        },
        "ai_engine": {
            "gpu": gpu_info,
            "ollama": ollama_info
        },
        "inference_metrics": {
            "total_requests": tot,
            "consensus_count": consensus_count,
            "judge_count": judge_count,
            "cache_hit_count": cache_hit_count,
            "lstm_only_count": lstm_only_count,
            "fallback_count": fallback_count,
            "failed_count": failed_count,
            "consensus_rate_pct": consensus_rate,
            "judge_rate_pct": judge_rate,
            "avg_latency_ms": avg_latency,
            "dialects": dialects_dict
        },
        "recent_logs": logs_list
    }


def get_jetson_gpu_load() -> float:
    """Read GPU activity load from Tegra devfreq sysfs on Jetson boards."""
    path = "/sys/class/devfreq/17000000.gpu/device/load"
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                val = int(f.read().strip())
                # Out of 1000, convert to standard percentage
                return round(val / 10.0, 1)
        except Exception:
            pass
    return 0.0


def get_thermal_temperature(zone_type: str) -> float:
    """Read system temperature in degrees Celsius for a specific thermal zone type."""
    base_path = "/sys/class/thermal"
    if os.path.exists(base_path):
        try:
            for zone in os.listdir(base_path):
                if zone.startswith("thermal_zone"):
                    z_path = os.path.join(base_path, zone)
                    type_file = os.path.join(z_path, "type")
                    temp_file = os.path.join(z_path, "temp")
                    if os.path.exists(type_file) and os.path.exists(temp_file):
                        with open(type_file, "r") as tf:
                            if tf.read().strip() == zone_type:
                                with open(temp_file, "r") as tmpf:
                                    return round(int(tmpf.read().strip()) / 1000.0, 1)
        except Exception:
            pass
    return 0.0

