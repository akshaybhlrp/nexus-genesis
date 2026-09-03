#!/usr/bin/env python3
"""Nexus MoE Interactive Persistent Inference Web Server.

Keeps SmolLM2 MoE permanently loaded in NVIDIA T500 GPU VRAM (1,974 MB)
and proxies requests with sub-second turnaround.
"""

import json
import os
import subprocess
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import urllib.parse

NEXUS_ROOT = Path(__file__).parent.parent
UI_DIR = Path(__file__).parent
SERVE_BIN = NEXUS_ROOT / "target" / "release" / "nexus-serve"
RECTIFIED_MODEL = NEXUS_ROOT / "data" / "models" / "smollm2-135m-rectified"
DEFAULT_MODEL = RECTIFIED_MODEL if RECTIFIED_MODEL.exists() else (NEXUS_ROOT / "data" / "models" / "smollm2-135m")
PORT = 7860

# Persistent GPU inference process
_proc_lock = threading.Lock()
_gpu_worker = None


def start_gpu_worker():
    global _gpu_worker
    with _proc_lock:
        if _gpu_worker is not None and _gpu_worker.poll() is None:
            return

        bin_path = SERVE_BIN if SERVE_BIN.exists() else (NEXUS_ROOT / "target" / "debug" / "nexus-serve")
        print(f"⚡ Starting persistent GPU inference worker: {bin_path.name} on NVIDIA T500...")
        _gpu_worker = subprocess.Popen(
            [str(bin_path), "--model", str(DEFAULT_MODEL)],
            cwd=str(NEXUS_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Wait for the worker to report READY
        ready_line = _gpu_worker.stdout.readline()
        print(f"✓ GPU Worker initialized in VRAM: {ready_line.strip()}")


def query_gpu_worker(prompt: str, tokens: int, temperature: float) -> dict:
    global _gpu_worker
    start_gpu_worker()

    with _proc_lock:
        try:
            req_payload = {
                "prompt": prompt,
                "tokens": tokens,
                "temperature": temperature,
            }
            _gpu_worker.stdin.write(json.dumps(req_payload) + "\n")
            _gpu_worker.stdin.flush()

            resp_line = _gpu_worker.stdout.readline()
            if not resp_line:
                raise RuntimeError("GPU worker closed stdout unexpectedly")

            return json.loads(resp_line.strip())
        except Exception as e:
            # Restart if worker failed
            try:
                _gpu_worker.kill()
            except Exception:
                pass
            _gpu_worker = None
            return {
                "success": False,
                "prompt": prompt,
                "generated_text": "",
                "tokens_generated": 0,
                "elapsed_seconds": 0.0,
                "tokens_per_second": 0.0,
                "error": str(e),
            }


def get_gpu_telemetry():
    """Query nvidia-smi for live hardware telemetry."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "name": parts[0],
            "memory_used_mb": float(parts[1]),
            "memory_total_mb": float(parts[2]),
            "utilization_pct": float(parts[3]),
            "temperature_c": float(parts[4]),
            "status": "active",
        }
    except Exception as e:
        return {
            "name": "NVIDIA T500",
            "memory_used_mb": 1974.0,
            "memory_total_mb": 4096.0,
            "utilization_pct": 0.0,
            "temperature_c": 55.0,
            "status": f"simulated ({e})",
        }


class NexusUIHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/gpu":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            data = get_gpu_telemetry()
            self.wfile.write(json.dumps(data).encode("utf-8"))
        elif parsed.path == "/api/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            models_dir = NEXUS_ROOT / "data" / "models"
            available = []
            if models_dir.exists():
                for d in models_dir.iterdir():
                    if d.is_dir() and (d / "model.safetensors").exists():
                        available.append(d.name)
            self.wfile.write(json.dumps({"models": available}).encode("utf-8"))
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/generate":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                req = json.loads(body)
                prompt = req.get("prompt", "Hello")
                tokens = int(req.get("tokens", 25))
                temperature = float(req.get("temperature", 0.7))

                raw_resp = query_gpu_worker(prompt, tokens, temperature)

                response = {
                    "success": raw_resp.get("success", False),
                    "prompt": prompt,
                    "generated_text": raw_resp.get("generated_text", ""),
                    "tokens_requested": tokens,
                    "tokens_generated": raw_resp.get("tokens_generated", 0),
                    "elapsed_seconds": raw_resp.get("elapsed_seconds", 0.0),
                    "tokens_per_second": raw_resp.get("tokens_per_second", 0.0),
                    "backend": "NVIDIA T500 (CUDA Native)",
                    "model": "smollm2-135m (30 Blocks x 4 Experts MoE)",
                    "active_experts": "Top-2 per Block (60 active / 120 total)",
                    "error": raw_resp.get("error", ""),
                }
            except Exception as e:
                response = {
                    "success": False,
                    "error": str(e),
                }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self.send_error(404)


def run_server():
    start_gpu_worker()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), NexusUIHandler)
    print(f"⚡ Nexus MoE Inference UI running on http://localhost:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
