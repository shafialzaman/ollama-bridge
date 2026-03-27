"""
Hardware-aware configuration for Ollama Bridge.
Auto-detects M5 (Apple Silicon / Metal) vs Linux + CUDA (RTX 3090).
All settings can be overridden via environment variables or config.json.
"""

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

CONFIG_FILE = Path(__file__).parent / "config.json"


@dataclass
class HardwareProfile:
    backend: str           # "metal" | "cuda" | "cpu"
    platform: str          # "macos-arm64" | "linux-cuda" | "linux-cpu"
    vram_gb: int
    recommended_models: list[str]
    max_context: int
    num_gpu: int           # layers offloaded to GPU (-1 = all)
    ollama_host: str
    ollama_port: int
    server_host: str
    server_port: int

    @property
    def ollama_url(self) -> str:
        return f"http://{self.ollama_host}:{self.ollama_port}"

    @property
    def server_url(self) -> str:
        return f"http://{self.server_host}:{self.server_port}"


def _detect_cuda() -> tuple[bool, int]:
    """Returns (cuda_available, vram_gb)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            vram_mb = int(result.stdout.strip().split("\n")[0])
            return True, vram_mb // 1024
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return False, 0


def _detect_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def detect_hardware() -> HardwareProfile:
    """Auto-detect hardware and return the appropriate profile."""

    # --- Allow full override via environment ---
    env_backend = os.getenv("OLLAMA_BACKEND")  # "metal" | "cuda" | "cpu"
    ollama_host = os.getenv("OLLAMA_HOST", "localhost")
    ollama_port = int(os.getenv("OLLAMA_PORT", "11434"))
    server_host = os.getenv("SERVER_HOST", "0.0.0.0")
    server_port = int(os.getenv("SERVER_PORT", "8000"))

    # --- Apple Silicon (M5 / Metal) ---
    if _detect_apple_silicon() or env_backend == "metal":
        # M5 Pro/Max/Ultra unified memory — treat as ~18GB usable for models
        return HardwareProfile(
            backend="metal",
            platform="macos-arm64",
            vram_gb=18,
            recommended_models=[
                "llama3.2:3b",       # fast, great for chat
                "llama3.1:8b",       # balanced quality/speed
                "mistral:7b",        # strong reasoning
                "qwen2.5:7b",        # excellent code + reasoning
                "deepseek-r1:8b",    # reasoning model
                "phi4:14b",          # fits in M5 unified memory
            ],
            max_context=8192,
            num_gpu=99,             # Metal: all layers on GPU
            ollama_host=ollama_host,
            ollama_port=ollama_port,
            server_host=server_host,
            server_port=server_port,
        )

    # --- Linux + NVIDIA CUDA (RTX 3090 — 24 GB VRAM) ---
    cuda_available, vram_gb = _detect_cuda()
    if cuda_available or env_backend == "cuda":
        vram_gb = vram_gb or 24  # fallback if override
        return HardwareProfile(
            backend="cuda",
            platform="linux-cuda",
            vram_gb=vram_gb,
            recommended_models=[
                "llama3.1:8b",        # everyday chat
                "llama3.1:70b",       # fits in 24 GB at Q4
                "mixtral:8x7b",       # MoE, strong reasoning
                "qwen2.5:32b",        # top-tier code + reasoning
                "deepseek-r1:32b",    # reasoning powerhouse
                "deepseek-coder-v2",  # best open-source coder
            ],
            max_context=32768,
            num_gpu=99,             # CUDA: all layers on GPU
            ollama_host=ollama_host,
            ollama_port=ollama_port,
            server_host=server_host,
            server_port=server_port,
        )

    # --- CPU fallback ---
    return HardwareProfile(
        backend="cpu",
        platform="cpu-only",
        vram_gb=0,
        recommended_models=["llama3.2:3b", "phi3:mini", "gemma2:2b"],
        max_context=4096,
        num_gpu=0,
        ollama_host=ollama_host,
        ollama_port=ollama_port,
        server_host=server_host,
        server_port=server_port,
    )


def load_config() -> HardwareProfile:
    """Load config: env vars > config.json > auto-detect."""
    hw = detect_hardware()

    if CONFIG_FILE.exists():
        try:
            overrides = json.loads(CONFIG_FILE.read_text())
            for k, v in overrides.items():
                if hasattr(hw, k) and v is not None:
                    setattr(hw, k, v)
        except (json.JSONDecodeError, TypeError):
            print(f"[config] Warning: could not parse {CONFIG_FILE}, using defaults.")

    return hw


def save_config(profile: HardwareProfile) -> None:
    CONFIG_FILE.write_text(json.dumps(asdict(profile), indent=2))
    print(f"[config] Saved to {CONFIG_FILE}")


def print_profile(profile: HardwareProfile) -> None:
    lines = [
        "",
        "  Ollama Bridge — Hardware Profile",
        "  " + "─" * 38,
        f"  Backend   : {profile.backend.upper()}",
        f"  Platform  : {profile.platform}",
        f"  VRAM      : {profile.vram_gb} GB",
        f"  GPU layers: {profile.num_gpu} (all)" if profile.num_gpu == 99 else f"  GPU layers: {profile.num_gpu}",
        f"  Context   : {profile.max_context:,} tokens",
        f"  Ollama    : {profile.ollama_url}",
        f"  API Server: {profile.server_url}",
        "",
        "  Recommended models:",
    ]
    for m in profile.recommended_models:
        lines.append(f"    • {m}")
    lines.append("")
    print("\n".join(lines))


if __name__ == "__main__":
    profile = load_config()
    print_profile(profile)
