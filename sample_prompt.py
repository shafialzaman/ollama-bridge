"""
Sample prompts demonstrating Ollama Bridge capabilities.
Runs a quick benchmark + demo across different prompt types.

Usage:
    python sample_prompt.py
    python sample_prompt.py --host 192.168.1.50   # remote workstation
    python sample_prompt.py --model qwen2.5:7b
"""

import argparse
import json
import time

import httpx

BASE = "http://localhost:8000"


def _stream_chat(messages: list[dict], model: str, client: httpx.Client,
                 temperature: float = 0.7) -> tuple[str, float]:
    """Returns (full_response_text, elapsed_seconds)."""
    start = time.monotonic()
    parts = []
    with client.stream(
        "POST", f"{BASE}/chat",
        json={"model": model, "messages": messages, "stream": True, "temperature": temperature},
        timeout=300,
    ) as resp:
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
                chunk = data.get("message", {}).get("content", "")
                if chunk:
                    print(chunk, end="", flush=True)
                    parts.append(chunk)
            except json.JSONDecodeError:
                pass
    elapsed = time.monotonic() - start
    return "".join(parts), elapsed


def run_demo(host: str, port: int, model_override: str | None):
    global BASE
    BASE = f"http://{host}:{port}"

    http = httpx.Client(timeout=300)

    # ── Hardware info ──────────────────────────────────────────────────────
    hw = http.get(f"{BASE}/hardware").json()
    model_list = http.get(f"{BASE}/models").json()

    model = model_override
    if not model:
        available = [m["name"] for m in model_list.get("models", [])]
        recommended = hw.get("recommended_models", [])
        model = next((r for r in recommended if r in available), None)
        if not model and available:
            model = available[0]
        if not model:
            print("No model available. Pull one first:\n  python client.py --pull llama3.2:3b")
            return

    sep = "─" * 56
    print(f"\n{'='*56}")
    print(f"  Ollama Bridge — Sample Prompts")
    print(f"  Backend : {hw.get('backend','?').upper()}  |  VRAM: {hw.get('vram_gb','?')} GB")
    print(f"  Model   : {model}")
    print(f"{'='*56}\n")

    prompts = [
        {
            "label": "1. Concise Reasoning",
            "system": "You are a concise reasoning assistant.",
            "user": "Explain why a transformer's attention mechanism scales quadratically with sequence length, in 3 sentences.",
        },
        {
            "label": "2. Code Generation",
            "system": "You are an expert Python developer. Write clean, production-ready code.",
            "user": (
                "Write a Python async function that calls the Ollama /api/chat endpoint "
                "with streaming, yielding each token chunk as it arrives. "
                "Use httpx.AsyncClient."
            ),
        },
        {
            "label": "3. Hardware-Aware Planning",
            "system": "You are an AI infrastructure architect.",
            "user": (
                f"I have an AI workstation: AMD Ryzen 7 7700X, RTX 3090 (24 GB VRAM), "
                f"64 GB RAM, Linux + CUDA. "
                f"What is the largest quantized LLM I can run fully in VRAM, "
                f"and what quantization format should I use? Be specific."
            ),
        },
        {
            "label": "4. Multi-Turn Context",
            "system": "You are a helpful assistant.",
            "user": None,  # handled as multi-turn below
        },
    ]

    results = []

    for i, p in enumerate(prompts):
        print(f"\n{sep}")
        print(f"  {p['label']}")
        print(sep)

        if p["label"].startswith("4"):
            # Multi-turn demo
            history = [{"role": "system", "content": p["system"]}]
            turns = [
                "What CUDA version does Ollama require on Linux?",
                "And what environment variable sets the GPU layers?",
            ]
            elapsed_total = 0
            full_reply = ""
            for turn in turns:
                print(f"\n  > {turn}\n")
                history.append({"role": "user", "content": turn})
                reply, elapsed = _stream_chat(history, model, http)
                history.append({"role": "assistant", "content": reply})
                elapsed_total += elapsed
                full_reply += reply
            results.append((p["label"], full_reply, elapsed_total))
        else:
            messages = [
                {"role": "system", "content": p["system"]},
                {"role": "user",   "content": p["user"]},
            ]
            print()
            reply, elapsed = _stream_chat(messages, model, http)
            results.append((p["label"], reply, elapsed))

        tps = len(results[-1][1]) / 4 / results[-1][2]
        print(f"\n  [{results[-1][2]:.1f}s  ~{tps:.0f} tok/s]\n")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*56}")
    print(f"  Run Summary — {model}")
    print(f"  Backend: {hw.get('backend','?').upper()}  Platform: {hw.get('platform','?')}")
    print(sep)
    for label, reply, elapsed in results:
        tps = len(reply) / 4 / elapsed
        print(f"  {label[:35]:<35}  {elapsed:5.1f}s  ~{tps:4.0f} tok/s")
    total = sum(r[2] for r in results)
    print(sep)
    print(f"  {'Total':35}  {total:5.1f}s")
    print(f"{'='*56}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ollama Bridge sample prompts")
    parser.add_argument("--host",  default="localhost")
    parser.add_argument("--port",  default=8000, type=int)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    run_demo(args.host, args.port, args.model)
