"""
Interactive CLI client for Ollama Bridge.
Connects to local server (M5) or remote workstation seamlessly.

Usage:
    python client.py                          # auto local
    python client.py --host 192.168.1.50      # connect to workstation
    python client.py --model llama3.1:70b     # pick model
"""

import argparse
import json
import sys
import time
from typing import Iterator, Optional

import httpx


# ── ANSI colours ──────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
BLUE   = "\033[34m"


def _c(text: str, *codes: str) -> str:
    return "".join(codes) + text + RESET


# ── Client ────────────────────────────────────────────────────────────────────

class OllamaBridgeClient:
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.base = f"http://{host}:{port}"
        self.http = httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0))

    def health(self) -> dict:
        return self.http.get(f"{self.base}/health").json()

    def hardware(self) -> dict:
        return self.http.get(f"{self.base}/hardware").json()

    def list_models(self) -> dict:
        return self.http.get(f"{self.base}/models").json()

    def pull_model(self, model: str) -> None:
        print(f"\n{_c('Pulling', BOLD)} {_c(model, CYAN)} ...")
        with self.http.stream("POST", f"{self.base}/models/pull",
                              json={"model": model}, timeout=600) as resp:
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    status = data.get("status", "")
                    total  = data.get("total", 0)
                    compl  = data.get("completed", 0)
                    if total:
                        pct = int(compl / total * 100)
                        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                        print(f"\r  [{bar}] {pct:3d}%  {status[:40]:<40}", end="", flush=True)
                    else:
                        print(f"\r  {status:<60}", end="", flush=True)
                except json.JSONDecodeError:
                    pass
        print(f"\n{_c('Done.', GREEN)}\n")

    def chat_stream(self, messages: list[dict], model: str,
                    temperature: float = 0.7) -> Iterator[str]:
        with self.http.stream(
            "POST", f"{self.base}/chat",
            json={"model": model, "messages": messages,
                  "stream": True, "temperature": temperature},
            timeout=300,
        ) as resp:
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
                except json.JSONDecodeError:
                    pass


# ── Interactive REPL ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a sharp, knowledgeable assistant running on a high-performance AI workstation. "
    "Be concise but thorough. Use code blocks when writing code."
)


def run_repl(client: OllamaBridgeClient, model: str, temperature: float):
    hw = client.hardware()
    backend = hw.get("backend", "unknown").upper()
    vram = hw.get("vram_gb", "?")
    platform = hw.get("platform", "?")

    print(f"""
{_c('  Ollama Bridge — Interactive Chat', BOLD + CYAN)}
  {'─' * 42}
  Backend  : {_c(backend, YELLOW)}  ({platform})
  VRAM     : {_c(str(vram) + ' GB', GREEN)}
  Model    : {_c(model, CYAN)}
  Temp     : {temperature}

  Commands : {_c('/model <name>', DIM)}  {_c('/models', DIM)}  {_c('/pull <name>', DIM)}  {_c('/clear', DIM)}  {_c('/quit', DIM)}
""")

    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input(_c("You  › ", BOLD + GREEN)).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{_c('Bye.', DIM)}")
            break

        if not user_input:
            continue

        # ── Slash commands ─────────────────────────────────────────────────
        if user_input.startswith("/"):
            parts = user_input[1:].split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("quit", "exit", "q"):
                print(_c("Bye.", DIM))
                break

            elif cmd == "clear":
                history = [{"role": "system", "content": SYSTEM_PROMPT}]
                print(_c("  History cleared.", DIM))

            elif cmd == "model":
                if arg:
                    model = arg
                    print(f"  Switched to {_c(model, CYAN)}")
                else:
                    print(f"  Current model: {_c(model, CYAN)}")

            elif cmd == "models":
                data = client.list_models()
                print(f"\n  {_c('Available:', BOLD)}")
                for m in data.get("models", []):
                    tag = _c(" (default)", DIM) if m["name"] == data.get("default") else ""
                    print(f"    • {_c(m['name'], CYAN)}  {_c(str(m['size_gb']) + ' GB', DIM)}{tag}")
                print(f"\n  {_c('Recommended for this hardware:', BOLD)}")
                for m in data.get("recommended", []):
                    print(f"    ◦ {m}")
                print()

            elif cmd == "pull":
                if arg:
                    client.pull_model(arg)
                else:
                    print("  Usage: /pull <model-name>")

            elif cmd == "hw":
                hw = client.hardware()
                for k, v in hw.items():
                    print(f"  {k:20}: {v}")
                print()

            else:
                print(f"  Unknown command: /{cmd}")
            continue

        # ── Normal chat ─────────────────────────────────────────────────────
        history.append({"role": "user", "content": user_input})

        print(_c("\nAssistant › ", BOLD + BLUE), end="", flush=True)
        start = time.monotonic()
        reply_parts = []

        try:
            for chunk in client.chat_stream(history, model, temperature):
                print(chunk, end="", flush=True)
                reply_parts.append(chunk)
        except httpx.ConnectError:
            print(_c("\n[Error] Cannot reach server. Is it running?", RED))
            history.pop()  # remove unanswered user message
            continue
        except KeyboardInterrupt:
            print(_c(" [interrupted]", DIM))

        elapsed = time.monotonic() - start
        reply = "".join(reply_parts)
        history.append({"role": "assistant", "content": reply})

        # tokens-per-second estimate (rough: ~4 chars / token)
        tps = len(reply) / 4 / elapsed if elapsed > 0 else 0
        print(f"\n{_c(f'  [{elapsed:.1f}s  ~{tps:.0f} tok/s]', DIM)}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ollama Bridge CLI")
    parser.add_argument("--host",  default="localhost",  help="Server host")
    parser.add_argument("--port",  default=8000, type=int, help="Server port")
    parser.add_argument("--model", default=None,         help="Model name override")
    parser.add_argument("--temp",  default=0.7, type=float, help="Temperature")
    parser.add_argument("--pull",  default=None, metavar="MODEL",
                        help="Pull a model then exit")
    parser.add_argument("--models", action="store_true", help="List models then exit")
    args = parser.parse_args()

    client = OllamaBridgeClient(host=args.host, port=args.port)

    try:
        client.health()
    except httpx.ConnectError:
        print(_c(f"\n[Error] Cannot reach Ollama Bridge at {client.base}", RED))
        print("  Start it with:  python server.py\n")
        sys.exit(1)

    if args.pull:
        client.pull_model(args.pull)
        sys.exit(0)

    if args.models:
        data = client.list_models()
        for m in data.get("models", []):
            print(f"{m['name']:40}  {m['size_gb']} GB")
        sys.exit(0)

    model = args.model
    if not model:
        data = client.list_models()
        available = [m["name"] for m in data.get("models", [])]
        recommended = data.get("recommended", [])
        # pick first recommended model that is already pulled, else fallback
        model = next((r for r in recommended if r in available), None)
        if not model and available:
            model = available[0]
        if not model:
            model = recommended[0] if recommended else "llama3.2:3b"

    run_repl(client, model, args.temp)


if __name__ == "__main__":
    main()
