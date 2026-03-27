# Ollama Bridge — Project Summary

## Drawbacks to Building a Hardware-Aware Ollama Project with Hot-Swapping

Building a full hardware-aware Ollama project that runs natively on Apple Silicon (M5) and
hot-swaps to Linux/CUDA workstations introduces several potential drawbacks:

- **Complexity in Hardware Detection and Switching**: Implementing reliable detection of
  hardware (e.g., via CPU/GPU checks) and seamless hot-swapping can lead to bugs, such as
  incorrect model loading or state loss during transitions. This requires robust error
  handling and testing across environments.

- **Performance Inconsistencies**: Native M5 execution might underperform compared to
  CUDA-accelerated Linux setups for large models, causing user experience disparities.
  Hot-swapping could introduce latency or interruptions if not optimized.

- **Dependency Management**: Ollama and its dependencies (e.g., llama.cpp or CUDA libraries)
  may have version incompatibilities between macOS and Linux, leading to build failures or
  runtime errors. Maintaining cross-platform compatibility increases maintenance overhead.

- **Security and Portability Risks**: Hot-swapping might expose vulnerabilities if not
  secured (e.g., data leaks during transfers). Portability could be limited by
  platform-specific optimizations, making deployment harder on unsupported hardware.

- **Development and Testing Overhead**: Requires extensive testing on both platforms,
  increasing time and resources. Debugging cross-environment issues (e.g., via logs) can
  be challenging without unified tooling.

> **Mitigation approach used in this project**: `config.py` uses conditional detection
> (subprocess + platform checks) with a clean `HardwareProfile` abstraction layer so all
> other code is hardware-agnostic. CUDA availability is checked before initializing any
> GPU-specific settings. Environment variable overrides allow manual forcing of any backend
> without code changes.

---

## What This Is

A hardware-aware Ollama wrapper that runs natively on your **M5 Mac today** and
ports without any code changes to your **Linux + CUDA workstation** (RTX 3090).

---

## Hardware Profiles

| Hardware             | Backend | VRAM    | Context  | Best Models                  |
|----------------------|---------|---------|----------|------------------------------|
| Apple M5 (you now)   | Metal   | ~18 GB  | 8 192    | llama3.1:8b, qwen2.5:7b, phi4:14b |
| RTX 3090 (workstation) | CUDA  | 24 GB   | 32 768   | llama3.1:70b Q4, qwen2.5:32b, deepseek-r1:32b |

`config.py` auto-detects which hardware is present at startup — no manual toggle needed.

---

## File Map

```
ollama-bridge/
├── config.py          # Hardware detection + HardwareProfile dataclass
├── nodes.py           # Multi-node registry + async health-check loop
├── router.py          # VRAM-aware request router (best-node selection)
├── metrics.py         # Per-request stats collector + dashboard HTML
├── server.py          # FastAPI server (routing, metrics, multi-node)
├── client.py          # Interactive CLI chat client
├── sample_prompt.py   # 4-prompt benchmark / demo runner
├── requirements.txt   # Python deps (fastapi, uvicorn, httpx, pydantic)
├── Dockerfile         # Container for the bridge API (workstation)
├── docker-compose.yml # Ollama + bridge containers with CUDA GPU passthrough
├── setup.sh           # One-shot setup: installs Ollama, pulls starter model
└── SUMMARY.md         # This file
```

---

## Quick Start (M5 Mac — right now)

```bash
# 1. Install Ollama (if not already)
brew install ollama
ollama serve &

# 2. Install deps
pip install -r requirements.txt

# 3. Pull a model
ollama pull llama3.2:3b

# 4. Start the bridge server
python server.py

# 5. Chat
python client.py
```

Or just run `bash setup.sh` — it handles steps 1-3 automatically.

---

## API Endpoints

| Method | Path                  | Description                                      |
|--------|-----------------------|--------------------------------------------------|
| GET    | `/health`             | Liveness check + node count                      |
| GET    | `/hardware`           | Full hardware profile (backend/VRAM)             |
| GET    | `/nodes`              | All nodes — health, VRAM, latency, active reqs   |
| GET    | `/nodes/{id}`         | Single node detail                               |
| GET    | `/route/explain`      | Show which node would be picked for a model      |
| GET    | `/models`             | List pulled models + recommendations             |
| POST   | `/models/pull`        | Pull a model (streamed progress, any node)       |
| DELETE | `/models/{name}`      | Delete a model                                   |
| POST   | `/chat`               | Chat completion — auto-routed, streaming or full |
| POST   | `/generate`           | Raw generation — auto-routed                     |
| POST   | `/embeddings`         | Embeddings proxy (for RAG)                       |
| GET    | `/metrics`            | Aggregated stats: tok/s, latency, per model/node |
| GET    | `/metrics/recent`     | Last N requests with latency + tok/s             |
| GET    | `/dashboard`          | Live benchmark dashboard (auto-refreshes 5 s)    |

Interactive docs: `http://localhost:8000/docs`

---

## Upgrade 1 — Model Routing (`router.py`)

Every `/chat` and `/generate` request is automatically routed to the best node:

```
request: llama3.1:70b
  → needs ~38 GB VRAM
  → local M5 (18 GB) — cannot fit
  → workstation RTX 3090 (24 GB) — fits ✓  model already loaded ✓
  → routed to: workstation
```

Routing priority: backend match → model already loaded → lowest active requests → lowest latency.
Force a specific node with `"node_id": "workstation"` in the request body.

```bash
# See routing decision before sending a request
curl "http://localhost:8000/route/explain?model=llama3.1:70b"
```

---

## Upgrade 2 — Multi-Node Support (`nodes.py`)

Add your Linux workstation as a second node via environment variable:

```bash
# format: id|url|backend|vram_gb  (pipe-separated, comma between nodes)
export EXTRA_NODES="workstation|http://192.168.1.50:11434|cuda|24"
python server.py
```

The registry runs a background health-check loop every 30 s.
If a node goes down, the router automatically stops sending requests to it.

```bash
# Check all nodes live
curl http://localhost:8000/nodes
```

---

## Upgrade 3 — Benchmark Dashboard (`metrics.py`)

Open in browser after starting the server:

```
http://localhost:8000/dashboard
```

Shows live (5 s auto-refresh):
- Total requests / errors / uptime
- Per-node: VRAM available, active requests, latency
- Rolling average tok/s and latency (last 50 requests)
- Per-model breakdown: request count, errors, avg latency, avg tok/s
- Recent request log with model, node, status

---

## Connecting to the Workstation

Once your Linux workstation is up:

```bash
# On the workstation
docker compose up -d           # starts Ollama + bridge with CUDA

# From your Mac (or any machine on the LAN)
python client.py --host 192.168.x.x

# Run sample benchmarks against the workstation
python sample_prompt.py --host 192.168.x.x --model llama3.1:70b
```

The same client and API work identically — only `--host` changes.

---

## Environment Overrides

All settings can be overridden without touching code:

```bash
OLLAMA_HOST=localhost      # where Ollama is running
OLLAMA_PORT=11434
OLLAMA_BACKEND=cuda        # force backend: metal | cuda | cpu
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
```

Or edit `config.json` (created after first run with `python config.py`).

---

## Workstation Docker Setup

```bash
# Install NVIDIA Container Toolkit first (one-time)
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/

docker compose up -d

# Pull models inside the container
docker exec -it ollama ollama pull llama3.1:70b
docker exec -it ollama ollama pull qwen2.5:32b
```

---

## Sample Prompt Demo

```bash
python sample_prompt.py
```

Runs 4 prompts and prints a benchmark table (seconds + tok/s per prompt).
Great for comparing M5 vs RTX 3090 performance side-by-side.
