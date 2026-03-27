# ollama-bridge

Hardware-aware Ollama wrapper that runs on Apple Silicon (Metal) and Linux/CUDA without any code changes. Adds VRAM-aware multi-node routing, a metrics dashboard, and a unified REST API on top of Ollama.

## Hardware profiles

| Hardware | Backend | VRAM | Context | Good models |
|---|---|---|---|---|
| Apple M5 | Metal | ~18 GB | 8 192 | llama3.1:8b, qwen2.5:7b, phi4:14b |
| RTX 3090 | CUDA | 24 GB | 32 768 | llama3.1:70b Q4, qwen2.5:32b, deepseek-r1:32b |

`config.py` auto-detects hardware at startup — no manual toggle needed.

## Quick start (M5 Mac)

```bash
brew install ollama
ollama serve &
pip install -r requirements.txt
ollama pull llama3.2:3b
python server.py
```

Or just run `bash setup.sh` — it handles the Ollama install and model pull.

Then open a second terminal:

```bash
python client.py
```

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/hardware` | Backend, VRAM, context window |
| GET | `/nodes` | All nodes — health, VRAM, latency |
| GET | `/route/explain?model=X` | Show routing decision before sending |
| GET | `/models` | Pulled models + recommendations |
| POST | `/models/pull` | Pull a model (streamed) |
| DELETE | `/models/{name}` | Delete a model |
| POST | `/chat` | Chat completion, auto-routed |
| POST | `/generate` | Raw generation, auto-routed |
| POST | `/embeddings` | Embeddings proxy |
| GET | `/metrics` | Aggregated tok/s, latency, per-model stats |
| GET | `/dashboard` | Live dashboard (5 s refresh) |

Interactive docs: `http://localhost:8000/docs`

## Multi-node setup

Add your Linux workstation as a second node:

```bash
export EXTRA_NODES="workstation|http://192.168.1.50:11434|cuda|24"
python server.py
```

Format: `id|url|backend|vram_gb`, comma-separated for multiple nodes. The registry runs health checks every 30 s and stops routing to unhealthy nodes automatically.

Routing priority: backend match → model already loaded → fewest active requests → lowest latency. Force a specific node with `"node_id": "workstation"` in the request body.

## Workstation (Docker + CUDA)

```bash
# Install NVIDIA Container Toolkit first (one-time)
docker compose up -d
docker exec -it ollama ollama pull llama3.1:70b
```

Then connect from your Mac:

```bash
python client.py --host 192.168.x.x
python sample_prompt.py --host 192.168.x.x --model llama3.1:70b
```

## Environment variables

```
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
OLLAMA_BACKEND=cuda        # force: metal | cuda | cpu
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
EXTRA_NODES=id|url|backend|vram_gb
```

## Files

```
config.py          hardware detection + HardwareProfile dataclass
nodes.py           multi-node registry + async health-check loop
router.py          VRAM-aware request router
metrics.py         per-request stats + dashboard HTML
server.py          FastAPI server
client.py          interactive CLI chat client
sample_prompt.py   4-prompt benchmark runner
Dockerfile         bridge API container
docker-compose.yml Ollama + bridge with CUDA passthrough
setup.sh           one-shot setup script
```
