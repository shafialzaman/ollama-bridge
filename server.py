"""
FastAPI server wrapping Ollama.
Exposes a unified REST API identical on M5 (Metal) and Linux (CUDA).

Upgrades wired in:
  - router.py   → picks best node per request (model-size + VRAM-aware)
  - nodes.py    → multi-node registry with async health checks
  - metrics.py  → per-request stats + /metrics + /dashboard
"""

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from config import HardwareProfile, load_config, print_profile
from metrics import DASHBOARD_HTML, MetricsCollector
from nodes import NodeInfo, NodeRegistry, build_registry_from_env
from router import Router

# ── Globals ────────────────────────────────────────────────────────────────────
profile:  HardwareProfile  = None  # type: ignore
registry: NodeRegistry     = None  # type: ignore
router:   Router           = None  # type: ignore
metrics:  MetricsCollector = None  # type: ignore


@asynccontextmanager
async def lifespan(app: FastAPI):
    global profile, registry, router, metrics

    profile  = load_config()
    print_profile(profile)

    registry = build_registry_from_env()
    router   = Router(registry)
    metrics  = MetricsCollector()

    await registry.start()
    print(f"[server] Node registry started — {len(registry.all_nodes())} node(s) registered.")

    yield

    await registry.stop()


app = FastAPI(
    title="Ollama Bridge",
    description="Hardware-aware Ollama API — M5 Metal and Linux CUDA, multi-node routing",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model:       Optional[str]   = None
    messages:    list[ChatMessage]
    stream:      bool            = True
    temperature: Optional[float] = 0.7
    max_tokens:  Optional[int]   = None
    node_id:     Optional[str]   = None   # force a specific node


class GenerateRequest(BaseModel):
    model:       Optional[str]   = None
    prompt:      str
    stream:      bool            = True
    temperature: Optional[float] = 0.7
    system:      Optional[str]   = None
    node_id:     Optional[str]   = None


class PullRequest(BaseModel):
    model:   str
    node_id: Optional[str] = None   # pull on a specific node; default = local


# ── Helpers ────────────────────────────────────────────────────────────────────

def _default_model() -> str:
    return profile.recommended_models[0]


def _ollama_options(node: NodeInfo) -> dict:
    return {
        "num_gpu": node.vram_gb if node.backend != "cpu" else 0,
        "num_ctx": profile.max_context,
    }


def _node_client(node: NodeInfo) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=node.ollama_url,
        timeout=httpx.Timeout(300.0, connect=10.0),
    )


def _pick_node(model: str, node_id_override: Optional[str] = None) -> NodeInfo:
    if node_id_override:
        node = registry.get(node_id_override)
        if node is None:
            raise HTTPException(status_code=404, detail=f"Node '{node_id_override}' not found.")
        if not node.healthy:
            raise HTTPException(status_code=503, detail=f"Node '{node_id_override}' is unhealthy.")
        return node
    try:
        decision = router.pick_or_raise(model)
        return decision.node
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


async def _stream_from_node(node: NodeInfo, path: str, payload: dict) -> AsyncIterator[str]:
    async with _node_client(node) as client:
        async with client.stream("POST", path, json=payload) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise HTTPException(status_code=resp.status_code, detail=body.decode())
            async for line in resp.aiter_lines():
                if line:
                    yield line + "\n"


# ── Routes — health / info ─────────────────────────────────────────────────────

@app.get("/health")
async def health():
    healthy_count = len(registry.healthy_nodes())
    return {
        "status": "ok" if healthy_count > 0 else "degraded",
        "backend": profile.backend,
        "platform": profile.platform,
        "healthy_nodes": healthy_count,
        "total_nodes": len(registry.all_nodes()),
    }


@app.get("/hardware")
async def hardware_info():
    return {
        "backend": profile.backend,
        "platform": profile.platform,
        "vram_gb": profile.vram_gb,
        "num_gpu": profile.num_gpu,
        "max_context": profile.max_context,
        "recommended_models": profile.recommended_models,
        "ollama_url": profile.ollama_url,
    }


# ── Routes — node management ──────────────────────────────────────────────────

@app.get("/nodes")
async def list_nodes():
    """All nodes with health, VRAM, latency, active request count."""
    return registry.status()


@app.get("/nodes/{node_id}")
async def get_node(node_id: str):
    node = registry.get(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found.")
    return next(n for n in registry.status() if n["id"] == node_id)


# ── Routes — routing ──────────────────────────────────────────────────────────

@app.get("/route/explain")
async def route_explain(model: str):
    """Show which node would be selected for a model and why."""
    return router.explain(model)


# ── Routes — model management ─────────────────────────────────────────────────

@app.get("/models")
async def list_models(node_id: Optional[str] = None):
    target_id = node_id or "local"
    node = registry.get(target_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node '{target_id}' not found.")
    async with _node_client(node) as client:
        resp = await client.get("/api/tags")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="Failed to list models")
    data = resp.json()
    return {
        "node": target_id,
        "models": [
            {"name": m["name"], "size_gb": round(m.get("size", 0) / 1e9, 1)}
            for m in data.get("models", [])
        ],
        "recommended": profile.recommended_models,
        "default": _default_model(),
    }


@app.post("/models/pull")
async def pull_model(req: PullRequest):
    node = registry.get(req.node_id or "local")
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found.")

    async def _stream():
        async with _node_client(node) as client:
            async with client.stream("POST", "/api/pull",
                                     json={"name": req.model, "stream": True}) as resp:
                async for line in resp.aiter_lines():
                    if line:
                        yield line + "\n"

    return StreamingResponse(_stream(), media_type="application/x-ndjson")


@app.delete("/models/{model_name:path}")
async def delete_model(model_name: str, node_id: Optional[str] = None):
    node = registry.get(node_id or "local")
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found.")
    async with _node_client(node) as client:
        resp = await client.delete("/api/delete", json={"name": model_name})
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"Could not delete {model_name}")
    return {"deleted": model_name, "node": node.id}


# ── Routes — inference ────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(req: ChatRequest):
    model  = req.model or _default_model()
    node   = _pick_node(model, req.node_id)
    req_id = str(uuid.uuid4())[:8]

    payload = {
        "model": model,
        "messages": [m.model_dump() for m in req.messages],
        "stream": req.stream,
        "options": {
            **_ollama_options(node),
            "temperature": req.temperature,
            **({"num_predict": req.max_tokens} if req.max_tokens else {}),
        },
    }

    node.active_requests += 1
    rec = metrics.start_request(req_id, model, node.id, node.backend)

    if req.stream:
        async def _stream_and_track():
            token_count = 0
            try:
                async for chunk in _stream_from_node(node, "/api/chat", payload):
                    try:
                        data = json.loads(chunk.strip())
                        if data.get("message", {}).get("content"):
                            token_count += 1
                    except json.JSONDecodeError:
                        pass
                    yield chunk
                metrics.finish_request(rec, token_count)
            except Exception as e:
                metrics.finish_request(rec, token_count, success=False, error=str(e))
                raise
            finally:
                node.active_requests = max(0, node.active_requests - 1)

        return StreamingResponse(_stream_and_track(), media_type="application/x-ndjson")

    try:
        async with _node_client(node) as client:
            resp = await client.post("/api/chat", json=payload)
        if resp.status_code != 200:
            metrics.finish_request(rec, 0, success=False, error=resp.text)
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        data = resp.json()
        metrics.finish_request(rec, len(data.get("message", {}).get("content", "")) // 4)
        return data
    finally:
        node.active_requests = max(0, node.active_requests - 1)


@app.post("/generate")
async def generate(req: GenerateRequest):
    model  = req.model or _default_model()
    node   = _pick_node(model, req.node_id)
    req_id = str(uuid.uuid4())[:8]

    payload = {
        "model": model,
        "prompt": req.prompt,
        "stream": req.stream,
        "options": {**_ollama_options(node), "temperature": req.temperature},
    }
    if req.system:
        payload["system"] = req.system

    node.active_requests += 1
    rec = metrics.start_request(req_id, model, node.id, node.backend)

    if req.stream:
        async def _stream_and_track():
            token_count = 0
            try:
                async for chunk in _stream_from_node(node, "/api/generate", payload):
                    try:
                        if json.loads(chunk.strip()).get("response"):
                            token_count += 1
                    except json.JSONDecodeError:
                        pass
                    yield chunk
                metrics.finish_request(rec, token_count)
            except Exception as e:
                metrics.finish_request(rec, token_count, success=False, error=str(e))
                raise
            finally:
                node.active_requests = max(0, node.active_requests - 1)

        return StreamingResponse(_stream_and_track(), media_type="application/x-ndjson")

    try:
        async with _node_client(node) as client:
            resp = await client.post("/api/generate", json=payload)
        if resp.status_code != 200:
            metrics.finish_request(rec, 0, success=False, error=resp.text)
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        data = resp.json()
        metrics.finish_request(rec, len(data.get("response", "")) // 4)
        return data
    finally:
        node.active_requests = max(0, node.active_requests - 1)


@app.post("/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    model = body.get("model", _default_model())
    node = _pick_node(model)
    async with _node_client(node) as client:
        resp = await client.post("/api/embeddings", json=body)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


# ── Routes — metrics + dashboard ──────────────────────────────────────────────

@app.get("/metrics")
async def get_metrics():
    """Aggregated stats: tok/s, latency, per-model breakdown, per-node breakdown."""
    return metrics.summary()


@app.get("/metrics/recent")
async def recent_requests(limit: int = 20):
    """Last N requests with latency and tok/s."""
    return metrics.recent_requests(limit=limit)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Live benchmark dashboard (auto-refreshes every 5 s)."""
    return HTMLResponse(content=DASHBOARD_HTML)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    cfg = load_config()
    uvicorn.run(
        "server:app",
        host=cfg.server_host,
        port=cfg.server_port,
        reload=False,
        log_level="info",
    )
