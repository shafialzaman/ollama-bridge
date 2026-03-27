"""
Multi-node registry with async health checking.
Tracks all available inference nodes (local Metal, remote CUDA, etc.)
and maintains real-time availability + hardware profiles.

Architecture:
    bridge
      ├── node: local  (M5 / Metal)
      ├── node: gpu-1  (RTX 3090 / CUDA)
      └── node: gpu-2  (future)
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx


@dataclass
class NodeInfo:
    id: str                         # e.g. "local", "workstation"
    ollama_url: str                 # e.g. "http://localhost:11434"
    backend: str                    # "metal" | "cuda" | "cpu"
    vram_gb: int
    label: str = ""
    healthy: bool = False
    last_checked: float = 0.0
    latency_ms: float = 0.0
    loaded_models: list[str] = field(default_factory=list)
    active_requests: int = 0        # in-flight request counter

    @property
    def available_vram_gb(self) -> int:
        """Rough estimate: subtract ~2 GB per active request."""
        used = self.active_requests * 2
        return max(0, self.vram_gb - used)

    def can_fit(self, required_vram_gb: int) -> bool:
        return self.healthy and self.available_vram_gb >= required_vram_gb


class NodeRegistry:
    """
    Manages a pool of Ollama nodes.
    Runs a background health-check loop and exposes healthy nodes for routing.
    """

    HEALTH_INTERVAL = 30.0   # seconds between checks
    HEALTH_TIMEOUT  = 5.0    # per-node timeout

    def __init__(self):
        self._nodes: dict[str, NodeInfo] = {}
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None

    def register(self, node: NodeInfo) -> None:
        self._nodes[node.id] = node

    def register_many(self, nodes: list[NodeInfo]) -> None:
        for n in nodes:
            self.register(n)

    async def start(self) -> None:
        """Start background health-check loop."""
        await self._check_all()   # immediate first pass
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _check_node(self, node: NodeInfo, client: httpx.AsyncClient) -> None:
        t0 = time.monotonic()
        try:
            resp = await client.get(f"{node.ollama_url}/api/tags", timeout=self.HEALTH_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            async with self._lock:
                node.healthy = True
                node.latency_ms = (time.monotonic() - t0) * 1000
                node.last_checked = time.time()
                node.loaded_models = [m["name"] for m in data.get("models", [])]
        except Exception:
            async with self._lock:
                node.healthy = False
                node.last_checked = time.time()

    async def _check_all(self) -> None:
        async with httpx.AsyncClient() as client:
            await asyncio.gather(*[self._check_node(n, client) for n in self._nodes.values()])

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self.HEALTH_INTERVAL)
            await self._check_all()

    # ── Queries ───────────────────────────────────────────────────────────────

    def healthy_nodes(self) -> list[NodeInfo]:
        return [n for n in self._nodes.values() if n.healthy]

    def get(self, node_id: str) -> Optional[NodeInfo]:
        return self._nodes.get(node_id)

    def all_nodes(self) -> list[NodeInfo]:
        return list(self._nodes.values())

    def status(self) -> list[dict]:
        return [
            {
                "id": n.id,
                "label": n.label or n.id,
                "backend": n.backend,
                "vram_gb": n.vram_gb,
                "available_vram_gb": n.available_vram_gb,
                "healthy": n.healthy,
                "latency_ms": round(n.latency_ms, 1),
                "last_checked": n.last_checked,
                "loaded_models": n.loaded_models,
                "active_requests": n.active_requests,
                "ollama_url": n.ollama_url,
            }
            for n in self._nodes.values()
        ]


# ── Factory: build registry from config ───────────────────────────────────────

def build_registry_from_env() -> NodeRegistry:
    """
    Build a NodeRegistry from environment variables.

    Always adds the local Ollama instance.
    Additional nodes can be added via EXTRA_NODES env var:
        EXTRA_NODES=gpu1:http://192.168.1.50:11434:cuda:24,gpu2:http://10.0.0.5:11434:cuda:24
        format: id:url:backend:vram_gb  (comma-separated)
    """
    import os
    from config import load_config

    cfg = load_config()
    registry = NodeRegistry()

    # Local node (always present)
    registry.register(NodeInfo(
        id="local",
        label=f"Local ({cfg.backend.upper()})",
        ollama_url=cfg.ollama_url,
        backend=cfg.backend,
        vram_gb=cfg.vram_gb,
    ))

    # Extra remote nodes
    extra = os.getenv("EXTRA_NODES", "")
    for entry in extra.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":")
        # id:http:// needs special handling since URL contains ':'
        # format: id|url|backend|vram
        parts2 = entry.split("|")
        if len(parts2) == 4:
            nid, url, backend, vram = parts2
            registry.register(NodeInfo(
                id=nid.strip(),
                label=f"{nid.strip()} ({backend.upper()})",
                ollama_url=url.strip(),
                backend=backend.strip(),
                vram_gb=int(vram.strip()),
            ))

    return registry
