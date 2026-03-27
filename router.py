"""
Request router — picks the best node for each inference request.

Strategy:
  1. Determine the VRAM requirement for the requested model.
  2. Filter to nodes that are healthy AND can fit the model.
  3. Among candidates, prefer:
       a. Node that already has the model loaded  (avoids cold-load latency)
       b. Node with lowest active_requests        (load balancing)
       c. Node with lowest latency_ms             (tie-break)

Model VRAM registry (approximate, Q4 quantization unless noted):
  These are conservative minimums — actual VRAM depends on quantization level.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from nodes import NodeInfo, NodeRegistry


# ── Model VRAM requirements ────────────────────────────────────────────────────
# Format: pattern → (min_vram_gb, preferred_backend)
# Patterns matched in order; first match wins.

MODEL_VRAM_TABLE: list[tuple[re.Pattern, int, str]] = [
    # tiny / nano models — any backend
    (re.compile(r"(phi3:mini|gemma2:2b|llama3\.2:1b|qwen2\.5:0\.5b|smollm)", re.I), 2, "any"),
    # 3B class
    (re.compile(r"(llama3\.2:3b|phi3:small|qwen2\.5:3b|gemma2:9b-it-q2)", re.I),   4, "any"),
    # 7B / 8B class
    (re.compile(r"(llama3\.1:8b|llama3:8b|mistral:7b|qwen2\.5:7b|deepseek-r1:8b|phi4:14b-q2)", re.I), 8, "any"),
    # 14B class
    (re.compile(r"(phi4:14b|qwen2\.5:14b|deepseek-r1:14b)", re.I),                 12, "any"),
    # 32B class — needs CUDA or large unified memory
    (re.compile(r"(qwen2\.5:32b|deepseek-r1:32b|yi:34b|command-r)", re.I),          20, "cuda"),
    # 70B class — CUDA only at Q4
    (re.compile(r"(llama3\.1:70b|llama3:70b|mixtral:8x7b|deepseek-r1:70b)", re.I), 38, "cuda"),
    # 405B / massive — multi-GPU or CPU offload
    (re.compile(r"(llama3\.1:405b|llama3:405b)", re.I),                             80, "cuda"),
]

FALLBACK_VRAM = 8   # assume 8 GB if model not in table


@dataclass
class RoutingDecision:
    node: NodeInfo
    model: str
    vram_required_gb: int
    preferred_backend: str
    reason: str           # human-readable explanation


def _vram_for_model(model: str) -> tuple[int, str]:
    """Return (min_vram_gb, preferred_backend) for a model name."""
    for pattern, vram, backend in MODEL_VRAM_TABLE:
        if pattern.search(model):
            return vram, backend
    return FALLBACK_VRAM, "any"


def _score_node(node: NodeInfo, model: str, vram_needed: int) -> tuple[int, int, float]:
    """
    Returns a sort key (lower = better).
    (model_loaded_bonus, active_requests, latency_ms)
    """
    loaded_bonus = 0 if model in node.loaded_models else 1
    return (loaded_bonus, node.active_requests, node.latency_ms)


class Router:
    def __init__(self, registry: NodeRegistry):
        self.registry = registry

    def pick(self, model: str) -> Optional[RoutingDecision]:
        """Select the best node for this model. Returns None if no node available."""
        vram_needed, preferred_backend = _vram_for_model(model)

        candidates = [n for n in self.registry.healthy_nodes() if n.can_fit(vram_needed)]

        if not candidates:
            return None

        # Sort: prefer backend match, then score
        def sort_key(n: NodeInfo):
            backend_penalty = 0 if (preferred_backend == "any" or n.backend == preferred_backend) else 1
            score = _score_node(n, model, vram_needed)
            return (backend_penalty,) + score

        candidates.sort(key=sort_key)
        best = candidates[0]

        # Build human-readable reason
        reasons = []
        if model in best.loaded_models:
            reasons.append("model already loaded")
        if preferred_backend != "any" and best.backend == preferred_backend:
            reasons.append(f"backend match ({best.backend})")
        if best.active_requests == 0:
            reasons.append("idle node")
        reason = ", ".join(reasons) if reasons else "best available"

        return RoutingDecision(
            node=best,
            model=model,
            vram_required_gb=vram_needed,
            preferred_backend=preferred_backend,
            reason=reason,
        )

    def pick_or_raise(self, model: str) -> RoutingDecision:
        decision = self.pick(model)
        if decision is None:
            vram_needed, _ = _vram_for_model(model)
            healthy = self.registry.healthy_nodes()
            if not healthy:
                raise RuntimeError("No healthy nodes available.")
            raise RuntimeError(
                f"No node has enough VRAM for '{model}' "
                f"(needs ~{vram_needed} GB). "
                f"Healthy nodes: {[f'{n.id}({n.available_vram_gb}GB)' for n in healthy]}"
            )
        return decision

    def explain(self, model: str) -> dict:
        """Return routing explanation as dict — useful for /route/explain endpoint."""
        vram_needed, preferred_backend = _vram_for_model(model)
        all_nodes = self.registry.all_nodes()
        candidates = []
        for n in all_nodes:
            candidates.append({
                "id": n.id,
                "backend": n.backend,
                "vram_gb": n.vram_gb,
                "available_vram_gb": n.available_vram_gb,
                "healthy": n.healthy,
                "can_fit": n.can_fit(vram_needed),
                "model_loaded": model in n.loaded_models,
                "active_requests": n.active_requests,
                "latency_ms": round(n.latency_ms, 1),
            })
        decision = self.pick(model)
        return {
            "model": model,
            "vram_required_gb": vram_needed,
            "preferred_backend": preferred_backend,
            "selected_node": decision.node.id if decision else None,
            "reason": decision.reason if decision else "no node available",
            "candidates": candidates,
        }
