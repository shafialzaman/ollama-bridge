"""
In-process metrics collector.
Tracks per-request stats: latency, tokens/sec, node, model, VRAM.
Exposes aggregated stats for the dashboard and /metrics endpoint.

No external dependencies — pure stdlib + dataclasses.
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional


@dataclass
class RequestRecord:
    request_id: str
    model: str
    node_id: str
    backend: str
    started_at: float
    ended_at: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0
    success: bool = True
    error: Optional[str] = None

    @property
    def latency_ms(self) -> float:
        end = self.ended_at or time.monotonic()
        return (end - self.started_at) * 1000

    @property
    def tokens_per_sec(self) -> float:
        elapsed = (self.ended_at or time.monotonic()) - self.started_at
        if elapsed <= 0 or self.output_tokens == 0:
            return 0.0
        return self.output_tokens / elapsed


class MetricsCollector:
    """
    Thread-safe metrics store.
    Keeps the last MAX_RECORDS requests in a ring buffer.
    """

    MAX_RECORDS = 500

    def __init__(self):
        self._lock = Lock()
        self._records: deque[RequestRecord] = deque(maxlen=self.MAX_RECORDS)
        self._total_requests = 0
        self._total_errors = 0
        self._started_at = time.time()

    # ── Recording ─────────────────────────────────────────────────────────────

    def start_request(self, request_id: str, model: str,
                      node_id: str, backend: str) -> RequestRecord:
        rec = RequestRecord(
            request_id=request_id,
            model=model,
            node_id=node_id,
            backend=backend,
            started_at=time.monotonic(),
        )
        with self._lock:
            self._records.append(rec)
            self._total_requests += 1
        return rec

    def finish_request(self, rec: RequestRecord, output_tokens: int,
                       success: bool = True, error: Optional[str] = None) -> None:
        rec.ended_at = time.monotonic()
        rec.output_tokens = output_tokens
        rec.success = success
        rec.error = error
        if not success:
            with self._lock:
                self._total_errors += 1

    # ── Aggregation ───────────────────────────────────────────────────────────

    def summary(self) -> dict:
        with self._lock:
            records = list(self._records)

        completed = [r for r in records if r.ended_at > 0]
        recent = completed[-50:] if completed else []   # last 50 for rolling stats

        def _avg(vals):
            return round(sum(vals) / len(vals), 1) if vals else 0.0

        # Per-model breakdown
        by_model: dict[str, list[RequestRecord]] = defaultdict(list)
        by_node:  dict[str, list[RequestRecord]] = defaultdict(list)
        for r in completed:
            by_model[r.model].append(r)
            by_node[r.node_id].append(r)

        model_stats = {}
        for model, reqs in by_model.items():
            tps_vals = [r.tokens_per_sec for r in reqs if r.tokens_per_sec > 0]
            lat_vals  = [r.latency_ms for r in reqs]
            model_stats[model] = {
                "requests": len(reqs),
                "errors": sum(1 for r in reqs if not r.success),
                "avg_latency_ms": _avg(lat_vals),
                "avg_tokens_per_sec": _avg(tps_vals),
            }

        node_stats = {}
        for nid, reqs in by_node.items():
            tps_vals = [r.tokens_per_sec for r in reqs if r.tokens_per_sec > 0]
            node_stats[nid] = {
                "requests": len(reqs),
                "errors": sum(1 for r in reqs if not r.success),
                "avg_tokens_per_sec": _avg(tps_vals),
            }

        recent_tps = [r.tokens_per_sec for r in recent if r.tokens_per_sec > 0]
        recent_lat = [r.latency_ms for r in recent]

        uptime = time.time() - self._started_at

        return {
            "uptime_seconds": round(uptime, 1),
            "total_requests": self._total_requests,
            "total_errors": self._total_errors,
            "completed_requests": len(completed),
            "rolling_avg_tokens_per_sec": _avg(recent_tps),
            "rolling_avg_latency_ms": _avg(recent_lat),
            "by_model": model_stats,
            "by_node": node_stats,
        }

    def recent_requests(self, limit: int = 20) -> list[dict]:
        with self._lock:
            records = list(self._records)
        return [
            {
                "request_id": r.request_id,
                "model": r.model,
                "node": r.node_id,
                "backend": r.backend,
                "latency_ms": round(r.latency_ms, 1),
                "tokens_per_sec": round(r.tokens_per_sec, 1),
                "output_tokens": r.output_tokens,
                "success": r.success,
                "error": r.error,
            }
            for r in reversed(records)
        ][:limit]


# ── Dashboard HTML ────────────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Ollama Bridge — Dashboard</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'SF Mono', 'Fira Code', monospace; background: #0d0d0d; color: #e0e0e0; padding: 2rem; }
    h1   { font-size: 1.2rem; color: #00d4ff; margin-bottom: 0.25rem; letter-spacing: 0.05em; }
    h2   { font-size: 0.85rem; color: #888; margin: 1.5rem 0 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 1rem; }
    .card { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 8px; padding: 1rem; }
    .card .label { font-size: 0.7rem; color: #666; margin-bottom: 0.3rem; }
    .card .value { font-size: 1.6rem; color: #00d4ff; font-weight: bold; }
    .card .unit  { font-size: 0.75rem; color: #555; }
    table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
    th    { text-align: left; color: #555; font-weight: normal; padding: 0.4rem 0.6rem; border-bottom: 1px solid #222; }
    td    { padding: 0.35rem 0.6rem; border-bottom: 1px solid #1a1a1a; color: #ccc; }
    tr:hover td { background: #1a1a1a; }
    .ok   { color: #00c97a; }
    .err  { color: #ff4f4f; }
    .tag  { display: inline-block; padding: 0.1rem 0.4rem; border-radius: 3px; font-size: 0.7rem; }
    .cuda  { background: #1a3a1a; color: #00c97a; }
    .metal { background: #1a1a3a; color: #6b8aff; }
    .cpu   { background: #2a2a1a; color: #aaa; }
    footer { margin-top: 2rem; font-size: 0.7rem; color: #333; }
    #refresh { font-size: 0.7rem; color: #444; float: right; }
  </style>
</head>
<body>
  <h1>Ollama Bridge</h1>
  <span id="refresh">auto-refresh every 5s</span>
  <h2>Overview</h2>
  <div class="grid" id="overview"></div>

  <h2>Nodes</h2>
  <table id="nodes-table">
    <thead><tr><th>Node</th><th>Backend</th><th>VRAM</th><th>Avail</th><th>Active</th><th>Latency</th><th>Status</th></tr></thead>
    <tbody id="nodes-body"></tbody>
  </table>

  <h2>Recent Requests</h2>
  <table id="req-table">
    <thead><tr><th>Model</th><th>Node</th><th>Tokens/s</th><th>Latency</th><th>Tokens</th><th>Status</th></tr></thead>
    <tbody id="req-body"></tbody>
  </table>

  <h2>Model Stats</h2>
  <table id="model-table">
    <thead><tr><th>Model</th><th>Requests</th><th>Errors</th><th>Avg Lat (ms)</th><th>Avg tok/s</th></tr></thead>
    <tbody id="model-body"></tbody>
  </table>

  <footer>Ollama Bridge Dashboard — data from /metrics &amp; /nodes</footer>

<script>
async function refresh() {
  const [metrics, nodes] = await Promise.all([
    fetch('/metrics').then(r => r.json()),
    fetch('/nodes').then(r => r.json()),
  ]);

  // Overview cards
  const ov = document.getElementById('overview');
  ov.innerHTML = [
    ['Total Requests', metrics.total_requests, ''],
    ['Errors',         metrics.total_errors,    ''],
    ['Avg tok/s',      metrics.rolling_avg_tokens_per_sec, 'rolling-50'],
    ['Avg Latency',    metrics.rolling_avg_latency_ms,     'ms'],
    ['Uptime',         Math.round(metrics.uptime_seconds / 60), 'min'],
    ['Healthy Nodes',  nodes.filter(n => n.healthy).length, `/ ${nodes.length}`],
  ].map(([label, value, unit]) =>
    `<div class="card"><div class="label">${label}</div><div class="value">${value}</div><div class="unit">${unit}</div></div>`
  ).join('');

  // Nodes table
  const nb = document.getElementById('nodes-body');
  nb.innerHTML = nodes.map(n => `
    <tr>
      <td>${n.label || n.id}</td>
      <td><span class="tag ${n.backend}">${n.backend.toUpperCase()}</span></td>
      <td>${n.vram_gb} GB</td>
      <td>${n.available_vram_gb} GB</td>
      <td>${n.active_requests}</td>
      <td>${n.latency_ms} ms</td>
      <td class="${n.healthy ? 'ok' : 'err'}">${n.healthy ? 'healthy' : 'down'}</td>
    </tr>`).join('');

  // Recent requests
  const reqs = await fetch('/metrics/recent').then(r => r.json());
  const rb = document.getElementById('req-body');
  rb.innerHTML = reqs.map(r => `
    <tr>
      <td>${r.model}</td>
      <td>${r.node}</td>
      <td>${r.tokens_per_sec}</td>
      <td>${r.latency_ms} ms</td>
      <td>${r.output_tokens}</td>
      <td class="${r.success ? 'ok' : 'err'}">${r.success ? 'ok' : r.error || 'err'}</td>
    </tr>`).join('');

  // Model stats
  const mb = document.getElementById('model-body');
  mb.innerHTML = Object.entries(metrics.by_model).map(([model, s]) => `
    <tr>
      <td>${model}</td>
      <td>${s.requests}</td>
      <td class="${s.errors > 0 ? 'err' : ''}">${s.errors}</td>
      <td>${s.avg_latency_ms}</td>
      <td>${s.avg_tokens_per_sec}</td>
    </tr>`).join('');

  document.getElementById('refresh').textContent = 'last updated ' + new Date().toLocaleTimeString();
}

refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""
