"""
KV-cache aware routing.

Route each incoming request to the worker most likely to have its prompt
prefix already cached, avoiding a full re-prefill. Falls back to load
balancing when no cache hit is available.
"""

from typing import List, Optional

from models.request import Request
from models.worker import WorkerState
from scheduler.batching import compute_prefix_hash


def _worker_score(worker: WorkerState, prefix_hash: int) -> float:
    """
    Higher score = better target worker.

    Components:
      +cache_hit_tokens * 10   → strongly prefer workers with the prefix cached
      -utilization * 100       → penalise overloaded workers
      +kv_cache_free * 0.1     → mild preference for workers with free KV space
      +batch_capacity * 0.5    → mild preference for workers with free batch slots
    """
    cache_hit = worker.prefix_overlap_score(prefix_hash)
    return (
        cache_hit * 10.0
        - worker.utilization * 100.0
        + worker.kv_cache_free * 0.1
        + worker.batch_capacity * 0.5
    )


def route_request(
    req: Request,
    workers: List[WorkerState],
) -> Optional[WorkerState]:
    """
    Select the best worker for `req`.

    Returns None if all workers are at batch capacity or lack KV headroom.
    """
    prefix_hash = compute_prefix_hash(req.prompt_text)
    best_worker: Optional[WorkerState] = None
    best_score: float = float("-inf")

    for worker in workers:
        if worker.batch_capacity <= 0:
            continue
        # Only hard-block if the request's prompt alone exceeds the entire KV budget.
        # Otherwise, store_kv() handles eviction automatically via LRU.
        if req.prompt_tokens > worker.kv_cache_budget:
            continue
        score = _worker_score(worker, prefix_hash)
        if score > best_score:
            best_score = score
            best_worker = worker

    return best_worker
