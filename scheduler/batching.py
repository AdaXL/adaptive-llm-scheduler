"""
Continuous batching logic (Orca-style, iteration-level scheduling).

Key idea: after each decode step a worker can accept NEW requests without
waiting for the existing batch to finish. select_batch() greedily fills
available slots by priority order, checking KV-cache and batch capacity.
"""

from typing import List

from models.request import Request
from models.worker import WorkerState
from scheduler.priority_queue import PriorityRequestQueue

PREFIX_HASH_CHARS = 256  # characters used to compute prefix hash


def compute_prefix_hash(prompt_text: str) -> int:
    return hash(prompt_text[:PREFIX_HASH_CHARS])


def can_fit_request(req: Request, worker: WorkerState) -> bool:
    """
    Check whether req can be added to worker's batch this iteration.
    Accounts for KV-cache hits: if prefix is cached, we skip those tokens.
    """
    if worker.batch_capacity <= 0:
        return False
    prefix_hash = compute_prefix_hash(req.prompt_text)
    cached_tokens = worker.prefix_overlap_score(prefix_hash)
    needed_kv = max(0, req.prompt_tokens - cached_tokens)
    return worker.kv_cache_free >= needed_kv


def select_batch(
    queue: PriorityRequestQueue,
    worker: WorkerState,
    max_new: int = 8,
) -> List[Request]:
    """
    Greedy continuous-batch selector.

    Walks the priority-ordered queue and picks up to `max_new` requests that
    fit in this worker's current capacity. Selected requests are removed from
    the queue.

    Args:
        queue:   Shared priority queue of pending requests.
        worker:  Target worker whose current batch / KV state we check against.
        max_new: Maximum number of new requests to inject per iteration.

    Returns:
        List of requests to inject into this worker (may be empty).
    """
    candidates = queue.to_list()
    selected: List[Request] = []

    for req in candidates:
        if len(selected) >= max_new:
            break
        if can_fit_request(req, worker):
            selected.append(req)

    for req in selected:
        queue.remove(req)

    return selected
