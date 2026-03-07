"""
Simulated GPU worker.

Each SimulatedGPUWorker runs as an independent asyncio Task. It processes
requests through two phases:

  1. Prefill  — prompt tokens are processed, KV cache is populated.
               Time: prefill_latency(prompt_tokens, batch_size) seconds.
               On KV-cache hit: only process the uncached portion.

  2. Decode   — one token is generated per forward pass.
               Time: decode_latency(batch_size) seconds.
               Speculative decoding can generate multiple tokens per step.

The worker cooperates with the scheduler via:
  - add_request()    : scheduler injects a new request into this worker.
  - preempt_request(): scheduler signals that a request must be evicted.
  - completion_event : set on each Request when it finishes.
"""

import asyncio
import random
import time
from typing import Dict, Set

from models.request import Request, RequestStatus
from models.worker import WorkerState
from metrics.collector import MetricsCollector
from scheduler.batching import compute_prefix_hash
from scheduler.speculative import should_use_speculative, DRAFT_K, ACCEPT_PROB


# --------------------------------------------------------------------------- #
#  Latency formulas                                                            #
# --------------------------------------------------------------------------- #

def prefill_latency(prompt_tokens: int, batch_size: int) -> float:
    """Simulated prefill time in seconds.

    Formula: 0.05ms * prompt_tokens / batch_size + 0.2ms
    Scaled 10x faster than real hardware to make demos runnable in seconds.
    """
    return (0.00005 * prompt_tokens / max(batch_size, 1)) + 0.0002


def decode_latency(batch_size: int) -> float:
    """Simulated decode time per step in seconds.

    Formula: 0.1ms * batch_size + 0.05ms
    Scaled 10x faster than real hardware.
    """
    return 0.0001 * batch_size + 0.00005


DRAFT_LATENCY_PER_TOKEN = 0.00001   # 0.01 ms per speculative draft token


# --------------------------------------------------------------------------- #
#  Worker                                                                      #
# --------------------------------------------------------------------------- #

class SimulatedGPUWorker:
    def __init__(
        self,
        worker_id: int,
        kv_cache_budget: int,
        max_batch_size: int,
        metrics: MetricsCollector,
    ) -> None:
        self.state = WorkerState(
            worker_id=worker_id,
            kv_cache_budget=kv_cache_budget,
            max_batch_size=max_batch_size,
        )
        self.metrics = metrics
        self._active: Dict[str, Request] = {}   # id → Request (PREFILLING|DECODING)
        self._preempt_set: Set[str] = set()     # IDs to preempt before next decode
        self._stop = asyncio.Event()
        self._has_work = asyncio.Event()         # wakes worker when new request arrives

    # ---------------------------------------------------------------------- #
    #  Public API (called by scheduler)                                        #
    # ---------------------------------------------------------------------- #

    async def add_request(self, req: Request) -> None:
        """Inject a new request into this worker's active set.

        The scheduler eagerly appends req.id to current_batch as a slot
        reservation before this coroutine runs. We guard against double-active
        state in case a preempted request is re-dispatched before the worker
        finishes evicting it (cooperative preemption may lag by one cycle).
        """
        if req.id in self._active:
            return  # already active; ignore duplicate dispatch
        req.status = RequestStatus.PREFILLING
        self._active[req.id] = req
        if req.id not in self.state.current_batch:
            self.state.current_batch.append(req.id)
        self._has_work.set()

    async def preempt_request(self, req: Request) -> None:
        """Schedule req for cooperative preemption before the next decode step."""
        self._preempt_set.add(req.id)

    def stop(self) -> None:
        self._stop.set()

    # ---------------------------------------------------------------------- #
    #  Main worker loop                                                        #
    # ---------------------------------------------------------------------- #

    async def run(self) -> None:
        while not self._stop.is_set():
            # Wait until there is at least one active request
            if not self._active:
                self._has_work.clear()
                try:
                    await asyncio.wait_for(self._has_work.wait(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

            # --- Phase 1: Prefill all newly arrived requests ---
            prefilling = [r for r in self._active.values()
                          if r.status == RequestStatus.PREFILLING]
            for req in prefilling:
                await self._prefill(req)

            # --- Cooperative preemption before decode ---
            self._apply_preemptions()

            # --- Phase 2: One decode step for all DECODING requests ---
            decoding = [r for r in self._active.values()
                        if r.status == RequestStatus.DECODING]
            if decoding:
                await self._decode_step(decoding)

            self._update_utilization()

    # ---------------------------------------------------------------------- #
    #  Prefill phase                                                           #
    # ---------------------------------------------------------------------- #

    async def _prefill(self, req: Request) -> None:
        prefix_hash = compute_prefix_hash(req.prompt_text)
        cached = self.state.prefix_overlap_score(prefix_hash)

        if cached > 0:
            self.metrics.record_kv_hit()
            effective_tokens = max(0, req.prompt_tokens - cached)
        else:
            self.metrics.record_kv_miss()
            effective_tokens = req.prompt_tokens

        batch_size = len(self._active)
        lat = prefill_latency(effective_tokens, batch_size)
        await asyncio.sleep(lat)

        # Record TTFT
        now = time.monotonic()
        req.ttft_recorded = now
        self.metrics.record_ttft(now - req.arrival_time)

        # Store KV cache entry
        self.state.store_kv(prefix_hash, req.prompt_tokens)

        req.status = RequestStatus.DECODING

    # ---------------------------------------------------------------------- #
    #  Decode phase                                                            #
    # ---------------------------------------------------------------------- #

    async def _decode_step(self, decoding_batch: list) -> None:
        """
        One decode step for the entire batch.

        Speculative decode is applied as a batch-level optimization, not per-request:
        - ONE draft phase for the whole batch (same latency regardless of batch size).
        - ONE acceptance sampling per request (synchronous, no sleep).
        - ONE large-model verify pass (one sleep).

        This correctly models real speculative decoding hardware behaviour where the
        draft + verify runs over the whole batch in a single forward pass.
        """
        batch_size = len(decoding_batch)
        dec_lat = decode_latency(batch_size)
        draft_lat = DRAFT_LATENCY_PER_TOKEN

        spec_reqs = [r for r in decoding_batch if should_use_speculative(r, self.state)]
        normal_reqs = [r for r in decoding_batch if r not in spec_reqs]

        if spec_reqs:
            # Batch speculative decode: one draft phase, one verify phase.
            # Acceptance sampling is run synchronously (no sleep) for each request.
            draft_time = DRAFT_K * draft_lat  # single shared draft phase
            await asyncio.sleep(draft_time)

            spec_tokens: dict = {}
            for req in spec_reqs:
                accepted = 0
                for _ in range(DRAFT_K):
                    if random.random() < ACCEPT_PROB:
                        accepted += 1
                    else:
                        break
                spec_tokens[req.id] = max(1, accepted)

            await asyncio.sleep(dec_lat)   # verify pass (same cost as normal decode)
            for req in spec_reqs:
                tokens = spec_tokens[req.id]
                req.tokens_generated += tokens
                self.metrics.record_token(tokens)
            self.metrics.record_tbt(dec_lat)

        # Normal decode path (all at once — one forward pass)
        if normal_reqs:
            await asyncio.sleep(dec_lat)
            for req in normal_reqs:
                req.tokens_generated += 1
                self.metrics.record_token()
            self.metrics.record_tbt(dec_lat)

        # Check completions
        for req in list(decoding_batch):
            if req.tokens_generated >= req.max_output_tokens:
                req.status = RequestStatus.DONE
                self.metrics.record_slo(req.is_slo_violated)
                self.metrics.record_completion()
                req.completion_event.set()
                self._evict(req)

    # ---------------------------------------------------------------------- #
    #  Helpers                                                                 #
    # ---------------------------------------------------------------------- #

    def _apply_preemptions(self) -> None:
        for rid in list(self._preempt_set):
            req = self._active.get(rid)
            if req is not None:
                req.status = RequestStatus.PREEMPTED
                req.partial_kv_saved = True
                self._evict(req)
                self.metrics.record_preemption()
        self._preempt_set.clear()

    def _evict(self, req: Request) -> None:
        self._active.pop(req.id, None)
        self.state.remove_from_batch(req.id)

    def _update_utilization(self) -> None:
        used = self.state.kv_cache_used
        self.state.utilization = used / max(self.state.kv_cache_budget, 1)
        self.state.busy = bool(self._active)
