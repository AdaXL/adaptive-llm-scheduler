"""
AdaptiveScheduler — main scheduling loop.

Runs as a single asyncio Task and coordinates:
  1. SLO urgency bumps  — elevate priority of near-deadline requests.
  2. Preemption         — evict low-priority running requests when a
                          high-priority request is queued and all workers full.
  3. Routing & inject   — use KV-cache-aware routing to place requests
                          onto the best available worker.
  4. Continuous batching— workers independently drain their prefill/decode
                          queues; the scheduler only injects new work.

All shared state lives in the single asyncio event loop thread — no locks needed.
"""

import asyncio
import time
from typing import Dict, List, Optional

from models.request import Request, RequestStatus
from models.worker import WorkerState
from scheduler.priority_queue import PriorityRequestQueue
from scheduler.routing import route_request
from workers.gpu_worker import SimulatedGPUWorker
from metrics.collector import MetricsCollector

# How close to the deadline (seconds) before we urgency-bump priority
SLO_URGENCY_WINDOW_S = 0.050   # 50 ms
PRIORITY_BUMP = 2
MAX_PRIORITY = 9

# Minimum priority gap required to justify preempting a running request.
# Prevents low-value thrashing (e.g. priority 6 preempting priority 5).
MIN_PREEMPT_PRIORITY_GAP = 3

# Scheduler iteration cadence
ITERATION_INTERVAL_S = 0.001   # 1 ms


class AdaptiveScheduler:
    def __init__(
        self,
        workers: List[SimulatedGPUWorker],
        metrics: MetricsCollector,
        iteration_interval_s: float = ITERATION_INTERVAL_S,
    ) -> None:
        self.workers = workers
        self.metrics = metrics
        self.iteration_interval = iteration_interval_s
        self.queue = PriorityRequestQueue()
        self._running = False
        self._queue_depth: int = 0  # for display

    # ---------------------------------------------------------------------- #
    #  Public API                                                              #
    # ---------------------------------------------------------------------- #

    async def submit(self, req: Request) -> asyncio.Event:
        """
        Enqueue a request for scheduling.
        Returns the request's completion_event so callers can await it.
        """
        self.queue.push(req)
        self._queue_depth = len(self.queue)
        return req.completion_event

    # ---------------------------------------------------------------------- #
    #  Main loop                                                               #
    # ---------------------------------------------------------------------- #

    async def run(self) -> None:
        self._running = True
        while self._running:
            await self._scheduling_iteration()
            await asyncio.sleep(self.iteration_interval)

    async def stop(self) -> None:
        self._running = False
        for w in self.workers:
            w.stop()

    # ---------------------------------------------------------------------- #
    #  Scheduling iteration                                                    #
    # ---------------------------------------------------------------------- #

    async def _scheduling_iteration(self) -> None:
        # Step 1: SLO urgency bumps
        self._bump_urgent_requests()

        # Step 2: Preemption — one victim per iteration to avoid thrashing
        top = self.queue.peek()
        if top is not None:
            await self._check_preemption(top)

        # Step 3: Route and inject new requests into workers.
        # We walk the priority-ordered list rather than the heap directly, so that
        # an infeasible head-of-queue request (e.g. prompt > kv_budget) does not
        # block lower-priority requests that COULD be routed.
        worker_state_list = [w.state for w in self.workers]
        all_at_capacity = all(w.batch_capacity <= 0 for w in worker_state_list)
        if not all_at_capacity:
            for req in self.queue.to_list():
                target_state = route_request(req, worker_state_list)
                if target_state is None:
                    # This specific request can't fit anywhere right now; try next.
                    continue
                target_worker = self._worker_by_id(target_state.worker_id)
                if target_worker is None:
                    continue
                self.queue.remove(req)
                # Eagerly reserve the batch slot so routing sees updated capacity
                # within the same loop before add_request() actually runs.
                target_state.current_batch.append(req.id)
                asyncio.create_task(target_worker.add_request(req))
                # Re-check capacity after each dispatch
                if all(w.batch_capacity <= 0 for w in worker_state_list):
                    break

        self._queue_depth = len(self.queue)

    # ---------------------------------------------------------------------- #
    #  SLO urgency bumping                                                     #
    # ---------------------------------------------------------------------- #

    def _bump_urgent_requests(self) -> None:
        now = time.monotonic()
        for req in self.queue.to_list():
            time_to_deadline = req.deadline_absolute - now
            if time_to_deadline < SLO_URGENCY_WINDOW_S and req.priority < MAX_PRIORITY:
                new_prio = min(req.priority + PRIORITY_BUMP, MAX_PRIORITY)
                self.queue.bump_priority(req, new_prio)

    # ---------------------------------------------------------------------- #
    #  Preemption                                                              #
    # ---------------------------------------------------------------------- #

    async def _check_preemption(self, high_prio_req: Request) -> None:
        """
        If high_prio_req can't fit anywhere AND there is a running request
        with lower priority, preempt the lowest-priority victim.
        """
        worker_states = [w.state for w in self.workers]
        if any(w.batch_capacity > 0 for w in worker_states):
            return  # at least one worker has room; no need to preempt

        victim_req, victim_worker = self._find_preemption_candidate()
        if victim_req is None or victim_worker is None:
            return
        if high_prio_req.priority - victim_req.priority < MIN_PREEMPT_PRIORITY_GAP:
            return  # gap too small — not worth the thrash cost

        await victim_worker.preempt_request(victim_req)
        # Remove from in-flight tracking — it's being evicted cooperatively.
        # The re-queue happens after yield so the worker has had a cycle to evict.
        self._in_flight_ids.discard(victim_req.id)
        victim_req.status = RequestStatus.QUEUED   # reset so it can be re-dispatched
        self.queue.push(victim_req)

    def _find_preemption_candidate(self):
        """Return (Request, SimulatedGPUWorker) with the lowest-priority running request."""
        best_req: Optional[Request] = None
        best_worker: Optional[SimulatedGPUWorker] = None

        for worker in self.workers:
            for req in worker._active.values():
                if req.status not in (RequestStatus.PREFILLING, RequestStatus.DECODING):
                    continue
                if best_req is None or req.priority < best_req.priority:
                    best_req = req
                    best_worker = worker
                elif req.priority == best_req.priority:
                    # Among equal priority pick the one with the latest deadline
                    if req.deadline_absolute > best_req.deadline_absolute:
                        best_req = req
                        best_worker = worker

        return best_req, best_worker

    # ---------------------------------------------------------------------- #
    #  Helpers                                                                 #
    # ---------------------------------------------------------------------- #

    def _worker_by_id(self, worker_id: int) -> Optional[SimulatedGPUWorker]:
        for w in self.workers:
            if w.state.worker_id == worker_id:
                return w
        return None

    @property
    def queue_depth(self) -> int:
        return self._queue_depth
