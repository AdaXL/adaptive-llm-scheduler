"""
Naive FIFO baseline scheduler.

Requests are served in arrival order, no priority, no preemption,
no KV-cache routing (round-robin worker assignment), no speculative decoding.
Used to demonstrate the advantage of the adaptive scheduler.
"""

import asyncio
import random
from collections import deque
from typing import Deque, List, Optional

from models.request import Request, RequestStatus
from workers.gpu_worker import SimulatedGPUWorker
from metrics.collector import MetricsCollector

ITERATION_INTERVAL_S = 0.001


class FIFOScheduler:
    def __init__(
        self,
        workers: List[SimulatedGPUWorker],
        metrics: MetricsCollector,
    ) -> None:
        self.workers = workers
        self.metrics = metrics
        self._queue: Deque[Request] = deque()
        self._running = False
        self._rr_index = 0  # round-robin worker pointer

    async def submit(self, req: Request) -> asyncio.Event:
        self._queue.append(req)
        return req.completion_event

    async def run(self) -> None:
        self._running = True
        while self._running:
            await self._iteration()
            await asyncio.sleep(ITERATION_INTERVAL_S)

    async def stop(self) -> None:
        self._running = False
        for w in self.workers:
            w.stop()

    async def _iteration(self) -> None:
        while self._queue:
            # Pick next available worker (round-robin)
            worker = self._next_available_worker()
            if worker is None:
                break
            req = self._queue.popleft()
            # Eagerly reserve slot so subsequent loop iterations see correct capacity.
            worker.state.current_batch.append(req.id)
            asyncio.create_task(worker.add_request(req))

    def _next_available_worker(self) -> Optional[SimulatedGPUWorker]:
        n = len(self.workers)
        for _ in range(n):
            w = self.workers[self._rr_index % n]
            self._rr_index += 1
            if w.state.batch_capacity > 0:
                return w
        return None

    @property
    def queue_depth(self) -> int:
        return len(self._queue)
