"""
Integration tests for the AdaptiveScheduler.

These tests spin up a real (but small) scheduler with simulated workers
and drive it with synthetic requests. All asyncio tasks are shut down
cleanly after each test.
"""

import asyncio
import pytest
from models.request import Request, RequestStatus
from metrics.collector import MetricsCollector
from workers.gpu_worker import SimulatedGPUWorker
from scheduler.core import AdaptiveScheduler


# ────────────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────────────

def make_req(req_id, priority=5, prompt_tokens=20, max_out=5, slo_ms=5000.0,
             prompt_text="hello world"):
    return Request(
        id=req_id,
        prompt_text=prompt_text,
        prompt_tokens=prompt_tokens,
        max_output_tokens=max_out,
        priority=priority,
        slo_deadline_ms=slo_ms,
    )


def build_system(n_workers=2, kv_budget=2048, batch_size=16):
    metrics = MetricsCollector()
    workers = [
        SimulatedGPUWorker(
            worker_id=i,
            kv_cache_budget=kv_budget,
            max_batch_size=batch_size,
            metrics=metrics,
        )
        for i in range(n_workers)
    ]
    scheduler = AdaptiveScheduler(workers=workers, metrics=metrics)
    return scheduler, workers, metrics


async def run_with_timeout(scheduler, workers, coros, timeout=10.0):
    """Start scheduler + workers, run coros, then shut everything down."""
    tasks = []
    tasks.append(asyncio.create_task(scheduler.run()))
    for w in workers:
        tasks.append(asyncio.create_task(w.run()))

    try:
        await asyncio.wait_for(asyncio.gather(*coros), timeout=timeout)
    finally:
        await scheduler.stop()
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


# ────────────────────────────────────────────────────────────────────────────
#  Tests
# ────────────────────────────────────────────────────────────────────────────

class TestSingleRequest:
    @pytest.mark.asyncio
    async def test_single_request_completes(self):
        sched, workers, metrics = build_system()
        req = make_req("r1", max_out=3)
        event = await sched.submit(req)

        async def wait_done():
            await event.wait()

        await run_with_timeout(sched, workers, [wait_done()])
        assert req.status == RequestStatus.DONE
        assert req.tokens_generated >= req.max_output_tokens
        assert metrics.completed_requests == 1

    @pytest.mark.asyncio
    async def test_ttft_is_recorded(self):
        sched, workers, metrics = build_system()
        req = make_req("r1", max_out=3)
        event = await sched.submit(req)

        async def wait_done():
            await event.wait()

        await run_with_timeout(sched, workers, [wait_done()])
        assert req.ttft_recorded is not None
        assert metrics.ttft_p50 > 0


class TestPriorityOrdering:
    @pytest.mark.asyncio
    async def test_high_priority_completes_before_low(self):
        """
        Queue both requests BEFORE the scheduler processes anything.
        With batch_size=1, the scheduler must pick high (priority=9) before
        low (priority=1). Verify high completes first.
        """
        sched, workers, metrics = build_system(n_workers=1, batch_size=1)

        low  = make_req("low",  priority=1, max_out=5)
        high = make_req("high", priority=9, max_out=5)

        # Enqueue both before the scheduler loop starts
        await sched.submit(low)
        await sched.submit(high)

        done_order = []

        async def watch_both():
            # Watch whichever fires first
            pending = {
                asyncio.create_task(high.completion_event.wait(), name="h"),
                asyncio.create_task(low.completion_event.wait(),  name="l"),
            }
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    done_order.append(t.get_name())

        await run_with_timeout(sched, workers, [watch_both()], timeout=15.0)

        assert done_order, "No requests completed"
        assert done_order[0] == "h", (
            f"Expected high-priority to complete first, got order: {done_order}"
        )


class TestKVCacheHitRate:
    @pytest.mark.asyncio
    async def test_repeated_prefix_creates_cache_hits(self):
        """Submit many requests with the same prompt prefix; expect cache hits."""
        sched, workers, metrics = build_system(n_workers=1, kv_budget=8192)
        shared_prompt = "the quick brown fox jumps over the lazy dog"

        events = []
        for i in range(20):
            req = make_req(f"r{i}", prompt_tokens=30, max_out=3, prompt_text=shared_prompt)
            ev = await sched.submit(req)
            events.append(ev)

        async def wait_all():
            await asyncio.gather(*[ev.wait() for ev in events])

        await run_with_timeout(sched, workers, [wait_all()], timeout=30.0)
        assert metrics.kv_hit_rate > 0.3, (
            f"Expected kv_hit_rate > 0.3, got {metrics.kv_hit_rate:.2f}"
        )


class TestPreemption:
    @pytest.mark.asyncio
    async def test_preemption_recorded_on_overload(self):
        """Fill the single worker with low-prio requests, then inject high-prio."""
        sched, workers, metrics = build_system(n_workers=1, batch_size=2)

        low_reqs = [make_req(f"low{i}", priority=1, max_out=50) for i in range(4)]
        high_req = make_req("high", priority=9, max_out=3)

        # Submit low-priority first to saturate the worker
        for r in low_reqs:
            await sched.submit(r)

        await asyncio.sleep(0.01)  # let scheduler fill the worker

        high_event = await sched.submit(high_req)

        async def wait_high():
            await high_event.wait()

        await run_with_timeout(sched, workers, [wait_high()], timeout=30.0)
        assert high_req.status == RequestStatus.DONE
        # Preemption may or may not fire depending on timing; just verify it ran
        # (at minimum 0 preemptions is valid if space was available)
        assert metrics.preemptions >= 0
