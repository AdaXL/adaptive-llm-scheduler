import asyncio
import random
import pytest
from scheduler.speculative import (
    run_speculative_decode,
    should_use_speculative,
    DRAFT_K,
    ACCEPT_PROB,
    ALPHA,
    EXPECTED_SPEEDUP,
)
from models.request import Request, RequestStatus
from models.worker import WorkerState


def make_req(priority=5, remaining=50, slo_ms=2000.0):
    import time
    r = Request(
        id="r_spec",
        prompt_text="test prompt",
        prompt_tokens=100,
        max_output_tokens=100,
        priority=priority,
        slo_deadline_ms=slo_ms,
        arrival_time=time.monotonic(),
    )
    r.tokens_generated = 100 - remaining
    return r


def make_worker(utilization=0.5):
    w = WorkerState(worker_id=0, kv_cache_budget=2048, max_batch_size=32)
    w.utilization = utilization
    return w


class TestShouldUseSpeculative:
    def test_fires_under_normal_conditions(self):
        req = make_req(priority=5, remaining=20)
        w = make_worker(utilization=0.5)
        assert should_use_speculative(req, w)

    def test_disabled_when_too_few_remaining_tokens(self):
        req = make_req(remaining=DRAFT_K - 1)
        w = make_worker()
        assert not should_use_speculative(req, w)

    def test_disabled_when_worker_overloaded(self):
        req = make_req(remaining=20)
        w = make_worker(utilization=0.85)
        assert not should_use_speculative(req, w)

    def test_disabled_for_high_priority(self):
        req = make_req(priority=8, remaining=20)
        w = make_worker(utilization=0.3)
        assert not should_use_speculative(req, w)

    def test_disabled_near_deadline(self):
        import time
        req = make_req(slo_ms=10.0, remaining=20)   # 10ms deadline → nearly expired
        req.arrival_time = time.monotonic() - 0.009  # elapsed 9ms
        w = make_worker()
        assert not should_use_speculative(req, w)


class TestRunSpeculativeDecode:
    @pytest.mark.asyncio
    async def test_accepted_tokens_in_range(self):
        rng = random.Random(0)
        result = await run_speculative_decode(0.0, 0.0, rng=rng)
        assert 0 <= result.accepted_tokens <= DRAFT_K

    @pytest.mark.asyncio
    async def test_rejected_at_minus_one_means_all_accepted(self):
        # Use a deterministic RNG that always accepts
        class AlwaysAccept:
            def random(self):
                return 0.0  # < ACCEPT_PROB always
        result = await run_speculative_decode(0.0, 0.0, rng=AlwaysAccept())
        assert result.accepted_tokens == DRAFT_K
        assert result.rejected_at == -1

    @pytest.mark.asyncio
    async def test_rejection_at_index_zero(self):
        class AlwaysReject:
            def random(self):
                return 1.0  # > ACCEPT_PROB always
        result = await run_speculative_decode(0.0, 0.0, rng=AlwaysReject())
        assert result.accepted_tokens == 0
        assert result.rejected_at == 0

    @pytest.mark.asyncio
    async def test_statistical_acceptance_rate(self):
        """Over many runs, mean accepted matches the stop-at-first-rejection expectation.

        Because we stop sampling at the first rejection, the expected value is:
          E[X] = sum_{k=0}^{K-1} k * p^k * (1-p) + K * p^K  ≈ 2.71 for K=4, p=0.85
        (Not K*p=3.40, which would apply to K independent trials without stopping.)
        """
        p, K = ACCEPT_PROB, DRAFT_K
        # Analytical expectation for stop-at-first-rejection
        expected = sum(k * p**k * (1 - p) for k in range(K)) + K * p**K

        rng = random.Random(42)
        samples = []
        for _ in range(1000):
            result = await run_speculative_decode(0.0, 0.0, rng=rng)
            samples.append(result.accepted_tokens)
        mean = sum(samples) / len(samples)
        assert abs(mean - expected) < 0.3, f"Expected ≈{expected:.2f}, got {mean:.2f}"

    def test_expected_speedup_formula(self):
        computed = (1 + DRAFT_K * ACCEPT_PROB) / (1 + ALPHA)
        assert abs(EXPECTED_SPEEDUP - computed) < 1e-6
