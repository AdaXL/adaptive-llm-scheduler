"""
Speculative decoding simulation.

A small "draft" model proposes K tokens cheaply; the large "verifier" model
checks them in one forward pass. Accepted tokens are free throughput gains.

Reference: "Speculative Decoding" (Chen et al. 2023) / "Medusa" (Cai et al. 2024).
"""

import asyncio
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.request import Request
    from models.worker import WorkerState

DRAFT_K: int = 4            # tokens proposed by draft model per step
ACCEPT_PROB: float = 0.85   # per-token acceptance probability (rejection sampling)
ALPHA: float = 0.1          # draft model cost as fraction of large model cost

# Expected speedup: (1 + K * p) / (1 + alpha) = (1 + 4*0.85) / (1 + 0.1) ≈ 4.0x
EXPECTED_SPEEDUP: float = (1 + DRAFT_K * ACCEPT_PROB) / (1 + ALPHA)


@dataclass
class SpecResult:
    accepted_tokens: int       # how many draft tokens were accepted (0..K)
    rejected_at: int           # index of first rejection (-1 = all accepted)
    speedup: float             # realised speedup vs. purely autoregressive
    draft_time_s: float
    verify_time_s: float


def should_use_speculative(req: "Request", worker: "WorkerState") -> bool:
    """
    Activate speculative decoding when it is beneficial:
      - Enough remaining tokens to amortise draft overhead.
      - Worker not overloaded (draft adds batch pressure).
      - Request not dangerously close to its SLO deadline.
      - Medium/low priority (reserve headroom for high-prio urgent requests).
    """
    import time as _time
    time_to_deadline = req.deadline_absolute - _time.monotonic()
    return (
        req.remaining_tokens >= DRAFT_K
        and worker.utilization < 0.8
        and time_to_deadline > 0.050        # at least 50ms buffer
        and req.priority <= 6
    )


async def run_speculative_decode(
    draft_latency_per_token_s: float,
    verify_latency_s: float,
    rng: random.Random = random,
) -> SpecResult:
    """
    Simulate one speculative decoding step.

    Phase 1 – Draft:  sleep for K * draft_latency (cheap forward passes).
    Phase 2 – Sample: rejection-sampling loop to decide accepted tokens.
    Phase 3 – Verify: sleep for verify_latency (one large-model forward pass).

    Returns SpecResult with accepted token count and realised speedup.
    """
    # Phase 1: draft model generates K candidate tokens
    draft_time = DRAFT_K * draft_latency_per_token_s
    await asyncio.sleep(draft_time)

    # Phase 2: rejection sampling
    accepted = 0
    rejected_at = -1
    for i in range(DRAFT_K):
        if rng.random() < ACCEPT_PROB:
            accepted += 1
        else:
            rejected_at = i
            break

    # Phase 3: large model verifies in one forward pass
    verify_start = time.monotonic()
    await asyncio.sleep(verify_latency_s)
    verify_time = time.monotonic() - verify_start

    # Realised speedup relative to purely auto-regressive baseline
    # Baseline would have taken `accepted` sequential decode steps
    total_time = draft_time + verify_time
    baseline_time = accepted * verify_latency_s if accepted > 0 else verify_latency_s
    realised_speedup = baseline_time / total_time if total_time > 0 else 1.0

    return SpecResult(
        accepted_tokens=accepted,
        rejected_at=rejected_at,
        speedup=realised_speedup,
        draft_time_s=draft_time,
        verify_time_s=verify_time,
    )
