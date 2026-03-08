"""
Poisson load generator.

Simulates realistic LLM API traffic:
  - Inter-arrival times follow an exponential distribution (Poisson process).
  - Prompt lengths follow a log-normal distribution (realistic heavy tail).
  - Priority distribution is skewed toward low values with a few urgent requests.
  - Three traffic scenarios: steady, burst, mixed_priority.

The small prompt vocabulary (1 000 distinct prompts) deliberately creates
KV-cache hit opportunities, mirroring real-world repeated/similar queries.
"""

import asyncio
import random
import time
import uuid
from typing import Callable, Awaitable

from models.request import Request, RequestStatus

# Prompt vocabulary — small so workers see repeated prefixes (cache hits)
PROMPT_VOCAB_SIZE = 200

SCENARIO_CONFIGS = {
    "steady": {
        "base_rps": None,       # use --rps arg
        "burst_rps": None,
        "burst_start_frac": 0,
        "burst_duration_s": 0,
    },
    "burst": {
        "base_rps": 5,
        "burst_rps": 200,
        "burst_start_frac": 0.33,   # burst begins at 1/3 of total duration
        "burst_duration_s": 5.0,
    },
    "mixed_priority": {
        "base_rps": None,
        "burst_rps": 80,
        "burst_start_frac": 0.25,
        "burst_duration_s": 10.0,
    },
}

# Priority → SLO deadline mapping
_SLO_BY_PRIORITY = {
    range(0, 4): 5000.0,    # low priority: 5 s deadline
    range(4, 7): 2000.0,    # medium: 2 s
    range(7, 10): 500.0,    # high: 500 ms
}


def _slo_for_priority(priority: int) -> float:
    for r, slo in _SLO_BY_PRIORITY.items():
        if priority in r:
            return slo
    return 5000.0


class LoadGenerator:
    def __init__(
        self,
        submit_fn: Callable[[Request], Awaitable],
        rps: float,
        scenario: str,
        duration_s: float,
        seed: int = 42,
    ) -> None:
        self.submit_fn = submit_fn
        self.rps = rps
        self.scenario = scenario
        self.duration_s = duration_s
        self.rng = random.Random(seed)
        self._prompts = self._build_prompt_vocab()
        self.total_submitted = 0

    def _build_prompt_vocab(self) -> list:
        """Generate a small vocabulary of synthetic prompts for cache-hit testing."""
        rng = random.Random(0)
        words = ["the", "a", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
                 "llm", "model", "inference", "request", "token", "batch", "gpu",
                 "memory", "cache", "schedule", "latency", "throughput", "priority"]
        return [
            " ".join(rng.choices(words, k=rng.randint(5, 20)))
            for _ in range(PROMPT_VOCAB_SIZE)
        ]

    def _make_request(self) -> Request:
        # Prompt tokens: log-normal, skewed toward short, with heavy tail
        prompt_tokens = int(self.rng.lognormvariate(mu=5.5, sigma=1.0))
        # Cap at 1500 to stay within the default kv_cache_budget (2048).
        # Prompts larger than the budget can never be served and would block the queue.
        prompt_tokens = max(10, min(prompt_tokens, 1500))

        # Priority: mostly low, some medium, few high
        priority = self.rng.choices(
            range(10),
            weights=[12, 12, 10, 10, 10, 10, 8, 5, 2, 1],
        )[0]

        slo_deadline_ms = _slo_for_priority(priority)
        max_output_tokens = self.rng.randint(20, 256)

        # Reuse one of a small set of prompts to generate KV-cache hits
        prompt_text = self.rng.choice(self._prompts)

        return Request(
            id=str(uuid.uuid4()),
            prompt_text=prompt_text,
            prompt_tokens=prompt_tokens,
            max_output_tokens=max_output_tokens,
            priority=priority,
            slo_deadline_ms=slo_deadline_ms,
        )

    async def run(self) -> None:
        cfg = SCENARIO_CONFIGS[self.scenario]
        start = time.monotonic()

        base_rps = cfg["base_rps"] or self.rps
        burst_rps = cfg["burst_rps"] or self.rps
        burst_start = self.duration_s * cfg["burst_start_frac"]
        burst_end = burst_start + cfg["burst_duration_s"]

        while True:
            elapsed = time.monotonic() - start
            if elapsed >= self.duration_s:
                break

            # Determine current effective RPS
            in_burst = burst_start < elapsed < burst_end
            effective_rps = burst_rps if in_burst else base_rps

            # Poisson inter-arrival: exponential distribution
            inter_arrival = self.rng.expovariate(effective_rps)
            await asyncio.sleep(inter_arrival)

            req = self._make_request()
            await self.submit_fn(req)
            self.total_submitted += 1
