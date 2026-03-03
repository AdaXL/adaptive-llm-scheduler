import time
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List


@dataclass
class MetricsCollector:
    window_seconds: float = 10.0

    _ttft_samples: List[float] = field(default_factory=list)
    _tbt_samples: List[float] = field(default_factory=list)
    _slo_violations: int = 0
    _slo_total: int = 0
    _kv_hits: int = 0
    _kv_misses: int = 0
    _preemptions: int = 0
    _completed_requests: int = 0
    _token_timestamps: Deque = field(default_factory=lambda: deque(maxlen=100000))

    def record_ttft(self, latency_s: float) -> None:
        self._ttft_samples.append(latency_s)

    def record_tbt(self, latency_s: float) -> None:
        self._tbt_samples.append(latency_s)

    def record_token(self, count: int = 1) -> None:
        now = time.monotonic()
        for _ in range(count):
            self._token_timestamps.append(now)

    def record_slo(self, violated: bool) -> None:
        self._slo_total += 1
        if violated:
            self._slo_violations += 1

    def record_kv_hit(self) -> None:
        self._kv_hits += 1

    def record_kv_miss(self) -> None:
        self._kv_misses += 1

    def record_preemption(self) -> None:
        self._preemptions += 1

    def record_completion(self) -> None:
        self._completed_requests += 1

    # --- Derived properties ---

    @property
    def ttft_p50(self) -> float:
        if not self._ttft_samples:
            return 0.0
        return statistics.median(self._ttft_samples) * 1000  # ms

    @property
    def ttft_p99(self) -> float:
        if not self._ttft_samples:
            return 0.0
        s = sorted(self._ttft_samples)
        idx = max(0, int(len(s) * 0.99) - 1)
        return s[idx] * 1000  # ms

    @property
    def tbt_avg(self) -> float:
        if not self._tbt_samples:
            return 0.0
        return statistics.mean(self._tbt_samples) * 1000  # ms

    @property
    def throughput_tps(self) -> float:
        """Tokens generated in the last window_seconds."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        recent = sum(1 for t in self._token_timestamps if t >= cutoff)
        return recent / self.window_seconds

    @property
    def slo_violation_rate(self) -> float:
        if self._slo_total == 0:
            return 0.0
        return self._slo_violations / self._slo_total

    @property
    def kv_hit_rate(self) -> float:
        total = self._kv_hits + self._kv_misses
        if total == 0:
            return 0.0
        return self._kv_hits / total

    @property
    def preemptions(self) -> int:
        return self._preemptions

    @property
    def completed_requests(self) -> int:
        return self._completed_requests

    def snapshot(self) -> Dict:
        return {
            "throughput_tps": self.throughput_tps,
            "ttft_p50_ms": self.ttft_p50,
            "ttft_p99_ms": self.ttft_p99,
            "tbt_avg_ms": self.tbt_avg,
            "slo_violation_rate": self.slo_violation_rate,
            "kv_hit_rate": self.kv_hit_rate,
            "preemptions": self.preemptions,
            "completed_requests": self.completed_requests,
        }
