import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class RequestStatus(Enum):
    QUEUED = auto()
    PREFILLING = auto()
    DECODING = auto()
    DONE = auto()
    PREEMPTED = auto()


@dataclass
class Request:
    id: str
    prompt_text: str
    prompt_tokens: int
    max_output_tokens: int
    priority: int               # 0 (lowest) – 9 (highest)
    slo_deadline_ms: float
    arrival_time: float = field(default_factory=time.monotonic)
    status: RequestStatus = field(default=RequestStatus.QUEUED)

    # Mutable decode progress state
    tokens_generated: int = 0
    partial_kv_saved: bool = False
    ttft_recorded: Optional[float] = None
    completion_event: asyncio.Event = field(default_factory=asyncio.Event)

    def __lt__(self, other: "Request") -> bool:
        # Higher priority = lower heap key; equal priority → earlier deadline first
        return (-self.priority, self.slo_deadline_ms) < (-other.priority, other.slo_deadline_ms)

    def __le__(self, other: "Request") -> bool:
        return (-self.priority, self.slo_deadline_ms) <= (-other.priority, other.slo_deadline_ms)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Request):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    @property
    def deadline_absolute(self) -> float:
        return self.arrival_time + self.slo_deadline_ms / 1000.0

    @property
    def is_slo_violated(self) -> bool:
        return time.monotonic() > self.deadline_absolute

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.max_output_tokens - self.tokens_generated)
