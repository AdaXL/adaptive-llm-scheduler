import heapq
from typing import Dict, List, Optional, Tuple

from models.request import Request


class PriorityRequestQueue:
    """
    Min-heap priority queue for LLM requests.
    Ordering: highest priority first; ties broken by earliest SLO deadline.

    Uses generation-based lazy deletion: every push() assigns a new generation
    number to the request ID. Heap entries with an older generation are skipped
    on pop/peek, making stale duplicate entries harmless.

    This correctly handles bump_priority() (re-push invalidates the old entry)
    and re-enqueuing of preempted requests.
    """

    def __init__(self) -> None:
        self._heap: List[Tuple] = []
        self._seq: int = 0                      # global sequence (tiebreaker)
        self._generation: Dict[str, int] = {}   # req_id → current valid generation

    def _is_valid(self, req_id: str, gen: int) -> bool:
        return self._generation.get(req_id, -1) == gen

    def push(self, req: Request) -> None:
        """Push request; any previous heap entry for this ID becomes stale."""
        gen = self._generation.get(req.id, 0) + 1
        self._generation[req.id] = gen
        key = (-req.priority, req.slo_deadline_ms, self._seq, req.id, gen, req)
        self._seq += 1
        heapq.heappush(self._heap, key)

    def pop(self) -> Optional[Request]:
        while self._heap:
            _, _, _, rid, gen, req = self._heap[0]
            if not self._is_valid(rid, gen):
                heapq.heappop(self._heap)
                continue
            heapq.heappop(self._heap)
            del self._generation[rid]
            return req
        return None

    def peek(self) -> Optional[Request]:
        while self._heap:
            _, _, _, rid, gen, req = self._heap[0]
            if not self._is_valid(rid, gen):
                heapq.heappop(self._heap)
                continue
            return req
        return None

    def remove(self, req: Request) -> None:
        """Invalidate all heap entries for this request (lazy deletion)."""
        self._generation.pop(req.id, None)

    def bump_priority(self, req: Request, new_priority: int) -> None:
        """Re-insert with updated priority; old entry auto-invalidated by generation bump."""
        req.priority = new_priority
        self.push(req)  # increments generation, stales the old entry

    def to_list(self) -> List[Request]:
        """Return all valid pending requests in priority order (non-destructive)."""
        result = []
        for _, _, _, rid, gen, req in self._heap:
            if self._is_valid(rid, gen):
                result.append(req)
        result.sort(key=lambda r: (-r.priority, r.slo_deadline_ms))
        return result

    def __len__(self) -> int:
        return len(self._generation)

    def __bool__(self) -> bool:
        return bool(self._generation)
