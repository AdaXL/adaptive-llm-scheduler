import time
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class KVCacheSlot:
    prefix_hash: int
    token_count: int
    last_access: float = field(default_factory=time.monotonic)


@dataclass
class WorkerState:
    worker_id: int
    kv_cache_budget: int        # total token slots available
    max_batch_size: int = 32

    current_batch: List[str] = field(default_factory=list)     # request IDs
    kv_cache: Dict[int, KVCacheSlot] = field(default_factory=dict)
    utilization: float = 0.0    # 0.0–1.0
    busy: bool = False

    @property
    def kv_cache_used(self) -> int:
        return sum(s.token_count for s in self.kv_cache.values())

    @property
    def kv_cache_free(self) -> int:
        return self.kv_cache_budget - self.kv_cache_used

    @property
    def batch_capacity(self) -> int:
        return self.max_batch_size - len(self.current_batch)

    def prefix_overlap_score(self, prefix_hash: int) -> int:
        """Return cached token count if hash present (cache hit), else 0."""
        if prefix_hash in self.kv_cache:
            self.kv_cache[prefix_hash].last_access = time.monotonic()
            return self.kv_cache[prefix_hash].token_count
        return 0

    def evict_lru(self, needed_slots: int) -> None:
        """Evict least-recently-used entries until needed_slots are free."""
        while self.kv_cache_free < needed_slots and self.kv_cache:
            lru_hash = min(self.kv_cache, key=lambda h: self.kv_cache[h].last_access)
            del self.kv_cache[lru_hash]

    def store_kv(self, prefix_hash: int, token_count: int) -> None:
        """Store a KV cache entry, evicting LRU entries if needed."""
        if token_count > self.kv_cache_budget:
            return  # single entry too large to cache
        if self.kv_cache_free < token_count:
            self.evict_lru(token_count)
        self.kv_cache[prefix_hash] = KVCacheSlot(
            prefix_hash=prefix_hash,
            token_count=token_count,
        )

    def remove_from_batch(self, request_id: str) -> None:
        if request_id in self.current_batch:
            self.current_batch.remove(request_id)
