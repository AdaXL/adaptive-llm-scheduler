import time
import pytest
from models.request import Request, RequestStatus
from models.worker import WorkerState


def make_req(priority=5, slo_ms=2000.0, req_id="r1", prompt_tokens=100):
    return Request(
        id=req_id,
        prompt_text="hello world",
        prompt_tokens=prompt_tokens,
        max_output_tokens=50,
        priority=priority,
        slo_deadline_ms=slo_ms,
    )


class TestRequestOrdering:
    def test_higher_priority_is_less(self):
        high = make_req(priority=8, req_id="h")
        low  = make_req(priority=2, req_id="l")
        assert high < low

    def test_equal_priority_earlier_deadline_is_less(self):
        early = make_req(priority=5, slo_ms=500.0,  req_id="e")
        late  = make_req(priority=5, slo_ms=5000.0, req_id="l")
        assert early < late

    def test_equality_by_id(self):
        r1 = make_req(req_id="x")
        r2 = make_req(req_id="x")
        assert r1 == r2

    def test_hash_by_id(self):
        r = make_req(req_id="abc")
        assert hash(r) == hash("abc")

    def test_remaining_tokens(self):
        r = make_req()
        r.tokens_generated = 20
        assert r.remaining_tokens == 30


class TestWorkerState:
    def make_worker(self, budget=1000, batch=4):
        return WorkerState(worker_id=0, kv_cache_budget=budget, max_batch_size=batch)

    def test_initial_state(self):
        w = self.make_worker()
        assert w.kv_cache_used == 0
        assert w.kv_cache_free == 1000
        assert w.batch_capacity == 4

    def test_store_kv_and_hit(self):
        w = self.make_worker()
        w.store_kv(hash("prefix"), 200)
        assert w.kv_cache_used == 200
        score = w.prefix_overlap_score(hash("prefix"))
        assert score == 200

    def test_miss_returns_zero(self):
        w = self.make_worker()
        assert w.prefix_overlap_score(hash("not_cached")) == 0

    def test_lru_eviction(self):
        w = self.make_worker(budget=300)
        w.store_kv(1, 100)
        time.sleep(0.01)
        w.store_kv(2, 100)
        time.sleep(0.01)
        w.store_kv(3, 100)
        assert w.kv_cache_used == 300

        # Storing 100 more should evict the LRU entry (key 1)
        w.store_kv(4, 100)
        assert 1 not in w.kv_cache
        assert w.kv_cache_used == 300

    def test_evict_lru_removes_oldest(self):
        w = self.make_worker(budget=500)
        w.store_kv(10, 100)
        time.sleep(0.01)
        w.store_kv(20, 100)
        time.sleep(0.01)
        w.store_kv(30, 100)
        w.evict_lru(300)   # need 300 free; currently 200 free; evict two entries
        assert w.kv_cache_free >= 300

    def test_remove_from_batch(self):
        w = self.make_worker()
        w.current_batch = ["r1", "r2", "r3"]
        w.remove_from_batch("r2")
        assert "r2" not in w.current_batch
        assert len(w.current_batch) == 2
