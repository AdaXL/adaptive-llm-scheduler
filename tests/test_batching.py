import pytest
from models.request import Request, RequestStatus
from models.worker import WorkerState
from scheduler.priority_queue import PriorityRequestQueue
from scheduler.batching import select_batch, can_fit_request, compute_prefix_hash


def make_req(req_id, priority=5, prompt_tokens=100, prompt_text="hello world"):
    return Request(
        id=req_id,
        prompt_text=prompt_text,
        prompt_tokens=prompt_tokens,
        max_output_tokens=50,
        priority=priority,
        slo_deadline_ms=2000.0,
    )


def make_worker(budget=10000, batch_size=8, batch_used=0):
    w = WorkerState(worker_id=0, kv_cache_budget=budget, max_batch_size=batch_size)
    w.current_batch = [f"existing_{i}" for i in range(batch_used)]
    return w


class TestComputePrefixHash:
    def test_same_text_same_hash(self):
        assert compute_prefix_hash("hello") == compute_prefix_hash("hello")

    def test_different_text_different_hash(self):
        assert compute_prefix_hash("hello") != compute_prefix_hash("world")

    def test_uses_only_first_256_chars(self):
        long_text = "a" * 300 + "b" * 300
        same_prefix = "a" * 300 + "c" * 300
        assert compute_prefix_hash(long_text) == compute_prefix_hash(same_prefix)


class TestCanFitRequest:
    def test_fits_when_capacity_available(self):
        w = make_worker(budget=10000, batch_size=8)
        req = make_req("r1", prompt_tokens=100)
        assert can_fit_request(req, w)

    def test_does_not_fit_batch_full(self):
        w = make_worker(budget=10000, batch_size=4, batch_used=4)
        req = make_req("r1")
        assert not can_fit_request(req, w)

    def test_does_not_fit_kv_cache_full(self):
        w = make_worker(budget=50, batch_size=8)  # tiny KV budget
        req = make_req("r1", prompt_tokens=100)   # needs 100 slots
        assert not can_fit_request(req, w)

    def test_fits_with_cache_hit(self):
        # Budget=300: store 200-token prefix entry, fill 90 more → only 10 free.
        # A 200-token request should still fit due to full cache hit (needs 0 new slots).
        w = make_worker(budget=300, batch_size=8)
        prefix = "cached prompt text"
        h = compute_prefix_hash(prefix)
        w.store_kv(h, 200)              # cache the prefix (200 slots used)
        w.store_kv(hash("filler"), 90)  # fill 90 more → only 10 free
        req = make_req("r1", prompt_tokens=200, prompt_text=prefix)
        # needed_kv = max(0, 200 - 200) = 0 → fits despite only 10 free slots
        assert can_fit_request(req, w)


class TestSelectBatch:
    def _make_queue(self, requests):
        q = PriorityRequestQueue()
        for r in requests:
            q.push(r)
        return q

    def test_empty_queue_returns_empty(self):
        q = PriorityRequestQueue()
        w = make_worker()
        result = select_batch(q, w)
        assert result == []

    def test_selects_highest_priority_first(self):
        reqs = [
            make_req("low",  priority=1),
            make_req("high", priority=9),
            make_req("mid",  priority=5),
        ]
        q = self._make_queue(reqs)
        w = make_worker(batch_size=1)
        selected = select_batch(q, w, max_new=1)
        assert len(selected) == 1
        assert selected[0].id == "high"

    def test_respects_max_new_limit(self):
        reqs = [make_req(f"r{i}") for i in range(10)]
        q = self._make_queue(reqs)
        w = make_worker(batch_size=32)
        selected = select_batch(q, w, max_new=3)
        assert len(selected) == 3

    def test_selected_requests_removed_from_queue(self):
        reqs = [make_req(f"r{i}") for i in range(5)]
        q = self._make_queue(reqs)
        w = make_worker(batch_size=32)
        selected = select_batch(q, w, max_new=3)
        assert len(q) == 2
        selected_ids = {r.id for r in selected}
        remaining_ids = {r.id for r in q.to_list()}
        assert selected_ids.isdisjoint(remaining_ids)

    def test_skips_requests_that_do_not_fit(self):
        w = make_worker(budget=50, batch_size=8)  # tiny KV budget
        big_req = make_req("big", prompt_tokens=200)
        small_req = make_req("small", prompt_tokens=10)
        q = self._make_queue([big_req, small_req])
        selected = select_batch(q, w, max_new=2)
        # big_req cannot fit; small_req should be selected
        ids = [r.id for r in selected]
        assert "small" in ids
        assert "big" not in ids
