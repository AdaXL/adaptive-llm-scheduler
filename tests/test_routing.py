import pytest
from models.request import Request
from models.worker import WorkerState
from scheduler.routing import route_request, _worker_score
from scheduler.batching import compute_prefix_hash


def make_req(prompt_text="generic prompt", prompt_tokens=100):
    return Request(
        id="r1",
        prompt_text=prompt_text,
        prompt_tokens=prompt_tokens,
        max_output_tokens=50,
        priority=5,
        slo_deadline_ms=2000.0,
    )


def make_worker(worker_id=0, budget=2048, batch_size=32, batch_used=0, utilization=0.0):
    w = WorkerState(
        worker_id=worker_id,
        kv_cache_budget=budget,
        max_batch_size=batch_size,
    )
    w.current_batch = [f"x{i}" for i in range(batch_used)]
    w.utilization = utilization
    return w


class TestWorkerScore:
    def test_cache_hit_boosts_score(self):
        w = make_worker()
        prefix_hash = compute_prefix_hash("cached prompt")
        w.store_kv(prefix_hash, 100)

        score_hit  = _worker_score(w, prefix_hash)
        score_miss = _worker_score(w, hash("different"))
        assert score_hit > score_miss

    def test_high_utilization_penalises_score(self):
        w_busy = make_worker(utilization=0.9)
        w_free = make_worker(utilization=0.1)
        ph = hash("prompt")
        assert _worker_score(w_free, ph) > _worker_score(w_busy, ph)


class TestRouteRequest:
    def test_returns_worker_with_cache_hit(self):
        req = make_req(prompt_text="shared prefix here")
        prefix_hash = compute_prefix_hash(req.prompt_text)

        w_hit  = make_worker(worker_id=0)
        w_hit.store_kv(prefix_hash, 50)
        w_miss = make_worker(worker_id=1)

        result = route_request(req, [w_hit, w_miss])
        assert result is not None
        assert result.worker_id == 0

    def test_returns_none_when_all_full(self):
        req = make_req()
        w1 = make_worker(batch_size=4, batch_used=4)
        w2 = make_worker(batch_size=4, batch_used=4, worker_id=1)
        assert route_request(req, [w1, w2]) is None

    def test_returns_none_when_prompt_exceeds_kv_budget(self):
        # Request prompt is larger than the worker's entire KV budget → can never fit
        req = make_req(prompt_tokens=500)
        w = make_worker(budget=200)     # prompt_tokens(500) > budget(200)
        assert route_request(req, [w]) is None

    def test_prefers_less_loaded_worker(self):
        req = make_req()
        busy  = make_worker(worker_id=0, utilization=0.9)
        light = make_worker(worker_id=1, utilization=0.1)
        result = route_request(req, [busy, light])
        assert result is not None
        assert result.worker_id == 1

    def test_empty_worker_list_returns_none(self):
        req = make_req()
        assert route_request(req, []) is None
