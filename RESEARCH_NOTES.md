# Research Notes: Adaptive LLM Inference Scheduler

This document tracks the research exploration, design decisions, related work survey, and potential future directions developed alongside this project.

---

## Table of Contents

1. [The Core Problem](#1-the-core-problem)
2. [Design Decisions and Engineering Insights](#2-design-decisions-and-engineering-insights)
3. [Related Work Survey](#3-related-work-survey)
4. [Simulation Findings](#4-simulation-findings)
5. [Gaps in Existing Research](#5-gaps-in-existing-research)
6. [Open Questions](#6-open-questions)

---

## 1. The Core Problem

**Question:** When massive LLM requests arrive simultaneously, how do you schedule GPU inference to minimise latency and maximise throughput?

This is fundamentally a systems problem, not a modelling problem. The challenge arises from three interacting constraints:

1. **GPU memory is finite.** Every active request occupies KV cache space proportional to its prompt length plus tokens generated so far. When memory runs out, nothing can run.

2. **LLM inference has two distinct compute phases.** The *prefill* phase (processing the input prompt) is compute-bound and parallelisable. The *decode* phase (autoregressive token generation) is memory-bandwidth-bound and sequential. They have fundamentally different resource profiles and cannot be optimised with the same strategy.

3. **Request heterogeneity is extreme.** Prompts range from 10 to 100,000 tokens. Output lengths range from 1 token ("yes") to 8,000 tokens (a full essay). Priority tiers span interactive users (need response in 500ms) to batch jobs (can wait hours). No single static policy handles this well.

### Why FIFO Fails

A first-in-first-out scheduler fails on multiple fronts:

- **Head-of-line blocking.** A single long-prompt, long-output request holds a batch slot for minutes, blocking dozens of short requests that could have completed in milliseconds.
- **No SLO differentiation.** A background analytics job and a real-time user query are treated identically.
- **Wasted prefill computation.** If 1,000 requests all share the same system prompt, FIFO recomputes the same key-value tensors 1,000 times.
- **Inefficient batch utilisation.** Static batching (waiting for a fixed batch to fill and drain together) leaves GPU idle whenever one request finishes before others.

---

## 2. Design Decisions and Engineering Insights

This section records non-obvious implementation decisions made during development, and the reasoning behind each.

### 2.1 Generation-Based Lazy Deletion in the Priority Queue

**Problem encountered:** When a request's priority was bumped (via SLO urgency) or after preemption, the request was re-inserted into the heap. Simple ID-based lazy deletion — marking an ID as "removed" in a set — breaks on re-insertion: the new push calls `_removed_ids.discard(req.id)`, un-marking the *old* heap entry as well. Both the old and new entries then become live, causing the request to be dispatched twice.

**Root cause:** ID-based lazy deletion assumes each ID appears at most once in the heap. Re-insertion violates that assumption silently.

**Solution:** Generation-based lazy deletion. Each push increments a per-ID generation counter. Heap entries carry their generation at push time. On pop/peek, an entry is only valid if its generation matches the current generation for that ID. Re-pushing atomically invalidates all prior entries by bumping the generation, regardless of how many old copies remain in the heap.

```python
# Each push:
gen = self._generation.get(req.id, 0) + 1
self._generation[req.id] = gen
heapq.heappush(self._heap, (..., req.id, gen, req))

# On pop/peek: skip if generation doesn't match
if self._generation.get(rid, -1) != gen:
    continue  # stale entry
```

**Lesson:** Lazy deletion in heaps is only safe when each key is unique. Re-insertion — common in scheduling systems that bump priorities — requires a versioning scheme.

---

### 2.2 Eager Batch-Slot Reservation

**Problem encountered:** The scheduler dispatches requests via `asyncio.create_task(worker.add_request(req))`. Since this is non-blocking, `add_request` does not run until the scheduler yields. If the scheduler routes multiple requests in the same `_scheduling_iteration` loop without updating the batch count synchronously, it over-commits: all iterations see the same (stale) `batch_capacity` and route more requests than `max_batch_size` allows.

**Solution:** The scheduler eagerly appends the request ID to `worker.state.current_batch` *before* creating the task. This updates `batch_capacity` synchronously so subsequent loop iterations see the correct count.

```python
target_state.current_batch.append(req.id)           # synchronous reservation
asyncio.create_task(target_worker.add_request(req))  # async dispatch
```

`add_request` then checks `if req.id not in self.state.current_batch` before appending, preventing double-counting.

**Lesson:** In cooperative async systems, any state that multiple iterations of a tight loop depend on must be updated *before* yielding control, even if the downstream side effect (the actual work) happens later.

---

### 2.3 Skipping Infeasible Head-of-Queue

**Problem encountered:** Requests with `prompt_tokens > kv_cache_budget` can never be served by any worker. After SLO urgency bumps, such a request can reach priority 9 and permanently occupy the head of the priority queue. When `route_request()` returns `None` for it, a naive `break` exits the routing loop entirely — blocking all lower-priority feasible requests from being dispatched even when workers are completely idle.

**Observable symptom:** Workers showed `active=0, batch=0` (fully idle) while the queue held hundreds of requests. `route_request` was returning `None` 4,853 consecutive times per simulation.

**Solution:** Switch from heap-top-only routing to a full priority-ordered scan. Skip any request for which `route_request()` returns `None`; continue to the next. Only break when all workers have genuinely reached batch capacity.

```python
for req in self.queue.to_list():        # priority-ordered scan
    target = route_request(req, workers)
    if target is None:
        continue                         # skip infeasible, don't block
    ...dispatch...
    if all(w.batch_capacity <= 0 for w in workers):
        break                            # genuinely full, stop
```

**Lesson:** "No feasible assignment for this request" and "system is at capacity" are different conditions that must be handled differently in the routing loop. Conflating them causes total queue starvation.

---

### 2.4 Batch-Level Speculative Decoding

**Problem encountered:** The original implementation ran speculative decoding *per request* in a for-loop:

```python
for req in spec_reqs:
    result = await run_speculative_decode(draft_lat, dec_lat)  # 3.2ms each
```

With 32 requests in the batch, this made one decode step take `32 × 3.2ms = 104ms` instead of the expected `3.2ms`. Throughput dropped 32× compared to normal decode.

**Root cause:** Real speculative decoding runs a single draft forward pass and a single verify forward pass over the *entire batch*, not per-request sequential passes. The per-request loop was modelling single-request spec decoding, not batched spec decoding.

**Solution:** Run one shared draft sleep, then synchronous acceptance sampling per request (no sleep), then one shared verify sleep for all spec requests:

```python
await asyncio.sleep(DRAFT_K * draft_lat)          # one draft phase for whole batch
for req in spec_reqs:
    accepted = rejection_sample(DRAFT_K, ACCEPT_PROB)  # synchronous, no sleep
    ...
await asyncio.sleep(dec_lat)                       # one verify phase
```

**Lesson:** Simulation fidelity requires modelling the *hardware execution model* accurately. A GPU runs a forward pass over a batched tensor in one operation; simulating it as N sequential operations produces qualitatively wrong latency predictions.

---

### 2.5 Prompt Token Capping

**Problem encountered:** The log-normal distribution `lognormvariate(mu=5.5, sigma=1.0)` produces values above 2048 (the KV cache budget) with ~1.7% probability. These requests can never be served. After SLO urgency bumps they accumulate at the top of the priority queue and trigger the infeasible-head-of-queue problem at scale.

**Solution:** Cap prompt tokens at 1500 in the load generator (`min(prompt_tokens, 1500)`), staying safely within the 2048 default KV budget. A production system would either reject oversized requests at the API layer or route them to a worker with a larger KV budget.

---

### 2.6 Preemption Thrashing and the Minimum Priority Gap

**Problem encountered:** With `MIN_PREEMPT_PRIORITY_GAP = 1`, the scheduler preempted constantly — 97+ preemptions per 83 submitted requests. Each preemption re-queues the victim, which then competes for dispatch and may get preempted again. This cycling degraded SLO compliance and throughput simultaneously.

**Analysis:** Preemption is only net-positive when:
```
priority_benefit > re_prefill_cost
```
If the priority gap is small (e.g., priority 6 preempting priority 5), the benefit is marginal but the cost — losing the victim's KV cache, re-queuing, re-prefilling — is substantial.

**Solution:** Require a minimum gap of 3 priority levels before preempting. This limits preemption to genuinely urgent cases (e.g., a real-time user request displacing a background job) and eliminates thrashing.

**Remaining open question:** The right threshold should be a function of the victim's remaining output tokens (fewer remaining = cheaper to preempt) and current KV cache pressure (higher pressure = more expensive re-prefill). A dynamic threshold based on these signals would be strictly better than a fixed constant.

---

## 3. Related Work Survey

A comprehensive survey of academic and systems research addressing the same core question.

### 3.1 Foundational Systems

**Orca** (OSDI 2022) — Yu et al., Seoul National University / FriendliAI
https://www.usenix.org/conference/osdi22/presentation/yu
Introduced *iteration-level scheduling* (continuous batching): dispatch new requests the moment a slot frees, rather than waiting for the batch to drain. Also introduced selective batching (batch all ops except attention, which requires same sequence length). 36.9× throughput improvement over FasterTransformer.

**vLLM / PagedAttention** (SOSP 2023) — Kwon et al., UC Berkeley
https://arxiv.org/abs/2309.06180
Solves KV cache memory fragmentation by managing it like OS virtual memory — fixed-size pages, on-demand allocation, copy-on-write for shared prefixes. Eliminates the 60–80% memory waste from static reservation. 2–4× throughput improvement over Orca.

**SGLang / RadixAttention** (NeurIPS 2024) — Zheng et al., LMSYS / UC Berkeley
https://arxiv.org/abs/2312.07104
Stores KV cache in a radix tree with LRU eviction, enabling automatic prefix reuse across requests without user annotation. Cache-aware scheduling prioritises requests whose prefixes are already cached. Up to 5× throughput improvement by eliminating redundant prefill computation.

**TensorRT-LLM** (NVIDIA, 2023)
https://github.com/NVIDIA/TensorRT-LLM
Production inference runtime with kernel fusion, quantisation, paged attention, and in-flight batching. The industry baseline for throughput benchmarks.

---

### 3.2 Batching Refinements

**SARATHI / Sarathi-Serve** (OSDI 2024) — Agrawal et al., Microsoft Research
https://arxiv.org/abs/2308.16369 / https://arxiv.org/abs/2403.02310
Chunks long prefills into equal-size pieces and fills remaining batch slots with decode tokens ("decode-maximal batching"). Eliminates the prefill stall that forces a long prompt to monopolise an entire batch iteration. Reduces pipeline bubble fraction 6.29×.

**Learning to Rank for LLM Scheduling** (NeurIPS 2024) — Fu et al., UCSD / UC Berkeley
https://arxiv.org/abs/2408.15792
Trains a ranking model to approximate Shortest-Job-First scheduling without knowing output lengths in advance. Reduces P99 TTFT 6.9× by eliminating head-of-line blocking from long requests.

---

### 3.3 KV Cache Management

**Hydragen** (ICML 2024) — Juravsky et al., Stanford
https://arxiv.org/abs/2402.05099
Hardware-aware attention kernel that decomputes shared-prefix attention (matrix-matrix multiply once for the batch) from per-sequence unique-suffix attention. 3–30× throughput improvement depending on prefix length.

**Mooncake** (FAST 2025, Best Paper) — Qin et al., Moonshot AI / Kimi
https://arxiv.org/abs/2407.00079
Treats the KV cache as a first-class resource. Disaggregates it across GPU HBM, CPU DRAM, SSD, and remote GPU memory with a global SLO-aware scheduler. 525% throughput increase in simulation, deployed at production scale for Kimi.

**Infinite-LLM** (arXiv 2024) — Lin et al.
https://arxiv.org/abs/2401.02669
Introduces DistAttention: a mathematically equivalent distributed reformulation of multi-head attention across GPU/CPU memory in a datacenter. Supports context lengths up to 1.9M tokens across 32 A100s.

**Deja Vu** (ICML 2023) — Liu et al.
https://arxiv.org/abs/2310.17157
Identifies contextual sparsity: up to 85% of attention heads and MLP neurons are inactive for any given token. A lightweight predictor selects the active subset on-the-fly, reducing OPT-175B latency 2× with no accuracy drop.

---

### 3.4 Priority Scheduling and SLO-Aware Serving

**Llumnix** (OSDI 2024) — Sun et al., Alibaba DAMO Academy
https://arxiv.org/abs/2406.03243
Adds a cross-instance rescheduling layer that live-migrates requests and their in-flight KV caches between GPU instances for load balancing, priority enforcement, and auto-scaling. Cuts P99 tail latency by an order of magnitude.

**AlpaServe** (OSDI 2023) — Li et al., UC Berkeley
https://arxiv.org/abs/2302.11665
Shows that model parallelism enables statistical multiplexing of bursty workloads across multiple models. Handles 10× higher request rates or 6× more burstiness while maintaining >99% SLO attainment.

**ProServe** (arXiv 2024) — arXiv:2512.12928
Combines SlideBatching (priority-windowed batching at the engine) with GoRouting (queue-state-aware routing at the service layer) for multi-tenant multi-priority scheduling.

---

### 3.5 Speculative Decoding

**Speculative Decoding** (ICML 2023) — Leviathan et al., Google
https://arxiv.org/abs/2211.17192
The foundational paper. Draft model generates K tokens; large model verifies all K in one forward pass via rejection sampling. Provably preserves target distribution. 2–3× speedup on T5-XXL.

**Speculative Sampling** (arXiv 2023) — Chen et al., DeepMind
https://arxiv.org/abs/2302.01318
Concurrent independent development on Chinchilla (70B). 2–2.5× speedup in a distributed multi-accelerator setting.

**Medusa** (ICML 2024) — Cai et al., Together AI / Princeton
https://arxiv.org/abs/2401.10774
Avoids a separate draft model by attaching multiple lightweight decoding heads directly to the base LLM. Tree-based attention verifies multiple candidate continuations in one pass. 2.2–3.6× speedup.

**EAGLE / EAGLE-2 / EAGLE-3** (ICML 2024, EMNLP 2024, arXiv 2025) — Li et al., MSRA
https://arxiv.org/abs/2401.15077
Improves draft acceptance by predicting *feature vectors* from the penultimate LLM layer rather than token embeddings. EAGLE-2 adds dynamic draft trees. EAGLE-3 uses multi-layer features. Typically 3–4× speedup.

---

### 3.6 Disaggregated Serving

**DistServe** (OSDI 2024) — Zhong et al., PKU / UCSD
https://arxiv.org/abs/2401.09670
Assigns prefill and decode to separate GPU pools, eliminating interference between the compute-heavy prefill and memory-bandwidth-bound decode phases. 7.4× more requests served or 12.6× tighter SLO attainment versus colocated systems.

**Splitwise** (ISCA 2024) — Patel et al., Microsoft Research
https://arxiv.org/abs/2311.18677
Concurrent with DistServe. Three-pool design (prefill / decode / hybrid). Transfers KV caches over interconnect between phase-specialised nodes. 1.4–2.35× throughput improvement.

---

### 3.7 Multi-Tenant and Multi-Model Serving

**S-LoRA** (MLSys 2024) — Sheng et al., UC Berkeley / Stanford
https://arxiv.org/abs/2311.03285
Extends PagedAttention's unified paging to co-manage KV cache and LoRA adapter weights, enabling thousands of fine-tuned adapter variants to be served concurrently. 4× throughput improvement over naive vLLM with LoRA.

---

### 3.8 Surveys

| Survey | Venue | Scope |
|---|---|---|
| Towards Efficient Generative LLM Serving | ACM Computing Surveys 2025 | Full stack: algorithms, systems, quantisation, parallelism |
| Taming the Titans | arXiv 2025 (2504.19720) | Instance-level and cluster-level scheduling, 2023–2025 |
| LLM Inference Scheduling | TechRxiv Oct 2025 | Control-plane and data-plane scheduling, RL-based management |

---

### 3.9 Quick-Reference Table

| Paper | Year | Venue | Category |
|---|---|---|---|
| Orca | 2022 | OSDI | Continuous batching |
| vLLM / PagedAttention | 2023 | SOSP | KV cache management |
| FlexGen | 2023 | ICML | CPU/SSD offloading |
| Speculative Decoding (Leviathan) | 2023 | ICML | Speculative decode |
| Speculative Sampling (Chen) | 2023 | arXiv | Speculative decode |
| Deja Vu | 2023 | ICML | Contextual sparsity |
| AlpaServe | 2023 | OSDI | Statistical multiplexing |
| SARATHI | 2023 | arXiv | Chunked prefill |
| Medusa | 2024 | ICML | Speculative decode |
| EAGLE / EAGLE-2 | 2024 | ICML/EMNLP | Speculative decode |
| SGLang / RadixAttention | 2024 | NeurIPS | KV prefix caching |
| Hydragen | 2024 | ICML | Shared-prefix attention |
| S-LoRA | 2024 | MLSys | Multi-tenant LoRA |
| DistServe | 2024 | OSDI | Prefill/decode disaggregation |
| Splitwise | 2024 | ISCA | Phase-split architecture |
| Sarathi-Serve | 2024 | OSDI | Stall-free batching |
| Llumnix | 2024 | OSDI | Live KV migration |
| Learning to Rank | 2024 | NeurIPS | SJF approximation |
| Mooncake | 2025 | FAST | KVCache-centric architecture |
| EAGLE-3 | 2025 | arXiv | Speculative decode |

---

## 4. Simulation Findings

### 4.1 Burst Scenario (4 workers, 200 RPS spike for 5s)

| Metric | Adaptive | FIFO |
|---|---|---|
| Throughput (tok/s) | 8,147 | 7,974 |
| TTFT P50 | 5.1 ms | 4.1 ms |
| TTFT P99 | **46.3 ms** | 53.7 ms |
| SLO Violation % | 0.3% | 0.0% |
| KV Hit Rate | **11.3%** | 0.0% |

Under burst load within system capacity, both schedulers drain the queue. The adaptive scheduler's P99 TTFT is lower because priority ordering ensures urgent requests surface early, and KV cache routing avoids redundant prefill computation for ~1 in 9 requests.

### 4.2 Mixed-Priority Scenario (4 workers, 50 RPS sustained)

| Metric | Adaptive | FIFO |
|---|---|---|
| Throughput (tok/s) | 6,594 | 6,512 |
| TTFT P50 | **8.0 ms** | 10.9 ms |
| TTFT P99 | 74.2 ms | 75.9 ms |
| SLO Violation % | **0.0%** | 0.0% |
| KV Hit Rate | **11.4%** | 0.0% |

At moderate load both meet all SLOs, but the adaptive scheduler's TTFT P50 is 26% lower. High-priority requests are served before low-priority ones that arrived earlier — which FIFO cannot do.

### 4.3 Bugs Found During Development That Changed the Findings

The following bugs were found and fixed during implementation. Each meaningfully affected the results:

| Bug | Symptom | Impact if left unfixed |
|---|---|---|
| ID-based lazy deletion + re-insertion | Queue depth reported as 3,000+ for 987 requests | Phantom duplicates cause 10× over-dispatching |
| Sequential per-request speculative decode | Decode step took 104ms instead of 3.2ms for batch-32 | 32× throughput degradation under moderate load |
| `break` on infeasible head-of-queue | Workers idle with 800+ requests queued | Total queue starvation after first oversized request |
| No eager batch-slot reservation | Workers over-committed within one scheduler loop | Dispatch of 2× more requests than max_batch_size |
| Preemption gap of 1 | 97 preemptions per 83 requests | Thrashing degrades SLO compliance and throughput simultaneously |

---

## 5. Gaps in Existing Research

The following questions are not addressed (or not addressed satisfactorily) by any paper in the survey above. These are legitimate research opportunities.

### 5.1 Policy Interaction Effects (Primary Gap)

Every major paper studies one technique in isolation. When you combine them, they interact:

**KV-cache routing vs. load balancing conflict.**
Routing to the worker with the highest prefix cache hit score may concentrate requests on one worker, starving others. The optimal trade-off is unknown. A routing score that weights cache hits against queue depth imbalance should exist, but its form has not been derived analytically.

**Preemption cost depends on KV cache state.**
The cost of preempting a request is `re_prefill_cost`, which scales with prompt length. A request with a 100-token prompt costs 1/20th as much to preempt as one with a 2000-token prompt. Current systems use a fixed priority gap threshold for preemption. The optimal threshold should be:
```
preempt if: priority_gap × expected_remaining_wait > re_prefill_cost(prompt_tokens)
```
No paper derives this condition.

**Speculative decoding desynchronises batch alignment.**
Different requests in a batch accept different numbers of speculative tokens per step. After one speculative step, some requests are 4 tokens ahead, others 0. How to re-synchronise without wasting accepted tokens or stalling lagging requests is unresolved in a batched serving context.

**SLO urgency bumps interact with speculative decode gating.**
Urgency-bumped requests are the ones most likely to have spec decoding disabled (priority > 6 gate). But they are also the ones for which any extra throughput is most valuable. The current gate is a coarse heuristic. An optimal policy would gate on time-to-deadline relative to remaining output tokens, not on priority level.

### 5.2 No Simulation Framework for Scheduling Research

There is no open-source, modular LLM inference scheduling simulator analogous to NS-3 for networking or CSIM for processor simulation. Production systems (vLLM, SGLang) are too large to modify for scheduling experiments. This creates a high barrier to entry for scheduling research. A validated simulator with a clean plugin API would enable the research community to test new policies in hours rather than months.

### 5.3 Preemption Cost Model

No paper models the true cost of preemption accounting for KV cache loss, re-prefill time, and downstream effects on batch utilisation. The papers that implement preemption (vLLM swap mode, Llumnix) treat it as a mechanism rather than studying its cost-benefit threshold.

### 5.4 Speculative Decoding Under Batch Heterogeneity

All speculative decoding papers evaluate on either single-request or uniformly distributed batches. In a real serving system, a batch contains requests at very different stages of generation — some on token 5, others on token 500. The interaction between batch heterogeneity and speculative decode acceptance rates, and the resulting batch management complexity, is unstudied.

---

## 6. Open Questions

Questions that arose during implementation that have not been resolved:

**Q1. What is the optimal preemption cost threshold?**
Currently a fixed priority gap of 3. Should it depend on `prompt_tokens` (re-prefill cost), `tokens_generated` (sunk cost), and `remaining_tokens` (future cost)? A formal derivation would be a clean theoretical contribution.

**Q2. How does KV cache routing interact with load balancing under overload?**
Under normal load, routing to the cache-hit worker is always beneficial (saved prefill > routing suboptimality). Under overload, concentrating load on a few workers starves others. What is the crossover point?

**Q3. Can speculative decoding be applied to a heterogeneous batch safely?**
When requests are at different stages of generation (different token counts), does joint speculative decoding still preserve the target distribution? The rejection sampling proof assumes all requests start from the same state, which is never true in a real serving batch.

**Q4. Is SLO urgency bumping the right mechanism, or should it be continuous?**
Current implementation bumps by +2 at 50ms before deadline. A continuous urgency signal (priority is a function of `time_to_deadline / expected_remaining_service_time`) would be more principled. The Earliest Deadline First (EDF) scheduling literature has results on this, but they assume preemptive systems with known service times — neither assumption holds cleanly for LLM serving.

**Q5. What is the minimum simulation fidelity needed to predict real-system behaviour?**
This simulator uses two scalar latency formulas. Real GPUs have CUDA kernel launch overhead, memory bandwidth saturation curves, NVLink contention, and NCCL communication costs. Which of these matter enough to include in a simulator intended to predict scheduling policy ranking (not absolute latency)?

**Q6. Does the priority queue need to be lock-free for production use?**
This implementation is single-threaded asyncio. A production scheduler running at 10,000 RPS would need the scheduler loop on a separate thread from the request-handling threads, requiring lock-free or lock-minimal data structures. How much does this change the design?

---

*Last updated: March 2026*
*This document is a living record — update it as experiments are run, papers are read, and new questions arise.*
