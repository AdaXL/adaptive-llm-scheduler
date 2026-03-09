"""
Adaptive LLM Inference Scheduler — CLI entry point.

Usage examples:
  python main.py
  python main.py --workers 4 --rps 30 --scenario burst --duration 20 --compare-fifo
  python main.py --workers 2 --rps 10 --scenario mixed_priority --duration 30
"""

import argparse
import asyncio
import sys
import time
from typing import List

from rich.console import Console

from metrics.collector import MetricsCollector
from metrics.display import LiveDisplay
from workers.gpu_worker import SimulatedGPUWorker
from scheduler.core import AdaptiveScheduler
from baselines.fifo import FIFOScheduler
from load_gen.generator import LoadGenerator


def build_workers(
    n: int,
    kv_cache_budget: int,
    max_batch_size: int,
    metrics: MetricsCollector,
) -> List[SimulatedGPUWorker]:
    return [
        SimulatedGPUWorker(
            worker_id=i,
            kv_cache_budget=kv_cache_budget,
            max_batch_size=max_batch_size,
            metrics=metrics,
        )
        for i in range(n)
    ]


async def run_simulation(
    n_workers: int,
    rps: float,
    scenario: str,
    duration_s: float,
    kv_cache_budget: int,
    max_batch_size: int,
    compare_fifo: bool,
    seed: int,
    console: Console,
) -> None:
    # ── Adaptive scheduler setup ──────────────────────────────────────────────
    a_metrics = MetricsCollector()
    a_workers = build_workers(n_workers, kv_cache_budget, max_batch_size, a_metrics)
    adaptive = AdaptiveScheduler(workers=a_workers, metrics=a_metrics)

    # ── FIFO baseline setup (optional, parallel run) ──────────────────────────
    f_metrics: MetricsCollector | None = None
    fifo: FIFOScheduler | None = None
    f_workers: List[SimulatedGPUWorker] = []

    if compare_fifo:
        f_metrics = MetricsCollector()
        f_workers = build_workers(n_workers, kv_cache_budget, max_batch_size, f_metrics)
        fifo = FIFOScheduler(workers=f_workers, metrics=f_metrics)

    # ── Live display ──────────────────────────────────────────────────────────
    display = LiveDisplay(
        adaptive_scheduler=adaptive,
        adaptive_metrics=a_metrics,
        fifo_scheduler=fifo,
        fifo_metrics=f_metrics,
        console=console,
    )

    # ── Load generators ───────────────────────────────────────────────────────
    a_gen = LoadGenerator(
        submit_fn=adaptive.submit,
        rps=rps,
        scenario=scenario,
        duration_s=duration_s,
        seed=seed,
    )
    f_gen = None
    if fifo is not None:
        f_gen = LoadGenerator(
            submit_fn=fifo.submit,
            rps=rps,
            scenario=scenario,
            duration_s=duration_s,
            seed=seed,
        )

    # ── Launch all tasks ──────────────────────────────────────────────────────
    console.print(
        f"\n[bold cyan]Starting simulation[/bold cyan] | "
        f"workers={n_workers} | rps={rps} | scenario={scenario} | "
        f"duration={duration_s}s | compare_fifo={compare_fifo}\n"
    )

    tasks = []

    # Scheduler loops
    tasks.append(asyncio.create_task(adaptive.run(), name="adaptive-scheduler"))
    if fifo:
        tasks.append(asyncio.create_task(fifo.run(), name="fifo-scheduler"))

    # Worker loops
    for w in a_workers:
        tasks.append(asyncio.create_task(w.run(), name=f"adaptive-worker-{w.state.worker_id}"))
    for w in f_workers:
        tasks.append(asyncio.create_task(w.run(), name=f"fifo-worker-{w.state.worker_id}"))

    # Load generators
    tasks.append(asyncio.create_task(a_gen.run(), name="adaptive-load-gen"))
    if f_gen:
        tasks.append(asyncio.create_task(f_gen.run(), name="fifo-load-gen"))

    # Display loop
    tasks.append(asyncio.create_task(display.run(), name="display"))

    # ── Wait for load generators to finish, then gracefully shut down ─────────
    gen_tasks = [t for t in tasks if "load-gen" in t.get_name()]
    if gen_tasks:
        await asyncio.gather(*gen_tasks)

    # Give workers a moment to drain
    await asyncio.sleep(2.0)

    await adaptive.stop()
    if fifo:
        await fifo.stop()
    display.stop()

    # Cancel remaining background tasks
    for t in tasks:
        if not t.done():
            t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    # ── Final summary ─────────────────────────────────────────────────────────
    _print_summary(console, a_metrics, a_gen, f_metrics, f_gen)


def _print_summary(
    console: Console,
    a_metrics: MetricsCollector,
    a_gen: LoadGenerator,
    f_metrics,
    f_gen,
) -> None:
    from rich.table import Table
    from rich import box

    console.print("\n[bold green]=== Final Results ===[/bold green]\n")

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
    table.add_column("Metric", style="bold")
    table.add_column("Adaptive", justify="right", style="green")
    if f_metrics:
        table.add_column("FIFO Baseline", justify="right", style="yellow")

    def row(label, a_val, f_val=None):
        if f_metrics:
            table.add_row(label, a_val, f_val or "")
        else:
            table.add_row(label, a_val)

    row("Total Submitted",
        str(a_gen.total_submitted),
        str(f_gen.total_submitted) if f_gen else None)
    row("Total Completed",
        str(a_metrics.completed_requests),
        str(f_metrics.completed_requests) if f_metrics else None)
    row("Throughput (tok/s)",
        f"{a_metrics.throughput_tps:,.0f}",
        f"{f_metrics.throughput_tps:,.0f}" if f_metrics else None)
    row("TTFT P50 (ms)",
        f"{a_metrics.ttft_p50:.1f}",
        f"{f_metrics.ttft_p50:.1f}" if f_metrics else None)
    row("TTFT P99 (ms)",
        f"{a_metrics.ttft_p99:.1f}",
        f"{f_metrics.ttft_p99:.1f}" if f_metrics else None)
    row("SLO Violation %",
        f"{a_metrics.slo_violation_rate*100:.1f}%",
        f"{f_metrics.slo_violation_rate*100:.1f}%" if f_metrics else None)
    row("KV Cache Hit Rate",
        f"{a_metrics.kv_hit_rate*100:.1f}%",
        "0.0% (no routing)" if f_metrics else None)
    row("Preemptions",
        str(a_metrics.preemptions),
        "0" if f_metrics else None)

    console.print(table)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Adaptive LLM Inference Scheduler — research prototype",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--workers",      type=int,   default=4,        help="Number of simulated GPU workers")
    p.add_argument("--rps",          type=float, default=30.0,     help="Base requests per second")
    p.add_argument("--scenario",     choices=["steady", "burst", "mixed_priority"],
                                                 default="steady", help="Traffic pattern")
    p.add_argument("--duration",     type=float, default=20.0,     help="Simulation duration (seconds)")
    p.add_argument("--kv-slots",     type=int,   default=2048,     help="KV cache budget per worker (token slots)")
    p.add_argument("--batch-size",   type=int,   default=32,       help="Max batch size per worker")
    p.add_argument("--seed",         type=int,   default=42,       help="Random seed")
    p.add_argument("--compare-fifo", action="store_true",          help="Also run FIFO baseline for comparison")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    console = Console()
    try:
        asyncio.run(
            run_simulation(
                n_workers=args.workers,
                rps=args.rps,
                scenario=args.scenario,
                duration_s=args.duration,
                kv_cache_budget=args.kv_slots,
                max_batch_size=args.batch_size,
                compare_fifo=args.compare_fifo,
                seed=args.seed,
                console=console,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        sys.exit(0)
