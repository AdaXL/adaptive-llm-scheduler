"""
Rich live dashboard for the scheduler benchmark.

Renders a side-by-side comparison table refreshed every second.
"""

import asyncio
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich import box

from metrics.collector import MetricsCollector

REFRESH_INTERVAL_S = 1.0


def _fmt(value: float, decimals: int = 1, suffix: str = "") -> str:
    return f"{value:,.{decimals}f}{suffix}"


def build_table(
    adaptive: MetricsCollector,
    adaptive_queue: int,
    adaptive_workers_busy: int,
    adaptive_workers_total: int,
    fifo: Optional[MetricsCollector] = None,
    fifo_queue: int = 0,
    fifo_workers_busy: int = 0,
) -> Table:
    table = Table(
        title="[bold cyan]Adaptive LLM Inference Scheduler — Live Metrics[/bold cyan]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )

    table.add_column("Metric", style="bold", width=24)
    table.add_column("Adaptive Scheduler", justify="right", style="green")
    if fifo is not None:
        table.add_column("FIFO Baseline", justify="right", style="yellow")

    rows = [
        ("Throughput (tok/s)",
         _fmt(adaptive.throughput_tps, 0),
         _fmt(fifo.throughput_tps, 0) if fifo else None),
        ("TTFT P50 (ms)",
         _fmt(adaptive.ttft_p50),
         _fmt(fifo.ttft_p50) if fifo else None),
        ("TTFT P99 (ms)",
         _fmt(adaptive.ttft_p99),
         _fmt(fifo.ttft_p99) if fifo else None),
        ("Avg TBT (ms)",
         _fmt(adaptive.tbt_avg),
         _fmt(fifo.tbt_avg) if fifo else None),
        ("SLO Violation %",
         _fmt(adaptive.slo_violation_rate * 100, 1, "%"),
         _fmt(fifo.slo_violation_rate * 100, 1, "%") if fifo else None),
        ("KV Hit Rate",
         _fmt(adaptive.kv_hit_rate * 100, 1, "%"),
         "N/A" if fifo else None),
        ("Preemptions",
         str(adaptive.preemptions),
         "0" if fifo else None),
        ("Completed Requests",
         str(adaptive.completed_requests),
         str(fifo.completed_requests) if fifo else None),
        ("Queue Depth",
         str(adaptive_queue),
         str(fifo_queue) if fifo else None),
        ("Workers Busy",
         f"{adaptive_workers_busy}/{adaptive_workers_total}",
         f"{fifo_workers_busy}/{adaptive_workers_total}" if fifo else None),
    ]

    for label, adaptive_val, fifo_val in rows:
        if fifo is not None:
            table.add_row(label, adaptive_val, fifo_val or "")
        else:
            table.add_row(label, adaptive_val)

    return table


class LiveDisplay:
    def __init__(
        self,
        adaptive_scheduler,
        adaptive_metrics: MetricsCollector,
        fifo_scheduler=None,
        fifo_metrics: Optional[MetricsCollector] = None,
        console: Optional[Console] = None,
    ) -> None:
        self.adaptive_sched = adaptive_scheduler
        self.adaptive_metrics = adaptive_metrics
        self.fifo_sched = fifo_scheduler
        self.fifo_metrics = fifo_metrics
        self.console = console or Console()
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        with Live(console=self.console, refresh_per_second=4, screen=False) as live:
            while not self._stop.is_set():
                adaptive_busy = sum(
                    1 for w in self.adaptive_sched.workers if w.state.busy
                )
                fifo_busy = 0
                if self.fifo_sched is not None:
                    fifo_busy = sum(
                        1 for w in self.fifo_sched.workers if w.state.busy
                    )

                table = build_table(
                    adaptive=self.adaptive_metrics,
                    adaptive_queue=self.adaptive_sched.queue_depth,
                    adaptive_workers_busy=adaptive_busy,
                    adaptive_workers_total=len(self.adaptive_sched.workers),
                    fifo=self.fifo_metrics,
                    fifo_queue=self.fifo_sched.queue_depth if self.fifo_sched else 0,
                    fifo_workers_busy=fifo_busy,
                )
                live.update(table)
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=REFRESH_INTERVAL_S)
                except asyncio.TimeoutError:
                    pass
