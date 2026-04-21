from __future__ import annotations

import cProfile
import gc
import io
import logging
import pstats
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    name: str
    method: str
    elapsed_seconds: float
    peak_memory_mb: float = 0.0
    input_size: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def rows_per_second(self) -> float:
        if self.elapsed_seconds > 0 and self.input_size > 0:
            return self.input_size / self.elapsed_seconds
        return 0.0


class BenchmarkSuite:
    def __init__(self):
        self.results: list[BenchmarkResult] = []

    def time_function(
        self,
        name: str,
        method: str,
        fn: Callable,
        *args,
        input_size: int = 0,
        n_runs: int = 1,
        **kwargs,
    ) -> tuple[Any, BenchmarkResult]:
        gc.collect()

        times = []
        result = None
        for _ in range(n_runs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            times.append(time.perf_counter() - t0)

        elapsed = np.mean(times)
        br = BenchmarkResult(
            name=name,
            method=method,
            elapsed_seconds=elapsed,
            input_size=input_size,
            extra={"n_runs": n_runs, "std": float(np.std(times))} if n_runs > 1 else {},
        )
        self.results.append(br)
        logger.info("[%s] %s: %.4fs (n=%d)", name, method, elapsed, input_size)
        return result, br

    def compare(self, name: str, input_size: int | None = None) -> dict[str, float]:
        relevant = [r for r in self.results if r.name == name]
        if input_size is not None:
            relevant = [r for r in relevant if r.input_size == input_size]
        if len(relevant) < 2:
            return {}

        baseline = relevant[0]
        speedups = {}
        for r in relevant[1:]:
            if r.elapsed_seconds > 0:
                speedups[r.method] = baseline.elapsed_seconds / r.elapsed_seconds
        return speedups

    def summary_table(self) -> list[dict]:
        rows = []
        for r in self.results:
            rows.append({
                "component": r.name,
                "method": r.method,
                "time_seconds": round(r.elapsed_seconds, 4),
                "input_size": r.input_size,
                "rows_per_sec": round(r.rows_per_second),
                "peak_memory_mb": round(r.peak_memory_mb, 1),
            })
        return rows

    def print_summary(self):
        print(f"\n{'Component':<25} {'Method':<20} {'Time (s)':<12} {'Rows':<12} {'Rows/s':<12}")
        print("-" * 81)
        for r in self.results:
            print(
                f"{r.name:<25} {r.method:<20} {r.elapsed_seconds:<12.4f} "
                f"{r.input_size:<12} {r.rows_per_second:<12.0f}"
            )

        seen = set()
        for r in self.results:
            key = (r.name, r.input_size)
            if key in seen:
                continue
            seen.add(key)
            speedups = self.compare(r.name, input_size=r.input_size)
            if speedups:
                baseline_method = [
                    x for x in self.results
                    if x.name == r.name and x.input_size == r.input_size
                ][0].method
                size_label = f"n={r.input_size:,}" if r.input_size else ""
                for method, speedup in speedups.items():
                    print(f"  {r.name} ({size_label}): {method} is {speedup:.1f}x faster than {baseline_method}")


def profile_function(fn: Callable, *args, output_path: str | None = None, **kwargs) -> str:
    profiler = cProfile.Profile()
    profiler.enable()
    fn(*args, **kwargs)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(30)

    report = stream.getvalue()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(output_path)
        logger.info("Profile saved to %s", output_path)

    return report


def measure_memory(fn: Callable, *args, **kwargs) -> tuple[Any, float]:
    import tracemalloc
    tracemalloc.start()

    gc.collect()
    result = fn(*args, **kwargs)

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / 1e6
    return result, peak_mb
