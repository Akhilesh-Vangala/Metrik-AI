from __future__ import annotations

import itertools
import logging
import multiprocessing
import operator
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from typing import Any, Callable

import pandas as pd

logger = logging.getLogger(__name__)

import platform
_MP_CONTEXT = multiprocessing.get_context("fork" if platform.system() != "Windows" else "spawn")


def parallel_feature_build(
    site_ids: list[int],
    build_fn: Callable[[int], pd.DataFrame],
    n_workers: int = 4,
) -> list[pd.DataFrame]:
    logger.info("Parallel feature build: %d sites, %d workers", len(site_ids), n_workers)
    t0 = time.perf_counter()
    results = []

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=_MP_CONTEXT) as executor:
        future_map = {executor.submit(build_fn, sid): sid for sid in site_ids}
        for future in as_completed(future_map):
            sid = future_map[future]
            try:
                df = future.result()
                results.append(df)
                logger.info("Site %d done: %d rows", sid, len(df))
            except Exception as e:
                logger.error("Site %d failed: %s", sid, e)

    elapsed = time.perf_counter() - t0
    logger.info("Parallel feature build completed in %.2fs", elapsed)
    return results


def parallel_model_training(
    site_ids: list[int],
    train_fn: Callable[[int], dict],
    n_workers: int = 4,
    **kwargs,
) -> dict[int, dict]:
    logger.info("Parallel model training: %d sites, %d workers", len(site_ids), n_workers)
    t0 = time.perf_counter()
    results = {}

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=_MP_CONTEXT) as executor:
        future_map = {executor.submit(train_fn, sid): sid for sid in site_ids}
        for future in as_completed(future_map):
            sid = future_map[future]
            try:
                results[sid] = future.result()
                logger.info("Site %d training complete", sid)
            except Exception as e:
                logger.error("Site %d training failed: %s", sid, e)

    elapsed = time.perf_counter() - t0
    logger.info("Parallel training completed in %.2fs (%d sites)", elapsed, len(results))
    return results


def threaded_chunk_reader(
    file_paths: list[str | Path],
    read_fn: Callable[[str | Path], pd.DataFrame],
    n_threads: int = 4,
) -> list[pd.DataFrame]:
    logger.info("Threaded chunk reading: %d files, %d threads", len(file_paths), n_threads)
    t0 = time.perf_counter()
    results = []

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        future_map = {executor.submit(read_fn, fp): fp for fp in file_paths}
        for future in as_completed(future_map):
            fp = future_map[future]
            try:
                results.append(future.result())
            except Exception as e:
                logger.error("Failed reading %s: %s", fp, e)

    elapsed = time.perf_counter() - t0
    logger.info("Threaded reading completed in %.2fs", elapsed)
    return results


def sequential_site_training(
    site_ids: list[int],
    train_fn: Callable[[int], dict],
) -> dict[int, dict]:
    t0 = time.perf_counter()
    results = {}
    for sid in site_ids:
        results[sid] = train_fn(sid)
    elapsed = time.perf_counter() - t0
    logger.info("Sequential training: %d sites in %.2fs", len(results), elapsed)
    return results


def generate_work_items(
    site_ids: list[int], meter_types: list[int]
) -> list[tuple[int, int]]:
    return list(itertools.product(site_ids, meter_types))


def chunked_dispatch(
    items: list, chunk_size: int
) -> list[list]:
    it = iter(items)
    return [list(itertools.islice(it, chunk_size)) for _ in range(0, len(items), chunk_size)]


class ThreadedResultCollector:
    def __init__(self, n_workers: int = 4):
        self._queue: Queue[tuple[Any, Callable, tuple] | None] = Queue()
        self._lock = Lock()
        self._results: dict[Any, Any] = {}
        self._errors: defaultdict[str, list] = defaultdict(list)
        self._workers: list[Thread] = []
        self._n_workers = n_workers

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            task_id, fn, args = item
            try:
                result = fn(*args)
                with self._lock:
                    self._results[task_id] = result
            except Exception as e:
                with self._lock:
                    self._errors[str(type(e).__name__)].append(task_id)
                logger.error("Task %s failed: %s", task_id, e)
            self._queue.task_done()

    def submit(self, task_id: Any, fn: Callable, *args: Any) -> None:
        self._queue.put((task_id, fn, args))

    def start(self) -> None:
        for i in range(self._n_workers):
            t = Thread(target=self._worker_loop, name=f"collector-{i}", daemon=True)
            t.start()
            self._workers.append(t)

    def collect(self) -> dict[Any, Any]:
        for _ in self._workers:
            self._queue.put(None)
        self._queue.join()
        for w in self._workers:
            w.join()
        return dict(self._results)

    def get_errors(self) -> dict[str, list]:
        return dict(self._errors)


def threaded_io_tasks(
    task_items: list[tuple[Any, Callable, tuple]],
    n_workers: int = 4,
) -> dict[Any, Any]:
    collector = ThreadedResultCollector(n_workers=n_workers)
    collector.start()
    for task_id, fn, args in task_items:
        collector.submit(task_id, fn, *args)
    results = collector.collect()
    logger.info("Threaded I/O: %d tasks completed, %d errors", len(results), sum(len(v) for v in collector.get_errors().values()))
    return results
