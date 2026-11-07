"""Repeatable latency measurements for the local analytics pipeline."""

from __future__ import annotations

import math
import platform
import time

from .cohorts import AnalyticsAbstention
from .engine import AnalysisEngine
from .service import AnalyticsService
from .types import AnalysisRequest, Intent


TARGETS_MS = {
    "cached_evidence": 100,
    "single_metric": 2_000,
    "chat_no_llm": 3_000,
}
QUESTION = "La vision est-elle associée à la victoire ?"


def percentile(values: list[float], value: int) -> float:
    """Return a nearest-rank percentile from a non-empty series."""

    if not values:
        raise ValueError("EMPTY_BENCHMARK_SERIES")
    ordered = sorted(values)
    index = max(0, math.ceil(value / 100 * len(ordered)) - 1)
    return ordered[min(index, len(ordered) - 1)]


def _measure(callable_) -> float:
    started = time.perf_counter()
    callable_()
    return (time.perf_counter() - started) * 1_000


def run_benchmark(
    path: str,
    source_path: str,
    iterations: int = 30,
) -> dict:
    """Measure cache, fresh analysis, and deterministic chat latency."""

    if iterations < 1:
        raise ValueError("INVALID_BENCHMARK_ITERATIONS")
    request = AnalysisRequest(Intent.EXPLAIN_METRIC, "vision_score")
    engine = AnalysisEngine(path)
    service = AnalyticsService(path, source_path)
    context = {
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        engine.run(request, question=QUESTION)
    except AnalyticsAbstention as exc:
        return {
            "available": False,
            "code": exc.code,
            "iterations": iterations,
            "p95_ms": {},
            "targets_ms": dict(TARGETS_MS),
            "passed": False,
            "context": context,
        }
    timings: dict[str, list[float]] = {
        "cached_evidence": [],
        "single_metric": [],
        "chat_no_llm": [],
    }
    # Warm up callables once to avoid cold-start disk/cache overhead in p95 calculation
    try:
        engine.cached(request, question=QUESTION)
        service.chat(QUESTION, None, {})
    except Exception:
        pass
    for _ in range(iterations):
        timings["cached_evidence"].append(
            _measure(lambda: engine.cached(request, question=QUESTION))
        )
    for _ in range(iterations):
        timings["single_metric"].append(
            _measure(lambda: engine.run_uncached(request, question=QUESTION))
        )
    service.freshness(force=True)
    for _ in range(iterations):
        timings["chat_no_llm"].append(
            _measure(lambda: service.chat(QUESTION, None, {}))
        )

    p95_ms = {
        name: percentile(values, 95) for name, values in timings.items()
    }
    return {
        "available": True,
        "code": None,
        "iterations": iterations,
        "p95_ms": p95_ms,
        "targets_ms": dict(TARGETS_MS),
        "passed": all(p95_ms[name] < limit for name, limit in TARGETS_MS.items()),
        "context": context,
    }
