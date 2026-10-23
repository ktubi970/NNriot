"""Reproduce the threshold-search comparison, without models or live data.

Run from the repository root:
  .venv312/Scripts/python.exe docs/research-improvement/2026-10-11-threshold-evidence/benchmark.py
The baseline source is the byte-for-byte snapshot captured before the change.
"""
from pathlib import Path
import hashlib
import importlib.machinery
import importlib.util
import json
import platform
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from core.selective_classification import find_threshold_for_accuracy

OUT = Path(__file__).resolve().parent
loader = importlib.machinery.SourceFileLoader("threshold_before", str(OUT / "selective_classification.py.before"))
spec = importlib.util.spec_from_loader(loader.name, loader)
before = importlib.util.module_from_spec(spec)
loader.exec_module(before)

seed = 20261011
rng = np.random.default_rng(seed)
probabilities = rng.uniform(.01, .99, size=45_000)
labels = rng.integers(0, 2, size=len(probabilities))
kwargs = dict(target_accuracy=1.0, min_count=1000)
searches = {"before": before.find_threshold_for_accuracy, "after": find_threshold_for_accuracy}
timings = {name: [] for name in searches}
results = {}
# Warm up imports/allocator, then alternate order across three paired repeats.
for search in searches.values():
    search(labels[:200], probabilities[:200], **kwargs)
for repeat in range(3):
    for name in (list(searches) if repeat % 2 == 0 else list(reversed(searches))):
        start = time.perf_counter()
        results[name] = searches[name](labels, probabilities, **kwargs)
        timings[name].append(time.perf_counter() - start)

counter_p = .600001 + np.arange(1000) * 1e-6
counter_y = np.ones(1000, dtype=int)
counter_y[:2] = 0
counter_y[-99:] = 0
summary = {
    "python": platform.python_version(), "numpy": np.__version__, "seed": seed,
    "n": len(labels), "parameters": kwargs,
    "input_sha256": hashlib.sha256(probabilities.tobytes() + labels.tobytes()).hexdigest(),
    "baseline_source_sha256": hashlib.sha256((OUT / "selective_classification.py.before").read_bytes()).hexdigest(),
    "after_source_sha256": hashlib.sha256((ROOT / "core/selective_classification.py").read_bytes()).hexdigest(),
    "seconds": timings,
    "median_seconds": {name: statistics.median(values) for name, values in timings.items()},
    "results": results,
    "counterexample": {name: search(counter_y, counter_p, target_accuracy=.9007) for name, search in searches.items()},
    "interpretation": "Synthetic empirical search cost and correctness only; no model accuracy or prospective risk guarantee.",
}
(OUT / "benchmark.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
