"""Command-line entry points for the analytical mart."""

from __future__ import annotations

import argparse
import json

from .build import build_analytics
from .config import AnalyticsConfig
from .db import all_meta


def main(argv: list[str] | None = None) -> int:
    config = AnalyticsConfig()
    parser = argparse.ArgumentParser(prog="python -m analytics")
    parser.add_argument("--source", default=config.source_path)
    parser.add_argument("--analytics", default=config.analytics_path)
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("build")
    subcommands.add_parser("update")
    subcommands.add_parser("health")
    train_model = subcommands.add_parser("train-model")
    train_model.add_argument("--artifacts", default="analytics_artifacts")
    benchmark = subcommands.add_parser("benchmark")
    benchmark.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args(argv)

    if args.command in {"build", "update"}:
        report = build_analytics(
            args.source,
            args.analytics,
            incremental=args.command == "update",
        )
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        return 0 if report.quality_gate_passed else 2
    if args.command == "train-model":
        from .interpretable import train_and_evaluate

        report = train_and_evaluate(args.analytics, args.artifacts)
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        return 0 if report.final_test["available"] else 2
    if args.command == "benchmark":
        from .benchmark import run_benchmark

        report = run_benchmark(
            args.analytics,
            args.source,
            iterations=args.iterations,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["passed"] else 2
    print(json.dumps(all_meta(args.analytics), indent=2, ensure_ascii=False))
    return 0
