"""Command-line entry points for the analytical mart."""

from __future__ import annotations

import argparse
import json

from .build import build_analytics
from core import config as db_config
from .db import all_meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m analytics")
    parser.add_argument("--source", help="Existing source SQLite database")
    parser.add_argument("--analytics", help="Analytics SQLite database")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("build")
    subcommands.add_parser("update")
    subcommands.add_parser("health")
    train_model = subcommands.add_parser("train-model")
    train_model.add_argument("--artifacts", default="analytics_artifacts")
    benchmark = subcommands.add_parser("benchmark")
    benchmark.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args(argv)
    try:
        args.source = db_config.get_db_path(args.source, require_exists=True)
        args.analytics = db_config.get_analytics_db_path(
            args.analytics, require_exists=args.command not in {"build", "update"}
        )
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

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
