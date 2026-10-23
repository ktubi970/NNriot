#!/usr/bin/env python3
"""Unified CLI for NNriot."""
import argparse
import logging
import os
import sys

from core import config
from core.feature_labels import VECTOR_DIM


def _source_db(parser, value):
    try:
        return config.get_db_path(value, require_exists=True)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


def run_web(db_path):
    """Run the Flask application on the selected database."""
    os.environ["NNRIOT_DB_PATH"] = db_path
    print(f"Source database: {db_path}")
    print(f"Starting Flask web app on http://{config.HOST}:{config.PORT}...")
    from app import app as default_app, create_app
    app = (default_app if default_app.config["NNRIOT_DB_PATH"] == db_path
           else create_app({"NNRIOT_DB_PATH": db_path}))
    app.run(debug=config.FLASK_DEBUG, host=config.HOST, port=config.PORT)


def run_train(args, db_path):
    """Build an evaluated pre-match candidate checkpoint."""
    if getattr(args, "split_strategy", None) is not None and getattr(args, "split_manifest", None) is not None:
        sys.exit("--split-strategy cannot be combined with --split-manifest")
    print(f"Training candidate from source database: {db_path}")
    from scripts.train_pre_match_model import main as train_main

    training_args = [
        "--db", db_path,
        "--samples", str(args.samples),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
    ]
    if args.dim is not None:
        training_args.extend(("--dim", str(args.dim)))
    if args.output is not None:
        training_args.extend(("--output", args.output))
    if getattr(args, "split_strategy", None) is not None:
        training_args.extend(("--split-strategy", args.split_strategy))
    if getattr(args, "split_manifest", None) is not None:
        training_args.extend(("--split-manifest", args.split_manifest))
    if getattr(args, "calibrate", False):
        training_args.append("--calibrate")
    if getattr(args, "feature_version", None) is not None:
        training_args.extend(("--feature-version", args.feature_version))
    training_args.extend(("--seed", str(args.seed)))
    return train_main(training_args)


def run_benchmark_signal(args, db_path):
    """Run statistical draft signal benchmark."""
    print(f"Benchmarking draft signal on source database: {db_path}")
    from scripts.benchmark_draft_signal import main as bench_main
    bench_args = ["--db", db_path, "--samples", str(args.samples)]
    if args.output is not None:
        bench_args.extend(("--output", args.output))
    if getattr(args, "feature_version", None) is not None:
        bench_args.extend(("--feature-version", args.feature_version))
    return bench_main(bench_args)


def run_train_continuous(db_path):
    """Run the existing continuous training loop."""
    os.environ["NNRIOT_DB_PATH"] = db_path
    print(f"Starting continuous trainer on {db_path} (Ctrl-C to stop)...")
    from scripts.continuous_trainer import main
    return main(db_path=db_path)


def run_backfill(db_path):
    """Run label backfilling on the selected database."""
    print(f"Backfilling labels in source database: {db_path}")
    from scripts.backfill_labels import backfill_labels
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = backfill_labels(db_path=db_path)
    print(f"Backfill result: {result}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="NNriot CLI")
    commands = parser.add_subparsers(dest="command", required=True)

    web = commands.add_parser("web", help="Start the Flask web application")
    web.add_argument("--db", help="Source SQLite database")

    train = commands.add_parser("train", help="Train an evaluated pre-match candidate")
    train.add_argument("--db", help="Source SQLite database (read-only)")
    train.add_argument("--samples", type=int, default=5000)
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--dim", type=int, default=None)
    train.add_argument("--output", help="New candidate .keras checkpoint path")
    train.add_argument("--split-strategy", choices=["chronological", "patch"], default=None,
                       help="Holdout split strategy: chronological or patch")
    train.add_argument("--split-manifest", help="Path to frozen split manifest JSON to replay")
    train.add_argument("--calibrate", action="store_true", help="Fit temperature scaling calibration for winner head")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--feature-version",
        choices=["v1_hasher", "v2_draft", "v3_enriched"],
        default=None,
        help="Feature extraction version (v3_enriched: draft + player priors, v2_draft: canonical dense draft, v1_hasher: legacy string hashing)",
    )

    bench = commands.add_parser("benchmark-signal", help="Benchmark whether features hold statistical signal")
    bench.add_argument("--db", help="Source SQLite database (read-only)")
    bench.add_argument("--samples", type=int, default=10000)
    bench.add_argument("--output", help="Report output path (defaults to runs/draft_signal_report.json)")
    bench.add_argument(
        "--feature-version",
        choices=["v2_draft", "v3_enriched"],
        default=None,
        help="Feature version to benchmark (v3_enriched: draft + player priors, v2_draft: draft only)",
    )

    continuous = commands.add_parser("train-continuous", help="Run the continuous training loop")
    continuous.add_argument("--db", help="Source SQLite database")

    backfill = commands.add_parser("backfill", help="Backfill labels in the source database")
    backfill.add_argument("--db", help="Source SQLite database")

    args = parser.parse_args(argv)
    db_path = _source_db(parser, args.db)
    if args.command == "web":
        return run_web(db_path)
    if args.command == "train":
        return run_train(args, db_path)
    if args.command == "benchmark-signal":
        return run_benchmark_signal(args, db_path)
    if args.command == "train-continuous":
        return run_train_continuous(db_path)
    return run_backfill(db_path)


if __name__ == "__main__":
    main()
