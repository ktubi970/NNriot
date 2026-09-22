#!/usr/bin/env python3
"""Unified CLI for NNriot."""
import argparse
import logging
import os
import sys

def run_web():
    """Run the Flask web application."""
    from core import config
    print(f"Starting Flask web app on http://{config.HOST}:{config.PORT}...")
    from app import app
    from app.core import init_trainer_async
    init_trainer_async()
    app.run(
        debug=config.FLASK_DEBUG,
        host=config.HOST,
        port=config.PORT,
    )

def run_train():
    """Run the continuous trainer."""
    print("Starting continuous trainer (Ctrl-C to stop)...")
    from scripts.continuous_trainer import main
    main()

def run_backfill():
    """Run the backfill labels script."""
    print("Running backfill labels...")
    from scripts.backfill_labels import backfill_labels
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = backfill_labels()
    print(f"Backfill result: {result}")

def main():
    parser = argparse.ArgumentParser(description="NNriot CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("web", help="Start the Flask web application")
    subparsers.add_parser("train", help="Start the continuous training loop")
    subparsers.add_parser("backfill", help="Run label backfilling on the database")

    args = parser.parse_args()

    if args.command == "web":
        run_web()
    elif args.command == "train":
        run_train()
    elif args.command == "backfill":
        run_backfill()
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
