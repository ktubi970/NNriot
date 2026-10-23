#!/usr/bin/env python3
"""Train a tabular tree baseline using HistGradientBoostingClassifier."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder

from core import config
from core.draft_features import (
    TABULAR_FEATURE_DIM,
    batch_extract_draft_features,
    get_tabular_feature_names,
)
from core.experiment import (
    SPLIT_NAMES,
    build_manifest,
    load_frozen_splits,
    read_manifest,
    restore_splits,
)
from core.pre_match_dataset import (
    load_records,
    split_records,
    to_pre_match_features,
)
from core.training_metrics import evaluate_winner

logger = logging.getLogger("train_tree_baseline")

ROLE_ORDER = {"TOP": 0, "JUNGLE": 1, "MIDDLE": 2, "BOTTOM": 3, "UTILITY": 4}
TABULAR_CATEGORICAL_FEATURE_INDICES: list[int] = list(range(10))


def extract_tabular_features(feat: dict) -> tuple[list[str], list[str]]:
    """Extract a legacy normalized 24-column categorical row and stable names.

    .. deprecated::
        Use :func:`core.draft_features.extract_tabular_draft_features` or
        :func:`core.draft_features.batch_extract_draft_features` instead.
    """
    clean = to_pre_match_features(feat)
    values: list[str] = []
    names: list[str] = []
    for team in ("team_a", "team_b"):
        participants = clean.get(team, [])
        if len(participants) != 5:
            raise ValueError("tree features require five participants per team")
        ordered = sorted(
            participants,
            key=lambda p: (ROLE_ORDER.get(p["role"], 5), p["role"], p["champion"]),
        )
        for index, player in enumerate(ordered):
            for field in ("champion", "role"):
                names.append(f"{team}_{index}_{field}")
                values.append(str(player[field]))
    for key in ("game_mode", "game_version", "season_id", "region"):
        names.append(key)
        value = clean.get(key)
        values.append("UNKNOWN" if value is None else str(value))
    return values, names


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def train_tree_model(
    records: list[dict],
    *,
    output_dir: str,
    split_strategy: str = "chronological",
    split_manifest: dict | None = None,
    seed: int = 42,
    compare_report: dict | None = None,
) -> dict:
    """Fit a HistGradientBoosting baseline, evaluate holdout, and optionally compare to neural."""
    if compare_report is not None and split_manifest is None:
        raise ValueError("split_manifest is required whenever compare_report is supplied")

    out_path = Path(output_dir).resolve()
    for artifact_name in ("model.joblib", "model.split.json", "model.metrics.json", "comparison.json"):
        if (out_path / artifact_name).exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {out_path / artifact_name}")

    if split_manifest is not None:
        if split_strategy != "chronological":
            raise ValueError("Cannot specify a non-default split_strategy alongside split_manifest")
        splits = restore_splits(records, split_manifest)
        split_meta = split_manifest.get(
            "split_metadata",
            {"requested_strategy": "frozen", "effective_strategy": "frozen", "fallback_reason": None},
        )
    else:
        splits, split_meta = split_records(records, strategy=split_strategy)

    manifest = build_manifest(splits, source_db=None, split_metadata=split_meta)

    # Validate against compare_report before fitting
    if compare_report is not None:
        if compare_report.get("dataset_sha256") != manifest["dataset_sha256"]:
            raise ValueError(
                f"dataset_sha256 mismatch with compare_report: expected {compare_report.get('dataset_sha256')}, got {manifest['dataset_sha256']}"
            )
        compare_split_sha = compare_report.get("split_sha256")
        if compare_split_sha and compare_split_sha != manifest["split_sha256"]:
            raise ValueError(
                f"split_sha256 mismatch with compare_report: expected {compare_split_sha}, got {manifest['split_sha256']}"
            )
        expected_counts = compare_report.get("split_counts")
        actual_counts = {k: len(splits[k]) for k in SPLIT_NAMES}
        if expected_counts and actual_counts != expected_counts:
            raise ValueError(f"split_counts mismatch: expected {expected_counts}, got {actual_counts}")

    feature_names = get_tabular_feature_names()
    X = {
        name: batch_extract_draft_features(
            [r.get("feature_json", {}) for r in splits[name]], mode="tabular"
        )
        for name in SPLIT_NAMES
    }
    X_train = X["train"]
    X_validation = X["validation"]
    X_test = X["test"]

    y = {
        name: np.asarray([r["labels_json"]["winner"] for r in splits[name]], dtype=int)
        for name in SPLIT_NAMES
    }

    if len(np.unique(y["train"])) < 2:
        raise ValueError("train partition must contain both classes 0 and 1")

    train_rate = float(y["train"].mean())
    if compare_report is not None:
        expected_prior = compare_report.get("train_positive_rate")
        if expected_prior is not None and not np.isclose(train_rate, expected_prior, atol=1e-5):
            raise ValueError(f"train_positive_rate mismatch: expected {expected_prior}, got {train_rate}")

    # Output directory creation and split manifest persistence before fitting
    out_path.mkdir(parents=True, exist_ok=True)
    _write_json(out_path / "model.split.json", manifest)

    cat_features = TABULAR_CATEGORICAL_FEATURE_INDICES
    best = None
    history = []
    for iterations in (50, 100, 150):
        model = HistGradientBoostingClassifier(
            categorical_features=cat_features,
            max_iter=iterations,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            learning_rate=0.1,
            early_stopping=False,
            random_state=seed,
        )
        model.fit(X_train, y["train"])
        val_probs = model.predict_proba(X_validation)[:, 1]
        validation = evaluate_winner(y["validation"], val_probs, train_rate)
        loss = validation["model"]["log_loss"]
        history.append({"max_iter": iterations, "validation": validation})
        if best is None or loss < best[0]:
            best = (loss, iterations, model)

    assert best is not None
    selected = best[2]
    selected_iter = best[1]

    # Evaluate final metrics on selected model
    metrics = {
        "train": evaluate_winner(y["train"], selected.predict_proba(X_train)[:, 1], train_rate, diagnostics=True),
        "validation": evaluate_winner(
            y["validation"], selected.predict_proba(X_validation)[:, 1], train_rate, diagnostics=True
        ),
        "test": evaluate_winner(y["test"], selected.predict_proba(X_test)[:, 1], train_rate, diagnostics=True),
    }

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_kind": "hist_gradient_boosting",
        "feature_names": feature_names,
        "evaluation_target": "winner",
        "is_calibrated": False,
        "dataset_sha256": manifest["dataset_sha256"],
        "split_sha256": manifest["split_sha256"],
        "train_positive_rate": train_rate,
        "split_counts": {k: len(splits[k]) for k in SPLIT_NAMES},
        "selected_max_iter": selected_iter,
        "selection_history": history,
        "metrics": metrics,
        "sklearn_version": sklearn.__version__,
    }

    # Save joblib bundle and metrics
    bundle = {
        "model": selected,
        "encoder": None,
        "feature_names": feature_names,
        "max_iter": selected_iter,
    }
    joblib.dump(bundle, out_path / "model.joblib")
    _write_json(out_path / "model.metrics.json", report)

    if compare_report is not None:
        comparison = {
            "dataset_sha256": manifest["dataset_sha256"],
            "split_sha256": manifest["split_sha256"],
            "selection_policy": "settings frozen using validation before test access",
            "test": {
                "constant": report["metrics"]["test"]["baseline"],
                "tree_raw": report["metrics"]["test"]["model"],
                "neural_raw": compare_report["metrics"]["test"]["model"],
            },
            "limitations": [
                "Point estimates; no significance claim.",
                "Players may overlap between temporal splits.",
                "Reusing a previously inspected test is retrospective benchmarking.",
            ],
        }
        if compare_report.get("calibrated_metrics") is not None:
            comparison["test"]["neural_calibrated"] = compare_report["calibrated_metrics"]["test"][
                "model"
            ]
        _write_json(out_path / "comparison.json", comparison)

    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="Explicit source SQLite database (read-only)")
    parser.add_argument("--samples", type=int, default=5000, help="Newest candidate matches (default: 5000)")
    parser.add_argument("--output", help="New output directory; defaults to runs/tree_<timestamp>/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-strategy", choices=["chronological", "patch"], default=None)
    parser.add_argument("--split-manifest", help="Path to frozen split manifest JSON to replay")
    parser.add_argument("--compare-to", help="Path to neural candidate model.metrics.json to benchmark against")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.compare_to and (args.split_manifest or args.split_strategy):
        parser.exit(2, "Error: --compare-to cannot be combined with explicit --split-manifest or --split-strategy\n")
    if args.split_strategy and args.split_manifest:
        parser.exit(2, "Error: Cannot combine --split-strategy with --split-manifest\n")

    try:
        db_path = config.get_db_path(args.db, require_exists=True)
        compare_report = None
        manifest = None

        if args.compare_to:
            compare_path = Path(args.compare_to).resolve()
            if not compare_path.exists():
                parser.exit(2, f"Error: Neural metrics report not found: {args.compare_to}\n")
            with compare_path.open("r", encoding="utf-8") as f:
                compare_report = json.load(f)
            # Find referenced split manifest, resolving relative paths against report directory
            manifest_ref = compare_report.get("split_manifest")
            if not manifest_ref:
                manifest_ref = str(compare_path.with_name("model.split.json"))
            manifest_path = Path(manifest_ref)
            if not manifest_path.is_absolute():
                manifest_path = (compare_path.parent / manifest_path).resolve()
            if not manifest_path.exists():
                parser.exit(2, f"Error: Referenced split manifest not found: {manifest_path}\n")
            manifest = read_manifest(manifest_path)
            splits = load_frozen_splits(db_path, manifest)
            records = [r for name in SPLIT_NAMES for r in splits[name]]
            strategy = "chronological"
        elif args.split_manifest:
            manifest_path = Path(args.split_manifest).resolve()
            if not manifest_path.exists():
                parser.exit(2, f"Error: Manifest not found: {args.split_manifest}\n")
            manifest = read_manifest(manifest_path)
            splits = load_frozen_splits(db_path, manifest)
            records = [r for name in SPLIT_NAMES for r in splits[name]]
            strategy = "chronological"
        else:
            strategy = args.split_strategy or "chronological"
            records = load_records(db_path, limit=args.samples)

        if args.output:
            output_dir = args.output
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            output_dir = str(REPO_ROOT / "runs" / f"tree_{stamp}")

        report = train_tree_model(
            records,
            output_dir=output_dir,
            split_strategy=strategy,
            split_manifest=manifest,
            seed=args.seed,
            compare_report=compare_report,
        )
        logger.info("Tree model successfully trained into %s", output_dir)
        logger.info("Held-out test: %s", json.dumps(report["metrics"]["test"]))
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        parser.exit(2, f"Tree training aborted: {exc}\n")


if __name__ == "__main__":
    main()
