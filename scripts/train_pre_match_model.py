#!/usr/bin/env python3
"""Train a pre-match candidate with chronological validation and an untouched test."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np

from core import config, json_utils
from core.feature_labels import VECTOR_DIM
from core.pre_match_dataset import (
    chronological_split,
    load_records,
    split_records,
    to_pre_match_features,
)
from core.calibration import (
    apply_temperature_to_probabilities,
    fit_temperature,
    load_calibration_artifact,
    save_calibration_artifact,
)
from core.experiment import (
    build_manifest,
    load_frozen_splits,
    read_manifest,
    restore_splits,
)
from core.training_metrics import evaluate_winner

logger = logging.getLogger("train_pre_match_model")


def build_multi_output_model(input_dim: int):
    # Keep dataset checks and --help available without loading TensorFlow.
    from core.generate_graph import build_multi_output_model as build
    return build(input_dim=input_dim)


def build_draft_dense_model(input_dim: int | None = None):
    # Keep dataset checks and --help available without loading TensorFlow.
    from core.generate_graph import build_draft_dense_model as build
    return build(input_dim=input_dim)


class ModelWithReport(tuple):
    __slots__ = ()

    def __new__(cls, model, report):
        return super().__new__(cls, (model, report))

    @property
    def model(self):
        return self[0]

    @property
    def report(self):
        return self[1]

    def __getattr__(self, name):
        return getattr(self[0], name)

    def __setattr__(self, name, value):
        if name in dir(ModelWithReport) or name in dir(tuple):
            super().__setattr__(name, value)
        else:
            setattr(self[0], name, value)

    def __dir__(self):
        return sorted(set(super().__dir__() + dir(self[0])))


def fetch_training_records(limit: int = 5000, db_path: str | None = None) -> list[dict]:
    """Read the newest eligible candidates without writing to the source database."""
    return load_records(config.get_db_path(db_path, require_exists=True), limit=limit)


def _winner_probabilities(model, x, batch_size: int) -> np.ndarray:
    probabilities = []
    n = len(x["team_a"]) if isinstance(x, dict) else x.shape[0]
    for start in range(0, n, batch_size):
        if isinstance(x, dict):
            batch_x = {
                "team_a": x["team_a"][start:start + batch_size],
                "team_b": x["team_b"][start:start + batch_size],
            }
        else:
            batch_x = x[start:start + batch_size]
            if hasattr(batch_x, "toarray"):
                batch_x = batch_x.toarray()
        predictions = model.predict_on_batch(batch_x)
        if isinstance(predictions, dict):
            probabilities.append(np.asarray(predictions["winner"])[:, 1])
        else:
            probabilities.append(np.asarray(predictions)[:, 1])
    return np.concatenate(probabilities)


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _get_feat(r: dict) -> dict:
    fj = r.get("feature_json") or {}
    if isinstance(fj, str):
        try:
            fj = json.loads(fj)
        except (ValueError, TypeError):
            fj = {}
    if isinstance(fj, dict) and "team_a" in fj and "team_b" in fj:
        return fj
    return to_pre_match_features(fj)


def _log_selective_metrics_summary(metrics: dict, title_prefix: str = "") -> None:
    for split_name in ("validation", "test"):
        split_eval = metrics.get(split_name, {})
        diag = split_eval.get("model", {}).get("diagnostics", {})
        selective = diag.get("selective_metrics", {})
        if not selective:
            continue
        uncond_acc = split_eval.get("model", {}).get("accuracy", 0.0)
        prefix = f"{title_prefix} " if title_prefix else ""
        logger.info(
            "Selective Evaluation [%s%s] - Unconditional Accuracy: %.4f (100.0%% coverage)",
            prefix, split_name.upper(), uncond_acc
        )
        if "conviction_ge_4pct" in selective:
            t4 = selective["conviction_ge_4pct"]
            logger.info(
                "  * Conviction >= 4%% (Target Tier): accuracy=%.4f (95%% Wilson CI: [%.4f, %.4f]) | coverage=%.1f%% (n=%d)",
                t4["accuracy"], t4["wilson_lower"], t4["wilson_upper"], t4["coverage"] * 100.0, t4["count"]
            )
        for tier_key, s in selective.items():
            if tier_key in ("conviction_ge_0pct", "conviction_ge_4pct"):
                continue
            logger.info(
                "    %s (tau=%.2f): accuracy=%.4f (CI: [%.4f, %.4f]) | coverage=%.1f%% (n=%d)",
                tier_key, s["threshold"], s["accuracy"], s["wilson_lower"], s["wilson_upper"],
                s["coverage"] * 100.0, s["count"]
            )


def train_model(
    records: list[dict],
    epochs: int = 5,
    batch_size: int = 64,
    checkpoint_path: str | None = None,
    dim: int | None = None,
    *,
    seed: int = 42,
    source_db: str | None = None,
    split_manifest: dict | None = None,
    split_strategy: str = "chronological",
    calibrate: bool = False,
    feature_version: str | None = None,
    architecture: str = "auto",
    use_pcgrad: bool = True,
) -> Any:
    """Select on validation winner log-loss, then evaluate test exactly once.

    Outputs are candidate artifacts. They are neither statistically calibrated
    nor installed as the application's active model.
    """
    if feature_version is None:
        if (dim is not None and dim != 8) or architecture == "mlp":
            feature_version = "v1_hasher"
        else:
            feature_version = "deepsets"

    if feature_version not in ("v1_hasher", "v2_draft", "v3_enriched", "deepsets", "deepsets_v2"):
        raise ValueError(f"feature_version must be 'v1_hasher', 'v2_draft', 'v3_enriched', 'deepsets', or 'deepsets_v2', got {feature_version!r}")
    if dim is None:
        if feature_version == "deepsets_v2":
            from core.features.pipeline import ENRICHED_PLAYER_TENSOR_DIM
            dim = ENRICHED_PLAYER_TENSOR_DIM
        elif feature_version == "deepsets":
            from core.deepsets_dataset import PLAYER_TENSOR_DIM
            dim = PLAYER_TENSOR_DIM
        elif feature_version == "v3_enriched":
            from core.enriched_features import ENRICHED_DENSE_DIM
            dim = ENRICHED_DENSE_DIM
        elif feature_version == "v2_draft":
            from core.draft_features import DENSE_FEATURE_DIM
            dim = DENSE_FEATURE_DIM
        else:
            dim = VECTOR_DIM
    for name, value in (("epochs", epochs), ("batch_size", batch_size), ("dim", dim)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if checkpoint_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        checkpoint_path = str(REPO_ROOT / "runs" / f"pre_match_{stamp}" / "model.keras")
    output = Path(checkpoint_path).resolve()
    if output.suffix != ".keras":
        raise ValueError("checkpoint_path must end in .keras")
    if output == (BASE_DIR / "model_v2.keras").resolve():
        raise ValueError("Use a separate candidate checkpoint, not the active model")
    active_env_checkpoint = os.environ.get("NNRIOT_CHECKPOINT_PATH")
    if active_env_checkpoint and output == Path(active_env_checkpoint).resolve():
        raise ValueError("Use a separate candidate checkpoint, not the configured active model")

    manifest_path = output.with_suffix(".split.json")
    report_path = output.with_suffix(".metrics.json")
    calibration_path = output.with_suffix(".calibration.json")
    candidate = output.with_name(output.stem + ".candidate.keras")
    for artifact in (output, manifest_path, report_path, candidate, calibration_path):
        if artifact.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {artifact}")

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

    if calibrate:
        val_rows = splits["validation"]
        if len(val_rows) < 20:
            raise ValueError(f"Calibration requires at least 20 validation rows, got {len(val_rows)}")
        val_labels = set(r["labels_json"]["winner"] for r in val_rows)
        if val_labels != {0, 1}:
            raise ValueError("Calibration requires both classes 0 and 1 in validation partition")

    counts = {name: len(rows) for name, rows in splits.items()}
    logger.info("Dataset splits: %s", counts)
    manifest = build_manifest(splits, source_db=source_db, split_metadata=split_meta)
    dataset_sha256 = manifest["dataset_sha256"]
    split_sha256 = manifest["split_sha256"]
    # Materialize the frozen membership before any training. A failed run keeps
    # its manifest for diagnosis; subsequent runs must use a fresh output path.
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(manifest_path, manifest)

    if feature_version in ("deepsets", "deepsets_v2"):
        if feature_version == "deepsets_v2" or dim == 14:
            from core.features.matchup_matrix import LaneMatchupTracker
            from core.features.pipeline import DraftFeaturePipeline
            logger.info("Fitting point-in-time LaneMatchupTracker on train split (%d games)...", len(splits["train"]))
            matchup_tracker = LaneMatchupTracker()
            for r in splits["train"]:
                fj = _get_feat(r)
                w = r["labels_json"]["winner"]
                matchup_tracker.update_match(fj.get("team_a") or [], fj.get("team_b") or [], winner=w)
            pipeline = DraftFeaturePipeline(matchup_tracker=matchup_tracker)
            x = {name: pipeline.batch_extract_tensors([_get_feat(r) for r in rows])
                 for name, rows in splits.items()}
        else:
            from core.deepsets_dataset import batch_extract_team_tensors
            x = {name: batch_extract_team_tensors([_get_feat(r) for r in rows])
                 for name, rows in splits.items()}
    elif feature_version == "v3_enriched":
        from core.enriched_features import batch_extract_enriched_features
        x = {name: batch_extract_enriched_features([_get_feat(r) for r in rows], mode="dense")
             for name, rows in splits.items()}
    elif feature_version == "v2_draft":
        from core.draft_features import batch_extract_draft_features
        x = {name: batch_extract_draft_features([_get_feat(r) for r in rows], mode="dense")
             for name, rows in splits.items()}
    else:
        features = {name: [to_pre_match_features(r["feature_json"]) for r in rows]
                    for name, rows in splits.items()}
        x = {name: json_utils.json_to_vector(rows, dim=dim) for name, rows in features.items()}

    y = {name: np.asarray([r["labels_json"]["winner"] for r in rows], dtype=int)
         for name, rows in splits.items()}
    train_rate = float(y["train"].mean())

    import tensorflow as tf
    from core.generate_graph import LOSS_WEIGHTS, LOSS_PER_HEAD, build_enriched_model
    from core.training_utils import _build_targets, sparse_microbatch_generator

    tf.keras.utils.set_random_seed(seed)
    if feature_version in ("deepsets", "deepsets_v2") or architecture in ("deepsets", "deepsets_v2"):
        from core.equivariant_model import build_equivariant_deepsets_model
        use_attn = True if (feature_version == "deepsets_v2" or dim == 14) else False
        model = build_equivariant_deepsets_model(player_feature_dim=dim, use_cross_attention=use_attn)
    elif feature_version == "v3_enriched":
        model = build_enriched_model(input_dim=dim)
    elif feature_version == "v2_draft":
        model = build_draft_dense_model(input_dim=dim)
    else:
        model = build_multi_output_model(input_dim=dim)
    targets = _build_targets(splits["train"])
    if feature_version in ("v2_draft", "v3_enriched"):
        targets = {"winner": targets["winner"]}
    best_loss = float("inf")
    selected_epoch = None
    history = []
    started = time.monotonic()
    loss_fns = {k: tf.keras.losses.get(LOSS_PER_HEAD[k]) for k in LOSS_PER_HEAD} if (feature_version in ("deepsets", "deepsets_v2") and use_pcgrad) else None
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        seen = 0
        if feature_version in ("deepsets", "deepsets_v2"):
            from core.pcgrad import pcgrad_step
            n_train = len(x["train"]["team_a"])
            for start in range(0, n_train, batch_size):
                xb = {
                    "team_a": x["train"]["team_a"][start:start + batch_size],
                    "team_b": x["train"]["team_b"][start:start + batch_size],
                }
                yb = {k: targets[k][start:start + batch_size] for k in targets if k in model.output_names}
                if use_pcgrad:
                    losses = pcgrad_step(model, model.optimizer, xb, yb, loss_fns, loss_weights=LOSS_WEIGHTS)
                    batch_loss = sum(losses.values())
                else:
                    res = model.train_on_batch(xb, yb, return_dict=True)
                    batch_loss = float(res["loss"])
                total_loss += batch_loss * len(xb["team_a"])
                seen += len(xb["team_a"])
        else:
            for xb, yb in sparse_microbatch_generator(x["train"], targets, batch_size=batch_size):
                model.reset_metrics()
                result = model.train_on_batch(xb, yb, return_dict=True)
                loss = float(result["loss"])
                if not np.isfinite(loss):
                    raise ValueError("Training produced a non-finite loss")
                total_loss += loss * len(xb)
                seen += len(xb)
        validation = evaluate_winner(y["validation"],
            _winner_probabilities(model, x["validation"], batch_size), train_rate)
        val_loss = validation["model"]["log_loss"]
        history.append({"epoch": epoch, "train_loss": total_loss / seen,
                        "validation": validation})
        logger.info("Epoch %d/%d: train_loss=%.4f validation winner log_loss=%.4f accuracy=%.3f",
                    epoch, epochs, total_loss / seen, val_loss,
                    validation["model"]["accuracy"])
        if val_loss < best_loss:
            best_loss = val_loss
            selected_epoch = epoch
            model.save(candidate)

    # Reload the selected epoch (including its optimizer state); the test set
    # has not been passed to the model or used to choose this checkpoint.
    del model
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(candidate)

    raw = {name: _winner_probabilities(model, x[name], batch_size)
           for name in ("train", "validation")}
    fit = fit_temperature(raw["validation"], y["validation"]) if calibrate else None
    raw["test"] = _winner_probabilities(model, x["test"], batch_size)
    metrics = {name: evaluate_winner(y[name], raw[name], train_rate, diagnostics=True)
               for name in ("train", "validation", "test")}
    calibrated_metrics = None
    if fit is not None:
        calibrated_metrics = {
            name: evaluate_winner(
                y[name],
                apply_temperature_to_probabilities(raw[name], fit["temperature"]),
                train_rate,
                diagnostics=True,
            )
            for name in ("train", "validation", "test")
        }

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_db": manifest["source_db"], "dataset_sha256": dataset_sha256,
        "split_sha256": split_sha256,
        "checkpoint": str(output), "split_manifest": str(manifest_path),
        "parameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "dim": dim,
            "seed": seed,
            "feature_version": feature_version,
        },
        "tensorflow_version": tf.__version__,
        "gpu_devices": [device.name for device in tf.config.list_physical_devices("GPU")],
        "split_counts": counts, "train_positive_rate": train_rate,
        "selection_metric": "validation.winner.log_loss", "selected_epoch": selected_epoch,
        "is_calibrated": False, "evaluation_target": "winner",
        "baseline_policy": "constant positive rate fitted on training split only",
        "active_heads": ["winner"] if feature_version == "v2_draft" else [name for name, weight in LOSS_WEIGHTS.items() if weight > 0],
        "disabled_heads": [name for name in LOSS_WEIGHTS if name != "winner"] if feature_version == "v2_draft" else [name for name, weight in LOSS_WEIGHTS.items() if weight == 0],
        "limitations": [
            "No probability calibration fitted." if fit is None else "Winner-only temperature calibration fitted on validation split.",
            "Chronological holdout only; players may overlap between splits.",
            "Only winner is evaluated; other heads are not validated.",
            "A small pilot does not establish predictive quality."
        ],
        "metrics": metrics, "history": history,
        "elapsed_seconds": time.monotonic() - started,
    }
    symmetry_diagnostics = None
    if feature_version == "deepsets":
        test_swapped = {"team_a": x["test"]["team_b"], "team_b": x["test"]["team_a"]}
        p_norm = raw["test"]
        p_swap = _winner_probabilities(model, test_swapped, batch_size)
        max_violation = float(np.max(np.abs(p_norm - (1.0 - p_swap))))
        symmetry_diagnostics = {
            "is_equivariant": True,
            "max_symmetry_violation": max_violation,
            "anti_symmetry_guarantee": max_violation < 1e-4,
        }
        logger.info("Equivariant anti-symmetry check: max_violation=%.6e guaranteed=%s",
                    max_violation, max_violation < 1e-4)

    report["symmetry_diagnostics"] = symmetry_diagnostics

    # The application never reads this distinct output unless explicitly promoted.
    if output.exists():
        raise FileExistsError(f"Checkpoint appeared during training: {output}")
    candidate.replace(output)

    calibration_artifact_path = None
    if fit is not None:
        calibration_artifact_path = save_calibration_artifact(
            output, manifest_path, fit, input_dim=dim
        )
        verified = load_calibration_artifact(output, input_dim=dim)
        if verified is None:
            raise ValueError("written calibration artifact failed verification")
        model.is_calibrated = True
        model.calibration_temperature = verified["temperature"]
        model.calibration_metadata = verified
    else:
        model.is_calibrated = False
        model.calibration_temperature = None
        model.calibration_metadata = None

    report.update(
        is_calibrated=model.is_calibrated,
        calibrated_heads=["winner"] if fit is not None else [],
        calibrated_metrics=calibrated_metrics,
        calibration_artifact=str(calibration_artifact_path) if calibration_artifact_path else None,
    )
    _write_json(report_path, report)
    logger.info("Candidate checkpoint: %s", output)
    logger.info("Held-out test: %s", json.dumps(metrics["test"]))
    logger.info("Evaluation report: %s", report_path)
    _log_selective_metrics_summary(metrics)
    if calibrated_metrics is not None:
        _log_selective_metrics_summary(calibrated_metrics, title_prefix="Calibrated")
    return ModelWithReport(model, report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="Explicit source SQLite database (read-only)")
    parser.add_argument("--samples", type=int, default=5000, help="Newest candidate matches, total across splits")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--output", help="New .keras candidate path; defaults to a unique directory in runs/")
    parser.add_argument("--split-strategy", choices=["chronological", "patch"], default=None,
                        help="Holdout split strategy: chronological or patch")
    parser.add_argument("--split-manifest", help="Path to frozen split manifest JSON to replay")
    parser.add_argument("--calibrate", action="store_true", help="Fit temperature scaling calibration for winner head")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-version", choices=["v1_hasher", "v2_draft", "v3_enriched", "deepsets", "deepsets_v2"], default="deepsets",
                        help="Feature extraction version: v1_hasher, v2_draft, v3_enriched, deepsets, or deepsets_v2 (14-dim + cross-attention)")
    parser.add_argument("--architecture", choices=["auto", "mlp", "deepsets"], default="auto")
    parser.add_argument("--no-pcgrad", action="store_true", help="Disable PCGrad gradient deconfliction")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.split_strategy is not None and args.split_manifest is not None:
        parser.exit(2, "Training aborted: Cannot specify both --split-strategy and --split-manifest\n")
    try:
        args.db = config.get_db_path(args.db, require_exists=True)
        if args.split_manifest is not None:
            manifest_file = Path(args.split_manifest)
            if not manifest_file.exists():
                parser.exit(2, f"Training aborted: Manifest file not found: {args.split_manifest}\n")
            manifest = read_manifest(manifest_file)
            if args.samples != 5000:
                logger.info("--samples argument is ignored when replaying a frozen manifest.")
            splits = load_frozen_splits(args.db, manifest)
            records = [r for name in ("train", "validation", "test") for r in splits[name]]
            logger.info("Replaying %d frozen matches from %s using manifest %s",
                        len(records), Path(args.db).resolve(), manifest_file)
            train_model(records, epochs=args.epochs, batch_size=args.batch_size,
                        checkpoint_path=args.output, dim=args.dim, seed=args.seed,
                        source_db=args.db, split_manifest=manifest, calibrate=args.calibrate,
                        feature_version=args.feature_version, architecture=args.architecture,
                        use_pcgrad=not args.no_pcgrad)
        else:
            strategy = args.split_strategy or "chronological"
            if args.feature_version in ("v3_enriched", "deepsets", "deepsets_v2"):
                from core.point_in_time_dataset import load_enriched_records
                warmup = min(3000, max(500, args.samples // 3))
                records = load_enriched_records(db_path=args.db, limit=args.samples, warmup_matches=warmup)
            else:
                records = fetch_training_records(limit=args.samples, db_path=args.db)
            logger.info("Using %d eligible matches from %s (split_strategy=%s)",
                        len(records), Path(args.db).resolve(), strategy)
            train_model(records, epochs=args.epochs, batch_size=args.batch_size,
                        checkpoint_path=args.output, dim=args.dim, seed=args.seed,
                        source_db=args.db, split_strategy=strategy, calibrate=args.calibrate,
                        feature_version=args.feature_version, architecture=args.architecture,
                        use_pcgrad=not args.no_pcgrad)
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        parser.exit(2, f"Training aborted: {exc}\n")


if __name__ == "__main__":
    main()
