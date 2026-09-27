"""Evaluate an anti-symmetric ensemble blending Equivariant DeepSets with GBDT."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import pickle
import sys
from uuid import uuid4
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.deepsets_layers import SiameseTeamEncoder, AntiSymmetricSoftmax
from core.equivariant_model import ThreeClassEquivariantSoftmax
from core.experiment import (
    compute_enriched_dataset_sha256, read_feature_contract, read_manifest,
    resolve_result_end_times, restore_splits,
)
from core.point_in_time_dataset import load_enriched_records
from core.deepsets_dataset import batch_extract_team_tensors
from core.features.ordered_matchup import extract_ordered_features
from core.calibration import apply_temperature_to_probabilities, sha256_file
from core.training_metrics import evaluate_winner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_ensemble")


def _model_input_policy(model) -> tuple[int, str]:
    """Inspect named inputs, irrespective of their Keras list order."""
    dimensions = {tensor.name.split(":")[0].split("/")[-1]: tuple(tensor.shape)
                  for tensor in model.inputs}
    names = set(dimensions)
    if names not in ({"team_a", "team_b"}, {"team_a", "team_b", "macro_clash"}):
        raise ValueError(f"unsupported neural inputs: {sorted(names)}")
    team_dims = [dimensions[name] for name in ("team_a", "team_b")]
    if any(len(shape) != 3 or shape[1] != 5 for shape in team_dims):
        raise ValueError("team inputs must have shape (N, 5, player_feature_dim)")
    if team_dims[0][-1] != team_dims[1][-1]:
        raise ValueError("team input dimensions disagree")
    dim = team_dims[0][-1]
    if dim not in (8, 14):
        raise ValueError(f"unsupported player feature dimension: {dim}")
    if "macro_clash" in names:
        if dim != 14 or len(dimensions["macro_clash"]) != 2 or dimensions["macro_clash"][-1] != 15:
            raise ValueError("macro_clash requires a 14-dimensional v3 model and 15 macro features")
        return 14, "v3"
    return int(dim), "tensors"


def _restore_cached_splits(cache_path: Path, manifest: dict) -> dict:
    with cache_path.open("rb") as handle:
        cached = pickle.load(handle)
    if not isinstance(cached, dict) or set(cached) != {"train", "validation", "test"}:
        raise ValueError("splits cache must contain train, validation, and test")
    return restore_splits([row for name in ("train", "validation", "test")
                           for row in cached[name]], manifest)


def evaluate_ensemble(
    model_path: str,
    db_path: str,
    samples: int = 45000,
    *,
    splits_cache: str | None = None,
    result_end_times: str | None = None,
    missing_result_end: str = "error",
    output_dir: str | None = None,
) -> dict:
    if output_dir is not None and Path(output_dir).exists():
        raise FileExistsError(f"ensemble output directory already exists: {output_dir}")
    checkpoint = Path(model_path).resolve()
    manifest_path = checkpoint.with_suffix(".split.json")
    calib_path = checkpoint.with_suffix(".calibration.json")
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = read_manifest(manifest_path)
    total_split_samples = sum(len(v) for v in manifest["splits"].values())
    effective_samples = max(samples, total_split_samples)
    cache_path = Path(splits_cache).resolve() if splits_cache is not None else checkpoint.parent / f"splits_cache_{effective_samples}.pkl"
    cache_was_loaded = cache_path.exists()
    if cache_was_loaded:
        logger.info("Loading cached splits from %s...", cache_path)
        splits = _restore_cached_splits(cache_path, manifest)
    else:
        if splits_cache is not None:
            raise FileNotFoundError(f"Explicit splits cache not found: {cache_path}")
        warmup = min(3000, max(500, effective_samples // 3))
        logger.info("Loading %d records (warmup=%d) from %s to match split manifest...", effective_samples, warmup, db_path)
        records = load_enriched_records(db_path=db_path, limit=effective_samples, warmup_matches=warmup)
        splits = restore_splits(records, manifest)
    
    logger.info("Restored splits: %s", {k: len(v) for k, v in splits.items()})

    # 1. Evaluate Equivariant DeepSets Neural Model
    logger.info("Evaluating Equivariant DeepSets model from %s...", checkpoint)
    custom_objects = {
        "SiameseTeamEncoder": SiameseTeamEncoder,
        "AntiSymmetricSoftmax": AntiSymmetricSoftmax,
        "ThreeClassEquivariantSoftmax": ThreeClassEquivariantSoftmax,
    }
    model = tf.keras.models.load_model(checkpoint, custom_objects=custom_objects)
    
    in_dim, representation = _model_input_policy(model)
    contract = None
    feature_contract_sha = None
    if in_dim == 14:
        try:
            contract = read_feature_contract(checkpoint)
        except FileNotFoundError as exc:
            raise ValueError("14-dimensional checkpoint has no ordered feature contract; retraining is required") from exc
        if contract["representation"] != representation:
            raise ValueError("model input representation disagrees with feature contract")
        if contract["split_sha256"] != manifest["split_sha256"]:
            raise ValueError("feature contract split_sha256 disagrees with manifest")
        if contract["enriched_dataset_sha256"] != compute_enriched_dataset_sha256(splits):
            raise ValueError("feature contract enriched_dataset_sha256 disagrees with restored splits")
        if cache_was_loaded and contract.get("source_cache_sha256"):
            if sha256_file(cache_path) != contract["source_cache_sha256"]:
                raise ValueError("feature contract source_cache_sha256 mismatch")
        if contract["missing_policy"] != missing_result_end:
            raise ValueError("requested missing-result-end policy disagrees with feature contract")
        end_path = Path(result_end_times) if result_end_times is not None else checkpoint.with_suffix(".result_end_times.json")
        if not end_path.exists():
            raise FileNotFoundError(f"ordered result end artifact not found: {end_path}")
        if sha256_file(end_path) != contract["result_end_times_sha256"]:
            raise ValueError("feature contract result_end_times_sha256 mismatch")
        ends, provenance = resolve_result_end_times(splits, artifact_path=end_path, missing=missing_result_end)
        if provenance["availability_assumption"] != contract["availability_assumption"]:
            raise ValueError("feature contract availability_assumption mismatch")
        x_nn, _, nn_diagnostics = extract_ordered_features(
            splits, ends, representation=representation,
            matchup_weight=contract["matchup_weight"],
        )
        if nn_diagnostics != contract["diagnostics"]:
            raise ValueError("feature contract extraction diagnostics mismatch")
        feature_contract_sha = sha256_file(checkpoint.with_suffix(".features.json"))
    else:
        logger.info("Detected 8-dimensional legacy player tensor model")
        x_nn = {name: batch_extract_team_tensors([r["feature_json"] for r in rows])
                for name, rows in splits.items()}
        ends, provenance = resolve_result_end_times(
            splits, db_path=db_path, artifact_path=result_end_times,
            missing=missing_result_end,
        )
        nn_diagnostics = None

    # Tree features use a separately recorded weight and the same causal ends.
    x_tab, x_tab_swapped, tree_diagnostics = extract_ordered_features(
        splits, ends, representation="tabular", matchup_weight=15.0,
    )

    y = {name: np.asarray([r["labels_json"]["winner"] for r in rows], dtype=int)
         for name, rows in splits.items()}
    train_rate = float(y["train"].mean())

    nn_raw_probs = {}
    for name in ("train", "validation", "test"):
        preds = model.predict(x_nn[name], batch_size=128, verbose=0)
        p = np.asarray(preds["winner"])[:, 1]
        nn_raw_probs[name] = p

    # Apply calibration if available
    temp = 1.0
    if calib_path.exists():
        with open(calib_path, "r", encoding="utf-8") as f:
            calib_data = json.load(f)
        if contract is not None and (not isinstance(calib_data, dict) or "temperature" not in calib_data):
            raise ValueError("ordered checkpoint calibration missing temperature")
        if contract is not None and calib_data.get("feature_contract_sha256") != feature_contract_sha:
            raise ValueError("calibration feature_contract_sha256 mismatch")
        temp = float(calib_data.get("temperature", 1.0))
        logger.info("Loaded temperature calibration: T=%.4f", temp)
        nn_probs = {name: apply_temperature_to_probabilities(p, temp) for name, p in nn_raw_probs.items()}
    else:
        if contract is not None:
            logger.info("No calibration sidecar for ordered checkpoint; evaluating raw probabilities")
        nn_probs = nn_raw_probs

    run_dir = Path(output_dir) if output_dir is not None else REPO_ROOT / "runs" / f"ensemble_{uuid4().hex[:12]}"
    run_dir.mkdir(parents=True, exist_ok=False)

    # 2. Train and Symmetrize Tree Models on Tabular Enriched Domain Features
    logger.info("Using ordered tabular domain features (232-dim) for tree models")

    # Data augmentation: Team swap symmetrization (doubles training size and forces exact anti-symmetry)
    logger.info("Creating symmetrized training pairs for tree models...")
    x_tab_aug = np.concatenate([x_tab["train"], x_tab_swapped["train"]], axis=0)
    y_aug = np.concatenate([y["train"], 1 - y["train"]], axis=0)

    # Function to get strictly anti-symmetric predictions from any tree model: 0.5 * (P(B|A,B) + 1 - P(B|B,A))
    def _predict_symmetrized(model, split_name: str) -> np.ndarray:
        p1 = model.predict_proba(x_tab[split_name])[:, 1]
        p2 = model.predict_proba(x_tab_swapped[split_name])[:, 1]
        return 0.5 * (p1 + (1.0 - p2))

    # A. HistGradientBoostingClassifier
    logger.info("Training HistGradientBoostingClassifier on symmetrized data (%d samples)...", len(y_aug))
    gbdt = HistGradientBoostingClassifier(
        categorical_features=list(range(10)),
        random_state=42,
        max_iter=200,
        early_stopping=True,
    )
    gbdt.fit(x_tab_aug, y_aug)
    gbdt_probs = {name: _predict_symmetrized(gbdt, name) for name in ("train", "validation", "test")}

    # B. LightGBM Classifier
    logger.info("Training LightGBM on symmetrized data (%d samples)...", len(y_aug))
    import lightgbm as lgb
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=45,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )
    lgb_model.fit(x_tab_aug, y_aug)
    lgb_probs = {name: _predict_symmetrized(lgb_model, name) for name in ("train", "validation", "test")}

    # 3. Optimize Blending Weights on Validation Split (Tri-blend: NN, LGBM, GBDT)
    val_y = y["validation"]
    best_weights = (1.0, 0.0, 0.0)
    best_val_log_loss = float("inf")

    # Grid search over simplex (w_nn + w_lgb + w_gbdt = 1.0)
    for w_nn in np.linspace(0.0, 1.0, 21):
        for w_lgb in np.linspace(0.0, 1.0 - w_nn, 21):
            w_gbdt = max(0.0, 1.0 - w_nn - w_lgb)
            blend_val = (w_nn * nn_probs["validation"] +
                         w_lgb * lgb_probs["validation"] +
                         w_gbdt * gbdt_probs["validation"])
            eval_res = evaluate_winner(val_y, blend_val, train_rate)
            if eval_res["model"]["log_loss"] < best_val_log_loss:
                best_val_log_loss = eval_res["model"]["log_loss"]
                best_weights = (float(w_nn), float(w_lgb), float(w_gbdt))

    logger.info("Optimal validation blending weights: w_nn=%.2f, w_lgb=%.2f, w_gbdt=%.2f (val_log_loss=%.4f)",
                best_weights[0], best_weights[1], best_weights[2], best_val_log_loss)

    # 4. Evaluate Test Metrics for all components and ensembles
    test_y = y["test"]
    nn_test_eval = evaluate_winner(test_y, nn_probs["test"], train_rate, diagnostics=True)
    gbdt_test_eval = evaluate_winner(test_y, gbdt_probs["test"], train_rate, diagnostics=True)
    lgb_test_eval = evaluate_winner(test_y, lgb_probs["test"], train_rate, diagnostics=True)

    w_nn, w_lgb, w_gbdt = best_weights
    ensemble_test_probs = (w_nn * nn_probs["test"] +
                           w_lgb * lgb_probs["test"] +
                           w_gbdt * gbdt_probs["test"])
    ensemble_test_eval = evaluate_winner(test_y, ensemble_test_probs, train_rate, diagnostics=True)

    # 2-way blend: DeepSets (0.65) + LightGBM (0.35)
    nn_lgb_probs = 0.65 * nn_probs["test"] + 0.35 * lgb_probs["test"]
    nn_lgb_eval = evaluate_winner(test_y, nn_lgb_probs, train_rate, diagnostics=True)

    # 5. Confidence Stratification (Accuracy by conviction margin)
    stratification = {}
    for margin in (0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15):
        mask = np.abs(ensemble_test_probs - 0.5) >= margin
        if mask.any():
            acc = float(np.mean((ensemble_test_probs[mask] >= 0.5) == test_y[mask]))
            stratification[f"conviction_ge_{int(margin*100)}pct"] = {
                "coverage_pct": round(float(mask.mean() * 100), 1),
                "count": int(mask.sum()),
                "accuracy": round(acc, 4),
            }

    report = {
        "artifact_scope": "training_evaluation",
        "model_path": str(checkpoint),
        "model_sha256": sha256_file(checkpoint),
        "manifest_sha256": sha256_file(manifest_path),
        "calibration_sha256": sha256_file(calib_path) if calib_path.exists() else None,
        "calibration_applied": calib_path.exists(),
        "feature_contract_sha256": feature_contract_sha,
        "source_cache_sha256": sha256_file(cache_path) if cache_was_loaded else None,
        "result_end_times_sha256": sha256_file(end_path) if contract is not None else (
            sha256_file(result_end_times) if result_end_times is not None else None
        ),
        "enriched_dataset_sha256": compute_enriched_dataset_sha256(splits),
        "result_end_provenance": provenance,
        "neural_matchup_diagnostics": nn_diagnostics,
        "tree_matchup_diagnostics": tree_diagnostics,
        "neural_matchup_weight": contract["matchup_weight"] if contract is not None else None,
        "tree_matchup_weight": 15.0,
        "split_counts": {k: len(v) for k, v in splits.items()},
        "best_weights": {"nn": w_nn, "lgb": w_lgb, "gbdt": w_gbdt},
        "deepsets_metrics": {
            "accuracy": nn_test_eval["model"]["accuracy"],
            "accuracy_lift": nn_test_eval["accuracy_lift"],
            "roc_auc": nn_test_eval["model"]["roc_auc"],
            "log_loss": nn_test_eval["model"]["log_loss"],
            "brier": nn_test_eval["model"]["brier"],
        },
        "lightgbm_metrics": {
            "accuracy": lgb_test_eval["model"]["accuracy"],
            "accuracy_lift": lgb_test_eval["accuracy_lift"],
            "roc_auc": lgb_test_eval["model"]["roc_auc"],
            "log_loss": lgb_test_eval["model"]["log_loss"],
            "brier": lgb_test_eval["model"]["brier"],
        },
        "gbdt_metrics": {
            "accuracy": gbdt_test_eval["model"]["accuracy"],
            "accuracy_lift": gbdt_test_eval["accuracy_lift"],
            "roc_auc": gbdt_test_eval["model"]["roc_auc"],
            "log_loss": gbdt_test_eval["model"]["log_loss"],
            "brier": gbdt_test_eval["model"]["brier"],
        },
        "deepsets_plus_lightgbm": {
            "accuracy": nn_lgb_eval["model"]["accuracy"],
            "accuracy_lift": nn_lgb_eval["accuracy_lift"],
            "roc_auc": nn_lgb_eval["model"]["roc_auc"],
            "log_loss": nn_lgb_eval["model"]["log_loss"],
            "brier": nn_lgb_eval["model"]["brier"],
        },
        "tri_blend_ensemble_metrics": {
            "accuracy": ensemble_test_eval["model"]["accuracy"],
            "accuracy_lift": ensemble_test_eval["accuracy_lift"],
            "roc_auc": ensemble_test_eval["model"]["roc_auc"],
            "log_loss": ensemble_test_eval["model"]["log_loss"],
            "brier": ensemble_test_eval["model"]["brier"],
            "calibration_error": ensemble_test_eval["model"]["calibration_error"],
        },
        "confidence_stratification": stratification,
        "baseline_accuracy": nn_test_eval["baseline"]["accuracy"],
    }

    out_file = run_dir / "ensemble_comparison.json"
    with open(out_file, "x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved ensemble comparison report to %s", out_file)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--samples", type=int, default=45000)
    parser.add_argument("--splits-cache")
    parser.add_argument("--result-end-times")
    parser.add_argument("--missing-result-end", choices=("error", "exclude"), default="error")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    evaluate_ensemble(
        args.model, args.db, args.samples, splits_cache=args.splits_cache,
        result_end_times=args.result_end_times,
        missing_result_end=args.missing_result_end, output_dir=args.output_dir,
    )
