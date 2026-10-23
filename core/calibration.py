"""Numeric temperature scaling and bound winner calibration artifacts."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.metrics import log_loss

from core.experiment import FEATURE_POLICY, read_manifest

logger = logging.getLogger(__name__)

EPS = 1e-12
MIN_T = 0.05
MAX_T = 10.0


class UncalibratedModelError(RuntimeError):
    """Raised when prediction is attempted on an uncalibrated model."""
    pass


def _validated_probabilities(probabilities: Any) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("probabilities must be a non-empty 1-dimensional array")
    if not np.isfinite(p).all():
        raise ValueError("probabilities must be finite")
    if (p < 0.0).any() or (p > 1.0).any():
        raise ValueError("probabilities must be within [0.0, 1.0]")
    return p


def _validated_temperature(temperature: Any) -> float:
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be a finite positive number")
    t = float(temperature)
    if not np.isfinite(t) or t <= 0.0:
        raise ValueError("temperature must be a finite positive number")
    return t


def apply_temperature_to_probabilities(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Scale positive-class probabilities P(B) with temperature T."""
    p = _validated_probabilities(probabilities)
    T = _validated_temperature(temperature)
    if T == 1.0:
        return p.copy()
    clipped = np.clip(p, EPS, 1.0 - EPS)
    logits = np.log(clipped) - np.log1p(-clipped)
    return expit(logits / T)


def fit_temperature(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    """Fit temperature scaling parameter on binary labels and predicted probabilities."""
    p = _validated_probabilities(probabilities)
    y_raw = np.asarray(labels)
    if y_raw.ndim != 1 or y_raw.size != p.size:
        raise ValueError("labels must be a 1-dimensional array matching probabilities size")
    if not np.isfinite(y_raw).all():
        raise ValueError("labels must be finite")
    if y_raw.size < 20:
        raise ValueError(f"fitting temperature requires at least 20 samples, got {y_raw.size}")

    unique_vals = set(np.unique(y_raw))
    if not unique_vals.issubset({0, 1}):
        raise ValueError("labels must contain only binary values 0 and 1")
    if unique_vals != {0, 1}:
        raise ValueError("labels must contain both classes 0 and 1 for calibration")

    y = y_raw.astype(float)
    clipped = np.clip(p, EPS, 1.0 - EPS)
    logits = np.log(clipped) - np.log1p(-clipped)

    def objective(log_temp):
        z = logits / np.exp(log_temp)
        return float(np.mean(np.logaddexp(0.0, z) - y * z))

    result = minimize_scalar(objective, bounds=(np.log(MIN_T), np.log(MAX_T)), method="bounded")
    if not result.success or not np.isfinite(result.fun):
        raise ValueError("temperature optimization failed")

    T = float(np.exp(result.x))
    before = float(log_loss(y, p, labels=[0, 1]))
    after = float(log_loss(y, apply_temperature_to_probabilities(p, T), labels=[0, 1]))
    if after >= before - 1e-12:
        T, after = 1.0, before

    boundary_hit = abs(np.log(T) - np.log(MIN_T)) < 1e-3 or abs(np.log(T) - np.log(MAX_T)) < 1e-3
    class_counts = [int(np.sum(y == 0)), int(np.sum(y == 1))]

    return {
        "status": "fitted",
        "partition": "validation",
        "count": len(y),
        "class_counts": class_counts,
        "temperature": T,
        "optimizer_success": True,
        "boundary_hit": bool(boundary_hit),
        "log_loss_before": before,
        "log_loss_after": after,
    }


def sha256_file(path: str | Path) -> str:
    """Compute streaming SHA-256 of file contents."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_calibration_artifact(
    checkpoint_path: str | Path,
    split_manifest_path: str | Path,
    fit: dict,
    *,
    input_dim: int,
) -> Path:
    """Validate fit and write a bound calibration sidecar."""
    if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim <= 0:
        raise ValueError("input_dim must be a positive integer")

    ckpt = Path(checkpoint_path).resolve()
    manifest_p = Path(split_manifest_path).resolve()
    expected_manifest_path = ckpt.with_suffix(".split.json")
    if manifest_p != expected_manifest_path:
        raise ValueError(
            f"Companion manifest must be {expected_manifest_path}, got {manifest_p}"
        )

    if not isinstance(fit, dict):
        raise ValueError("fit must be a dictionary")
    if fit.get("status") != "fitted" or fit.get("partition") != "validation":
        raise ValueError("fit must have status='fitted' and partition='validation'")
    if fit.get("optimizer_success") is not True:
        raise ValueError("fit optimizer_success must be True")

    count = fit.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 20:
        raise ValueError("fit count must be an integer >= 20")

    class_counts = fit.get("class_counts")
    if not (
        isinstance(class_counts, list)
        and len(class_counts) == 2
        and all(isinstance(c, int) and not isinstance(c, bool) and c > 0 for c in class_counts)
        and sum(class_counts) == count
    ):
        raise ValueError("fit class_counts must be two positive integers summing to count")

    temp = fit.get("temperature")
    if (
        isinstance(temp, bool)
        or not isinstance(temp, (int, float))
        or not (MIN_T <= temp <= MAX_T)
        or not np.isfinite(temp)
    ):
        raise ValueError(f"temperature must be in [{MIN_T}, {MAX_T}]")

    before = fit.get("log_loss_before")
    after = fit.get("log_loss_after")
    if (
        isinstance(before, bool)
        or not isinstance(before, (int, float))
        or not np.isfinite(before)
        or before < 0.0
    ):
        raise ValueError("log_loss_before must be a non-negative finite float")
    if (
        isinstance(after, bool)
        or not isinstance(after, (int, float))
        or not np.isfinite(after)
        or after < 0.0
        or after > before + 1e-12
    ):
        raise ValueError("log_loss_after must be non-negative, finite, and <= log_loss_before")

    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt}")
    if not manifest_p.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_p}")

    manifest_data = read_manifest(manifest_p)
    manifest_val_rows = len(manifest_data["splits"]["validation"])
    if manifest_val_rows != count:
        raise ValueError(
            f"fit count ({count}) does not match manifest validation count ({manifest_val_rows})"
        )

    ckpt_sha = sha256_file(ckpt)
    manifest_file_sha = sha256_file(manifest_p)

    payload = {
        "schema_version": 1,
        "method": "binary_temperature",
        "head": "winner",
        "class_order": ["A", "B"],
        "positive_class": 1,
        "feature_policy": FEATURE_POLICY,
        "input_dim": input_dim,
        "checkpoint_sha256": ckpt_sha,
        "split_manifest_sha256": manifest_file_sha,
        "dataset_sha256": manifest_data["dataset_sha256"],
        "split_sha256": manifest_data["split_sha256"],
        "temperature": temp,
        "fit": fit,
    }

    sidecar_path = ckpt.with_suffix(".calibration.json")
    with sidecar_path.open("x", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")

    return sidecar_path


def load_calibration_artifact(checkpoint_path: str | Path, *, input_dim: int) -> dict | None:
    """Load and strictly verify a bound calibration sidecar."""
    try:
        if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim <= 0:
            logger.warning("Invalid input_dim specified for calibration verification: %s", input_dim)
            return None

        ckpt = Path(checkpoint_path).resolve()
        sidecar = ckpt.with_suffix(".calibration.json")
        if not sidecar.exists():
            logger.info("No calibration sidecar found at %s", sidecar)
            return None
        if not ckpt.exists():
            logger.warning("Checkpoint file not found: %s", ckpt)
            return None

        with sidecar.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            logger.warning("Calibration sidecar is not a JSON object")
            return None

        if data.get("schema_version") != 1:
            logger.warning("Unsupported calibration schema_version: %s", data.get("schema_version"))
            return None
        if data.get("method") != "binary_temperature":
            logger.warning("Unsupported calibration method: %s", data.get("method"))
            return None
        if data.get("head") != "winner":
            logger.warning("Unsupported calibration head: %s", data.get("head"))
            return None
        if data.get("class_order") != ["A", "B"]:
            logger.warning("Unsupported class_order: %s", data.get("class_order"))
            return None
        if data.get("positive_class") != 1 or isinstance(data.get("positive_class"), bool):
            logger.warning("Unsupported positive_class: %s", data.get("positive_class"))
            return None
        if data.get("feature_policy") != FEATURE_POLICY:
            logger.warning("Unsupported feature_policy: %s", data.get("feature_policy"))
            return None
        if data.get("input_dim") != input_dim or isinstance(data.get("input_dim"), bool):
            logger.warning("input_dim mismatch: expected %s, got %s", input_dim, data.get("input_dim"))
            return None

        actual_ckpt_sha = sha256_file(ckpt)
        if data.get("checkpoint_sha256") != actual_ckpt_sha:
            logger.warning("Checkpoint SHA-256 mismatch with calibration artifact")
            return None

        manifest_path = ckpt.with_suffix(".split.json")
        if not manifest_path.exists():
            logger.warning("Companion split manifest not found: %s", manifest_path)
            return None

        actual_manifest_sha = sha256_file(manifest_path)
        if data.get("split_manifest_sha256") != actual_manifest_sha:
            logger.warning("Split manifest file SHA-256 mismatch with calibration artifact")
            return None

        manifest_data = read_manifest(manifest_path)
        if manifest_data.get("dataset_sha256") != data.get("dataset_sha256"):
            logger.warning("Dataset SHA-256 mismatch between manifest and calibration artifact")
            return None
        if manifest_data.get("split_sha256") != data.get("split_sha256"):
            logger.warning("Split SHA-256 mismatch between manifest and calibration artifact")
            return None

        temp = data.get("temperature")
        if (
            isinstance(temp, bool)
            or not isinstance(temp, (int, float))
            or not (MIN_T <= temp <= MAX_T)
            or not np.isfinite(temp)
        ):
            logger.warning("Invalid temperature in calibration artifact: %s", temp)
            return None

        fit = data.get("fit")
        if not isinstance(fit, dict):
            logger.warning("Missing or invalid fit dictionary in calibration artifact")
            return None
        if fit.get("status") != "fitted" or fit.get("partition") != "validation":
            logger.warning("Invalid fit status or partition: %s", fit)
            return None
        if fit.get("optimizer_success") is not True:
            logger.warning("Fit optimizer_success must be True in calibration artifact")
            return None

        count = fit.get("count")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 20
            or count != len(manifest_data["splits"]["validation"])
        ):
            logger.warning("Invalid fit sample count: %s", count)
            return None

        class_counts = fit.get("class_counts")
        if not (
            isinstance(class_counts, list)
            and len(class_counts) == 2
            and all(isinstance(c, int) and not isinstance(c, bool) and c > 0 for c in class_counts)
            and sum(class_counts) == count
        ):
            logger.warning("Invalid fit class_counts: %s", class_counts)
            return None

        before = fit.get("log_loss_before")
        after = fit.get("log_loss_after")
        if (
            isinstance(before, bool)
            or not isinstance(before, (int, float))
            or not np.isfinite(before)
            or before < 0.0
            or isinstance(after, bool)
            or not isinstance(after, (int, float))
            or not np.isfinite(after)
            or after < 0.0
            or after > before + 1e-12
        ):
            logger.warning("Invalid log_loss before/after in calibration artifact")
            return None

        return data
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load calibration artifact: %s", exc)
        return None


def invalidate_calibration(model: Any, checkpoint_path: str | Path) -> None:
    """Clear in-memory attributes and delete calibration sidecar on disk."""
    model.is_calibrated = False
    model.calibration_temperature = None
    model.calibration_metadata = None
    sidecar = Path(checkpoint_path).with_suffix(".calibration.json")
    if sidecar.exists():
        sidecar.unlink()
