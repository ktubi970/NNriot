"""Chronological evaluation of a fixed-candidate retrospective logistic model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import AnalyticsConfig
from .db import get_meta
from .model_data import ModelDataset, load_model_dataset
from .splits import chronological_split, player_disjoint_split


INTERACTIONS = (
    ("gold_delta", "kills_delta"),
    ("vision_score_delta", "dragons_delta"),
    ("lane_cs_delta", "towers_delta"),
)
CANDIDATE_C = (0.01, 0.1, 1.0, 10.0)


@dataclass(frozen=True)
class ModelReport:
    dataset_version: str
    retrospective: bool
    seed: int
    bootstrap_resamples: int
    selected_c: float | None
    split_counts: dict[str, int]
    metrics: dict[str, dict[str, Any]]
    coefficients: dict[str, float]
    odds_ratios: dict[str, float]
    coefficient_intervals: dict[str, list[float]]
    direction_agreement: dict[str, str]
    player_disjoint: dict[str, Any]
    final_test: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= left) & (probabilities < right)
        if right == 1.0:
            mask |= probabilities == 1.0
        if mask.any():
            error += mask.mean() * abs(
                probabilities[mask].mean() - y_true[mask].mean()
            )
    return float(error)


def evaluate_probabilities(
    y_true: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        raise ValueError("INSUFFICIENT_MATCHES")
    return {
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "calibration_error": calibration_error(y_true, probabilities),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def _numeric_design(dataset: ModelDataset) -> tuple[np.ndarray, tuple[str, ...]]:
    columns = [dataset.x]
    names = list(dataset.feature_names)
    by_name = {name: index for index, name in enumerate(dataset.feature_names)}
    for left, right in INTERACTIONS:
        columns.append(
            (
                dataset.x[:, by_name[left]] * dataset.x[:, by_name[right]]
            )[:, None]
        )
        names.append(f"{left}*{right}")
    return np.column_stack(columns), tuple(names)


def _design(
    dataset: ModelDataset,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    numeric, numeric_names = _numeric_design(dataset)
    values = np.empty((len(dataset.y), len(numeric_names) + 2), dtype=object)
    values[:, : len(numeric_names)] = numeric
    values[:, len(numeric_names)] = dataset.regions
    values[:, len(numeric_names) + 1] = dataset.patches
    all_names = numeric_names + ("region", "patch")
    return values, numeric_names, all_names


def _pipeline(
    selected_c: float,
    seed: int,
    numeric_count: int,
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=(
            ("numeric", StandardScaler(), list(range(numeric_count))),
            (
                "controls",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                [numeric_count, numeric_count + 1],
            ),
        ),
        verbose_feature_names_out=False,
    )
    classifier = LogisticRegression(
        C=selected_c,
        penalty="l2",
        solver="liblinear",
        max_iter=2000,
        random_state=seed,
    )
    return Pipeline(
        [("preprocessor", preprocessor), ("classifier", classifier)]
    )


def _metric_intervals(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    seed: int,
    resamples: int,
) -> dict[str, dict[str, list[float]]]:
    rng = np.random.default_rng(seed)
    names = ("log_loss", "brier", "calibration_error", "roc_auc")
    draws = {
        model: {metric: [] for metric in names} for model in predictions
    }
    for _ in range(resamples):
        indices = rng.integers(0, len(y_true), len(y_true))
        sampled_y = y_true[indices]
        if len(np.unique(sampled_y)) < 2:
            continue
        for model, probabilities in predictions.items():
            metrics = evaluate_probabilities(sampled_y, probabilities[indices])
            for metric in names:
                draws[model][metric].append(metrics[metric])
    result: dict[str, dict[str, list[float]]] = {}
    for model, metrics in draws.items():
        result[model] = {}
        for metric, values in metrics.items():
            if not values:
                result[model][metric] = [None, None]
            else:
                low, high = np.quantile(values, [0.025, 0.975])
                result[model][metric] = [float(low), float(high)]
    return result


def _coefficient_intervals(
    pipeline: Pipeline,
    design: np.ndarray,
    target: np.ndarray,
    seed: int,
    resamples: int,
    feature_names: tuple[str, ...],
) -> dict[str, list[float]]:
    transformed = pipeline.named_steps["preprocessor"].transform(design)
    transformed = np.asarray(transformed, dtype=float)
    classifier = pipeline.named_steps["classifier"]
    beta = classifier.coef_[0].astype(float)
    probabilities = classifier.predict_proba(transformed)[:, 1]
    weights = probabilities * (1.0 - probabilities)
    hessian = transformed.T @ (transformed * weights[:, None])
    hessian += np.eye(transformed.shape[1]) / classifier.C
    inverse = np.linalg.pinv(hessian)
    scores = transformed * (target - probabilities)[:, None]

    rng = np.random.default_rng(seed + 10_003)
    coefficient_draws = np.empty((resamples, len(beta)), dtype=float)
    chunk_size = 32
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        size = stop - start
        indices = rng.integers(
            0, len(target), size=(size, len(target)), dtype=np.int32
        )
        score_sums = scores[indices].sum(axis=1)
        coefficient_draws[start:stop] = beta + score_sums @ inverse.T
    lower = np.quantile(coefficient_draws, 0.025, axis=0)
    upper = np.quantile(coefficient_draws, 0.975, axis=0)
    return {
        name: [float(low), float(high)]
        for name, low, high in zip(feature_names, lower, upper)
    }


def _player_disjoint_result(
    dataset: ModelDataset,
    design: np.ndarray,
    selected_c: float,
    numeric_count: int,
    seed: int,
    min_matches: int,
) -> dict[str, Any]:
    split = player_disjoint_split(dataset.match_puuids, min_matches)
    base = {
        "available": split.available,
        "code": split.code,
        "train": len(split.train),
        "test": len(split.test),
        "discarded": len(split.discarded),
    }
    if not split.available:
        return base
    if (
        len(np.unique(dataset.y[split.train])) < 2
        or len(np.unique(dataset.y[split.test])) < 2
    ):
        return {**base, "available": False, "code": "INSUFFICIENT_MATCHES"}
    model = _pipeline(selected_c, seed, numeric_count)
    model.fit(design[split.train], dataset.y[split.train])
    probabilities = model.predict_proba(design[split.test])[:, 1]
    return {
        **base,
        "metrics": evaluate_probabilities(dataset.y[split.test], probabilities),
    }


def _atomic_artifacts(
    pipeline: Pipeline, report: ModelReport, artifact_dir: str
) -> None:
    directory = Path(artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)
    model_handle, model_temp_name = tempfile.mkstemp(
        prefix=".interpretable_model.", suffix=".joblib", dir=directory
    )
    os.close(model_handle)
    report_handle, report_temp_name = tempfile.mkstemp(
        prefix=".model_report.", suffix=".json", dir=directory
    )
    os.close(report_handle)
    model_temp = Path(model_temp_name)
    report_temp = Path(report_temp_name)
    try:
        joblib.dump(pipeline, model_temp)
        report_temp.write_text(
            json.dumps(
                report.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        os.replace(model_temp, directory / "interpretable_model.joblib")
        os.replace(report_temp, directory / "model_report.json")
    finally:
        model_temp.unlink(missing_ok=True)
        report_temp.unlink(missing_ok=True)


def train_and_evaluate(
    analytics_path: str,
    artifact_dir: str,
    min_grouped_test_matches: int = 500,
) -> ModelReport:
    config = AnalyticsConfig(analytics_path=analytics_path)
    dataset = load_model_dataset(analytics_path)
    dataset_version = get_meta(analytics_path, "dataset_version")
    if not dataset_version:
        raise ValueError("ANALYTICS_DB_STALE")
    split = chronological_split(np.arange(len(dataset.y)))
    split_counts = {
        "train": len(split.train),
        "validation": len(split.validation),
        "test": len(split.test),
    }
    if len(split.test) < config.min_matches:
        return ModelReport(
            dataset_version=dataset_version,
            retrospective=True,
            seed=config.seed,
            bootstrap_resamples=config.bootstrap_resamples,
            selected_c=None,
            split_counts=split_counts,
            metrics={},
            coefficients={},
            odds_ratios={},
            coefficient_intervals={},
            direction_agreement={},
            player_disjoint={
                "available": False,
                "code": "INSUFFICIENT_MATCHES",
            },
            final_test={
                "available": False,
                "code": "INSUFFICIENT_MATCHES",
            },
        )

    design, numeric_names, input_names = _design(dataset)
    validation_results: list[tuple[float, float]] = []
    for candidate_c in CANDIDATE_C:
        candidate = _pipeline(candidate_c, config.seed, len(numeric_names))
        candidate.fit(design[split.train], dataset.y[split.train])
        probabilities = candidate.predict_proba(design[split.validation])[:, 1]
        loss = float(
            log_loss(dataset.y[split.validation], probabilities, labels=[0, 1])
        )
        validation_results.append((loss, candidate_c))
    _, selected_c = min(validation_results)

    train_validation = np.concatenate([split.train, split.validation])
    pipeline = _pipeline(selected_c, config.seed, len(numeric_names))
    pipeline.fit(design[train_validation], dataset.y[train_validation])
    logistic_probabilities = pipeline.predict_proba(design[split.test])[:, 1]
    train_mean = float(dataset.y[train_validation].mean())
    majority_probability = float(train_mean >= 0.5)
    predictions = {
        "majority": np.full(len(split.test), majority_probability, dtype=float),
        "mean_probability": np.full(len(split.test), train_mean, dtype=float),
        "logistic": logistic_probabilities,
    }
    metrics = {
        name: evaluate_probabilities(dataset.y[split.test], probabilities)
        for name, probabilities in predictions.items()
    }
    intervals = _metric_intervals(
        dataset.y[split.test],
        predictions,
        seed=config.seed,
        resamples=config.bootstrap_resamples,
    )
    for name in metrics:
        metrics[name]["intervals_95"] = intervals[name]

    preprocessor = pipeline.named_steps["preprocessor"]
    transformed_names = tuple(
        str(name) for name in preprocessor.get_feature_names_out(input_names)
    )
    raw_coefficients = pipeline.named_steps["classifier"].coef_[0]
    coefficients = {
        name: float(value) for name, value in zip(transformed_names, raw_coefficients)
    }
    odds_ratios = {
        name: float(math.exp(max(-50.0, min(50.0, value))))
        for name, value in coefficients.items()
    }
    coefficient_intervals = _coefficient_intervals(
        pipeline,
        design[train_validation],
        dataset.y[train_validation],
        config.seed,
        config.bootstrap_resamples,
        transformed_names,
    )

    orientation = dataset.y.astype(float) * 2.0 - 1.0
    direction_agreement: dict[str, str] = {}
    for index, name in enumerate(dataset.feature_names):
        descriptive = float(np.mean(dataset.x[:, index] * orientation))
        coefficient = coefficients.get(name)
        if coefficient is None or abs(descriptive) <= 1e-12:
            direction_agreement[name] = "descriptive_unavailable"
        elif coefficient * descriptive > 0:
            direction_agreement[name] = "agrees"
        else:
            direction_agreement[name] = "conditional_only"

    grouped = _player_disjoint_result(
        dataset,
        design,
        selected_c,
        len(numeric_names),
        config.seed,
        min_grouped_test_matches,
    )
    report = ModelReport(
        dataset_version=dataset_version,
        retrospective=True,
        seed=config.seed,
        bootstrap_resamples=config.bootstrap_resamples,
        selected_c=selected_c,
        split_counts=split_counts,
        metrics=metrics,
        coefficients=coefficients,
        odds_ratios=odds_ratios,
        coefficient_intervals=coefficient_intervals,
        direction_agreement=direction_agreement,
        player_disjoint=grouped,
        final_test={"available": True, "code": None},
    )
    _atomic_artifacts(pipeline, report, artifact_dir)
    return report
