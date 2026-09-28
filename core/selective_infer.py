"""Selective Inference Wrapper & Decision Tiers.

Implements Chow's rejection option wrapper for single and batch match inferences.
Categorizes predictions into three actionable conviction tiers:
- HIGH_CONFIDENCE: Conviction C >= 0.08 (>= 8%), is_actionable=True
- ACTIONABLE: Conviction C >= 0.04 (>= 4%), is_actionable=True
- COINFLIP: Conviction C < 0.04 (< 4%), is_actionable=False, prediction="abstain"

Boundaries use six-decimal conviction and a 1e-9 comparison tolerance.
Tier names alone provide no guarantee of predictive accuracy.

Strictly guarantees anti-symmetry under team swap:
Swapping team A and team B (P -> 1 - P) flips prediction ("team_a" <-> "team_b")
while preserving conviction, tier, and is_actionable identically.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, NamedTuple, Sequence
import numpy as np

from core.selective_classification import compute_conviction, conviction_at_least

DEFAULT_ACTION_THRESHOLD: float = 0.04
DEFAULT_HIGH_THRESHOLD: float = 0.08

TIER_HIGH_CONFIDENCE: str = "HIGH_CONFIDENCE"
TIER_ACTIONABLE: str = "ACTIONABLE"
TIER_COINFLIP: str = "COINFLIP"

PRED_TEAM_A: str = "team_a"
PRED_TEAM_B: str = "team_b"
PRED_ABSTAIN: str = "abstain"


class SelectiveInferenceResult(NamedTuple):
    """Result of selective classification with conviction-gated tiers."""

    p_team_a: float
    p_team_b: float
    conviction: float
    prediction: str
    tier: str
    is_actionable: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to a dictionary."""
        return {
            "p_team_a": self.p_team_a,
            "p_team_b": self.p_team_b,
            "conviction": self.conviction,
            "prediction": self.prediction,
            "tier": self.tier,
            "is_actionable": self.is_actionable,
        }


def classify_match_conviction(
    prob_team_b: float,
    action_threshold: float = DEFAULT_ACTION_THRESHOLD,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
) -> SelectiveInferenceResult:
    """Classify match probability into actionable conviction tiers.

    Conviction metric C = |P(team_b) - 0.5|. Under team swap,
    P(team_a) = 1 - P(team_b), so |P(team_a) - 0.5| = |0.5 - P(team_b)| = C.

    Args:
        prob_team_b: Predicted probability of team B winning in [0.0, 1.0].
        action_threshold: Conviction threshold for actionability (default 0.04 = 4%).
        high_threshold: Conviction threshold for high confidence (default 0.08 = 8%).

    Returns:
        SelectiveInferenceResult namedtuple with p_team_a, p_team_b, conviction,
        prediction, tier, and is_actionable.
    """
    if action_threshold < 0.0 or high_threshold < 0.0:
        raise ValueError("Thresholds must be non-negative")
    if action_threshold > high_threshold:
        raise ValueError("action_threshold cannot exceed high_threshold")

    try:
        p_val = float(prob_team_b)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid probability value: {prob_team_b}") from exc

    if math.isnan(p_val) or np.isnan(p_val):
        raise ValueError("Probability contains NaN")

    if p_val < -1e-5 or p_val > 1.0 + 1e-5:
        raise ValueError(f"Probabilities must be within [0, 1], got {p_val}")

    p_team_b = float(np.clip(p_val, 0.0, 1.0))
    p_team_a = float(np.clip(1.0 - p_team_b, 0.0, 1.0))

    # Canonicalize before rounding so half-unit boundaries also survive team swap.
    conviction = round(float(compute_conviction(p_team_b)), 6)

    if conviction_at_least(conviction, high_threshold):
        tier = TIER_HIGH_CONFIDENCE
        is_actionable = True
        if p_team_b > 0.5:
            prediction = PRED_TEAM_B
        elif p_team_b < 0.5:
            prediction = PRED_TEAM_A
        else:
            prediction = PRED_ABSTAIN
    elif conviction_at_least(conviction, action_threshold):
        tier = TIER_ACTIONABLE
        is_actionable = True
        if p_team_b > 0.5:
            prediction = PRED_TEAM_B
        elif p_team_b < 0.5:
            prediction = PRED_TEAM_A
        else:
            prediction = PRED_ABSTAIN
    else:
        tier = TIER_COINFLIP
        is_actionable = False
        prediction = PRED_ABSTAIN

    return SelectiveInferenceResult(
        p_team_a=p_team_a,
        p_team_b=p_team_b,
        conviction=conviction,
        prediction=prediction,
        tier=tier,
        is_actionable=is_actionable,
    )


def classify_batch_conviction(
    probs_team_b: Sequence[float] | np.ndarray,
    action_threshold: float = DEFAULT_ACTION_THRESHOLD,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
) -> list[SelectiveInferenceResult]:
    """Classify a sequence of probabilities into SelectiveInferenceResult instances."""
    return [
        classify_match_conviction(
            p,
            action_threshold=action_threshold,
            high_threshold=high_threshold,
        )
        for p in probs_team_b
    ]


def format_batch_summary(results: Sequence[SelectiveInferenceResult]) -> dict[str, Any]:
    """Format aggregate statistics and coverage summary for a batch of predictions."""
    total = len(results)
    if total == 0:
        return {
            "total_matches": 0,
            "actionable_count": 0,
            "actionable_coverage": 0.0,
            "high_confidence_count": 0,
            "high_confidence_coverage": 0.0,
            "actionable_tier_count": 0,
            "actionable_tier_coverage": 0.0,
            "coinflip_count": 0,
            "coinflip_ratio": 0.0,
            "team_a_picks": 0,
            "team_b_picks": 0,
            "abstain_picks": 0,
        }

    actionable_count = sum(1 for r in results if r.is_actionable)
    high_count = sum(1 for r in results if r.tier == TIER_HIGH_CONFIDENCE)
    actionable_tier_count = sum(1 for r in results if r.tier == TIER_ACTIONABLE)
    coinflip_count = sum(1 for r in results if r.tier == TIER_COINFLIP)

    team_a_picks = sum(1 for r in results if r.prediction == PRED_TEAM_A)
    team_b_picks = sum(1 for r in results if r.prediction == PRED_TEAM_B)
    abstain_picks = sum(1 for r in results if r.prediction == PRED_ABSTAIN)

    return {
        "total_matches": total,
        "actionable_count": actionable_count,
        "actionable_coverage": float(actionable_count) / float(total),
        "high_confidence_count": high_count,
        "high_confidence_coverage": float(high_count) / float(total),
        "actionable_tier_count": actionable_tier_count,
        "actionable_tier_coverage": float(actionable_tier_count) / float(total),
        "coinflip_count": coinflip_count,
        "coinflip_ratio": float(coinflip_count) / float(total),
        "team_a_picks": team_a_picks,
        "team_b_picks": team_b_picks,
        "abstain_picks": abstain_picks,
    }


def predict_selective(
    model_or_checkpoint: Any,
    match_input: dict[str, Any],
    action_threshold: float = DEFAULT_ACTION_THRESHOLD,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
    tracker: Any = None,
) -> SelectiveInferenceResult:
    """Run selective inference on a single match input using a model or checkpoint.

    Args:
        model_or_checkpoint: Keras Model, checkpoint path (str/Path), or callable predictor.
        match_input: Dict containing pre-match details e.g. {"team_a": [...], "team_b": [...]}.
        action_threshold: Conviction threshold for actionability (default 0.04).
        high_threshold: Conviction threshold for high confidence (default 0.08).
        tracker: Optional PlayerPriorsTracker for player priors extraction.

    Returns:
        SelectiveInferenceResult with tier classification and actionability.
    """
    prob_team_b = _extract_probability_from_model(
        model_or_checkpoint=model_or_checkpoint,
        match_inputs=[match_input],
        tracker=tracker,
    )[0]

    return classify_match_conviction(
        prob_team_b,
        action_threshold=action_threshold,
        high_threshold=high_threshold,
    )


def batch_predict_selective(
    model_or_checkpoint: Any,
    matches: Sequence[dict[str, Any]],
    action_threshold: float = DEFAULT_ACTION_THRESHOLD,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
    tracker: Any = None,
) -> list[SelectiveInferenceResult]:
    """Run selective inference on a batch of matches using a model or checkpoint."""
    if not matches:
        return []

    probs_team_b = _extract_probability_from_model(
        model_or_checkpoint=model_or_checkpoint,
        match_inputs=matches,
        tracker=tracker,
    )

    return classify_batch_conviction(
        probs_team_b,
        action_threshold=action_threshold,
        high_threshold=high_threshold,
    )


def _extract_probability_from_model(
    model_or_checkpoint: Any,
    match_inputs: Sequence[dict[str, Any]],
    tracker: Any = None,
) -> list[float]:
    """Internal helper to extract team B win probabilities from model or callable."""
    # 1. Handle checkpoint path
    model = model_or_checkpoint
    if isinstance(model, (str, Path)):
        import tensorflow as tf
        try:
            import core.deepsets_layers  # noqa: F401
            import core.equivariant_model  # noqa: F401
        except ImportError:
            pass
        model = tf.keras.models.load_model(str(model), compile=False)

    def _extract_probs(winner_out: Any) -> list[float]:
        arr = np.asarray(winner_out)
        if arr.ndim >= 2 and arr.shape[1] > 1:
            probs = arr[:, 1]
        else:
            probs = arr.ravel()
        return [float(p) for p in probs]

    # 2. Check if model is a custom callable / mock
    if callable(model) and not hasattr(model, "layers") and not hasattr(model, "predict_on_batch"):
        results = []
        for m in match_inputs:
            out = model(m)
            if isinstance(out, dict) and "winner" in out:
                w = np.asarray(out["winner"])
                prob = float(w[0, 1] if (w.ndim >= 2 and w.shape[1] > 1) else w[1] if len(w) > 1 else w.ravel()[0])
            elif isinstance(out, (list, tuple, np.ndarray)):
                arr = np.asarray(out)
                prob = float(arr[0, 1] if (arr.ndim >= 2 and arr.shape[1] > 1) else arr[1] if len(arr) > 1 else arr.ravel()[0])
            else:
                prob = float(out)
            results.append(prob)
        return results

    # 3. Model is a Keras Model or object with predict_on_batch / predict
    from core.deepsets_dataset import batch_extract_team_tensors

    input_keys: list[str] = []
    if isinstance(getattr(model, "input", None), dict):
        input_keys = list(model.input.keys())
    elif hasattr(model, "inputs") and model.inputs:
        input_keys = [getattr(inp, "name", "") for inp in model.inputs]
    elif hasattr(model, "input_names") and model.input_names:
        input_keys = list(model.input_names)

    if any("team_a" in k for k in input_keys) and any("team_b" in k for k in input_keys):
        batch_tensors = batch_extract_team_tensors(match_inputs, tracker=tracker)
        if hasattr(model, "predict_on_batch"):
            preds = model.predict_on_batch(batch_tensors)
        else:
            preds = model(batch_tensors, training=False)

        winner_out = preds["winner"] if isinstance(preds, dict) else preds
        return _extract_probs(winner_out)

    # 4. Fallback for dense or other input shapes
    if hasattr(model, "predict"):
        from core.draft_features import extract_dense_draft_features
        from core.pre_match_dataset import to_pre_match_features

        features_list = [
            extract_dense_draft_features(to_pre_match_features(m))
            for m in match_inputs
        ]
        x_dense = np.stack(features_list)
        preds = model.predict(x_dense, verbose=0)
        winner_out = preds["winner"] if isinstance(preds, dict) else preds
        return _extract_probs(winner_out)

    raise TypeError(f"Unsupported model type: {type(model)}")


class SelectiveInferWrapper:
    """Wrapper class encapsulating a trained checkpoint and selective decision tiers."""

    def __init__(
        self,
        model_or_checkpoint: Any,
        action_threshold: float = DEFAULT_ACTION_THRESHOLD,
        high_threshold: float = DEFAULT_HIGH_THRESHOLD,
        tracker: Any = None,
    ):
        self.model_or_checkpoint = model_or_checkpoint
        self.action_threshold = action_threshold
        self.high_threshold = high_threshold
        self.tracker = tracker

    def predict(self, match_input: dict[str, Any]) -> SelectiveInferenceResult:
        """Predict outcome and conviction tier for a single match."""
        return predict_selective(
            self.model_or_checkpoint,
            match_input,
            action_threshold=self.action_threshold,
            high_threshold=self.high_threshold,
            tracker=self.tracker,
        )

    def predict_batch(self, matches: Sequence[dict[str, Any]]) -> list[SelectiveInferenceResult]:
        """Predict outcomes and conviction tiers for a batch of matches."""
        return batch_predict_selective(
            self.model_or_checkpoint,
            matches,
            action_threshold=self.action_threshold,
            high_threshold=self.high_threshold,
            tracker=self.tracker,
        )

    def summarize_batch(self, matches: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Predict outcomes and return aggregated summary statistics."""
        results = self.predict_batch(matches)
        return format_batch_summary(results)
