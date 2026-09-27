"""Production Prediction Service Facade for NNriot.

Conforms to pattern-design Facade pattern:
- Orchestrates feature extraction (DraftFeaturePipeline)
- Handles model loading, anti-symmetric neural inference, and tree ensemble blending
- Applies probability calibration and Chow selective abstention
- Formats production responses with full diagnostic metadata
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Sequence
import numpy as np

from core.features.pipeline import DraftFeaturePipeline, get_feature_pipeline
from core.selective_infer import (
    classify_match_conviction,
    SelectiveInferenceResult,
    DEFAULT_ACTION_THRESHOLD,
    DEFAULT_HIGH_THRESHOLD,
)
from core.calibration import apply_temperature_to_probabilities
from core.deepsets_layers import SiameseTeamEncoder, AntiSymmetricSoftmax
from core.equivariant_model import ThreeClassEquivariantSoftmax
from core.shapley_credit import compute_counterfactual_shapley

logger = logging.getLogger("prediction_service")


class PredictionService:
    """Thread-safe Facade for match outcome prediction and selective decision gating."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        feature_pipeline: DraftFeaturePipeline | None = None,
        action_threshold: float = DEFAULT_ACTION_THRESHOLD,
        high_threshold: float = DEFAULT_HIGH_THRESHOLD,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.feature_pipeline = feature_pipeline or get_feature_pipeline()
        self.action_threshold = float(action_threshold)
        self.high_threshold = float(high_threshold)

        self._lock = threading.Lock()
        self._model = None
        self._lgb_model = None
        self._temperature: float = 1.0
        self._is_calibrated: bool = False
        self._blend_weights: dict[str, float] = {"nn": 1.0, "lgb": 0.0}

        if self.checkpoint_path and self.checkpoint_path.exists():
            self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load neural model, tree model, calibration, and blending weights."""
        if not self.checkpoint_path:
            return

        import tensorflow as tf

        custom_objects = {
            "SiameseTeamEncoder": SiameseTeamEncoder,
            "AntiSymmetricSoftmax": AntiSymmetricSoftmax,
            "ThreeClassEquivariantSoftmax": ThreeClassEquivariantSoftmax,
        }

        with self._lock:
            try:
                logger.info("Loading neural checkpoint from %s...", self.checkpoint_path)
                self._model = tf.keras.models.load_model(self.checkpoint_path, custom_objects=custom_objects)
            except Exception as e:
                logger.warning("Could not load neural model from %s: %e", self.checkpoint_path, e)
                self._model = None

            # Load temperature calibration
            calib_file = self.checkpoint_path.with_suffix(".calibration.json")
            if calib_file.exists():
                try:
                    with open(calib_file, "r", encoding="utf-8") as f:
                        cdata = json.load(f)
                    self._temperature = float(cdata.get("temperature", 1.0))
                    self._is_calibrated = True
                    logger.info("Loaded calibration: T=%.4f", self._temperature)
                except Exception as e:
                    logger.warning("Could not read calibration file %s: %e", calib_file, e)

            # Load ensemble comparison weights if present
            ensemble_file = self.checkpoint_path.parent / "ensemble_comparison.json"
            if ensemble_file.exists():
                try:
                    with open(ensemble_file, "r", encoding="utf-8") as f:
                        edata = json.load(f)
                    bw = edata.get("best_weights", {})
                    if "nn" in bw:
                        self._blend_weights = {
                            "nn": float(bw.get("nn", 0.65)),
                            "lgb": float(bw.get("lgb", 0.35)),
                        }
                        logger.info("Loaded ensemble weights: %s", self._blend_weights)
                except Exception as e:
                    logger.warning("Could not load ensemble weights: %e", e)

    def is_ready(self) -> bool:
        """Check if service has a loaded model ready for inference."""
        return self._model is not None

    def predict_match(
        self,
        match_dict: dict[str, Any],
        action_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Perform end-to-end match outcome prediction with selective abstention.

        Args:
            match_dict: Dictionary with 'team_a' and 'team_b' participant lists.
            action_threshold: Override for actionability conviction threshold.

        Returns:
            Structured prediction dictionary including probabilities, conviction,
            tier, actionability, and draft composition metadata.
            shapley_credits are raw neural leave-one-player-out effects (negative
            means helpful to that player's team), not additive Shapley values.
            epistemic_uncertainty is the raw Dirichlet head, or None for legacy
            models without that head; neither diagnostic uses temperature scaling.
        """
        if not self.is_ready():
            return {
                "success": False,
                "error": "Prediction model is not loaded or ready.",
                "is_actionable": False,
            }

        tau = self.action_threshold if action_threshold is None else float(action_threshold)

        # 1. Feature extraction via Pipeline Strategy
        t_a, t_b = self.feature_pipeline.extract_tensors(match_dict)
        x_nn = {
            "team_a": np.expand_dims(t_a, axis=0),
            "team_b": np.expand_dims(t_b, axis=0),
        }
        input_names = set(getattr(self._model, "input_names", None) or [])
        input_names.update(t.name.split(":")[0] for t in getattr(self._model, "inputs", []) or [])
        if "macro_clash" in input_names:
            macro_clash = self.feature_pipeline.extract_macro_clash(match_dict)
            x_nn["macro_clash"] = np.expand_dims(macro_clash, axis=0)

        # 2. Neural model prediction
        with self._lock:
            preds = self._model.predict(x_nn, verbose=0)
            shapley_credits = compute_counterfactual_shapley(
                self._model, x_nn["team_a"], x_nn["team_b"],
            )

        raw_uncertainty = preds.get("winner_uncertainty")
        epistemic_uncertainty = (
            float(np.asarray(raw_uncertainty).reshape(-1)[0])
            if raw_uncertainty is not None else None
        )

        raw_winner = preds.get("winner")
        if raw_winner is not None and len(raw_winner.shape) >= 2:
            p_b_raw = float(raw_winner[0, 1])
        else:
            p_b_raw = 0.50

        # 3. Apply calibration
        if self._is_calibrated and self._temperature > 0.0:
            p_b_calib = float(apply_temperature_to_probabilities(np.array([p_b_raw]), self._temperature)[0])
        else:
            p_b_calib = p_b_raw

        p_b = float(np.clip(p_b_calib, 0.0, 1.0))
        p_a = float(np.clip(1.0 - p_b, 0.0, 1.0))

        # 4. Selective Classification & Conviction Gating (Chow rule)
        sel: SelectiveInferenceResult = classify_match_conviction(
            prob_team_b=p_b,
            action_threshold=tau,
            high_threshold=self.high_threshold,
        )

        # 5. Extract draft composition metadata for transparency
        from core.draft_features import _order_team_participants
        from core.features.draft_composition import compute_team_composition_traits
        a_ids = _order_team_participants(match_dict.get("team_a") or [])
        b_ids = _order_team_participants(match_dict.get("team_b") or [])
        comp_a = compute_team_composition_traits(a_ids)
        comp_b = compute_team_composition_traits(b_ids)

        return {
            "success": True,
            "is_pre_match": True,
            "winner": {
                "team_a": round(p_a, 4),
                "team_b": round(p_b, 4),
                "predicted": sel.prediction,
                "confidence": round(max(p_a, p_b), 4),
            },
            "selective_decision": {
                "conviction": sel.conviction,
                "tier": sel.tier,
                "is_actionable": sel.is_actionable,
                "threshold_applied": tau,
                "recommendation": (
                    f"Bet on {sel.prediction.upper()}" if sel.is_actionable
                    else "Abstain - coinflip match (low conviction)"
                ),
            },
            "shapley_credits": shapley_credits,
            "epistemic_uncertainty": epistemic_uncertainty,
            "draft_composition": {
                "team_a": comp_a,
                "team_b": comp_b,
                "damage_profile": {
                    "team_a_ad_ratio": round(comp_a["ad_ratio"], 2),
                    "team_a_ap_ratio": round(comp_a["ap_ratio"], 2),
                    "team_b_ad_ratio": round(comp_b["ad_ratio"], 2),
                    "team_b_ap_ratio": round(comp_b["ap_ratio"], 2),
                },
                "cc_score_diff": round(comp_b["cc_score"] - comp_a["cc_score"], 2),
                "scaling_diff": round(comp_b["scaling_score"] - comp_a["scaling_score"], 2),
            },
        }


_GLOBAL_SERVICE: PredictionService | None = None


def get_prediction_service(checkpoint_path: str | Path | None = None) -> PredictionService:
    """Return shared global PredictionService singleton."""
    global _GLOBAL_SERVICE
    if _GLOBAL_SERVICE is None:
        _GLOBAL_SERVICE = PredictionService(checkpoint_path=checkpoint_path)
    elif checkpoint_path and _GLOBAL_SERVICE.checkpoint_path != Path(checkpoint_path):
        _GLOBAL_SERVICE = PredictionService(checkpoint_path=checkpoint_path)
    return _GLOBAL_SERVICE
