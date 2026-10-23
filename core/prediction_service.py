"""Manifest-bound Prediction Service Facade for NNriot.

Conforms to pattern-design Facade pattern:
- Orchestrates feature extraction (DraftFeaturePipeline)
- Handles model loading and neural inference with declared feature semantics
- Applies probability calibration and Chow selective abstention
- Formats responses with explicit calibration and validation metadata
"""
from __future__ import annotations

import json
import logging
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence
import numpy as np

from core.features.pipeline import DraftFeaturePipeline, get_feature_pipeline
from core.aligned_features import extract_aligned_inputs
from core.features.matchup_matrix import LaneMatchupTracker
from core.selective_infer import (
    classify_match_conviction,
    SelectiveInferenceResult,
    DEFAULT_ACTION_THRESHOLD,
    DEFAULT_HIGH_THRESHOLD,
)
from core.calibration import apply_temperature_to_probabilities
from core.deepsets_layers import SiameseTeamEncoder, AntiSymmetricSoftmax
from core.equivariant_model import ThreeClassEquivariantSoftmax, NNRIOT_CUSTOM_OBJECTS
from core.shapley_credit import compute_counterfactual_shapley
from core.serving_contract import (
    FeatureContract, ContractError, HISTORY_WINDOW_LIMIT, load_model_manifest,
    validate_probabilities, serving_decision_policy,
)

logger = logging.getLogger("prediction_service")


class PredictionService:
    """Thread-safe Facade for match outcome prediction and selective decision gating."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        feature_pipeline: DraftFeaturePipeline | None = None,
        action_threshold: float = DEFAULT_ACTION_THRESHOLD,
        high_threshold: float = DEFAULT_HIGH_THRESHOLD,
        manifest_path: str | Path | None = None,
        history_provider=None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.feature_pipeline = feature_pipeline or DraftFeaturePipeline(matchup_tracker=LaneMatchupTracker())
        self.action_threshold = float(action_threshold)
        self.high_threshold = float(high_threshold)
        if not (0 <= self.action_threshold <= self.high_threshold <= 0.5):
            raise ContractError('Thresholds must satisfy 0 <= action <= high <= 0.5')
        self.contract = FeatureContract('nnriot.players14.v1')
        self.manifest = None
        self.manifest_path = manifest_path
        self.history_provider = history_provider

        self._lock = threading.Lock()
        self._model = None
        self._lgb_model = None
        self._temperature: float = 1.0
        self._is_calibrated: bool = False
        self._verified_calibration = None
        self._verified_manifest = None
        self._verified_decision_policy = None
        self._blend_weights: dict[str, float] = {"nn": 1.0, "lgb": 0.0}

        if self.checkpoint_path:
            self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load the manifest-bound neural model and verified winner calibration."""
        if not self.checkpoint_path:
            return

        from core.config import get_model_manifest_path
        if not self.manifest_path:
            self.manifest_path = get_model_manifest_path(checkpoint_path=self.checkpoint_path)
        self.manifest = load_model_manifest(self.manifest_path, checkpoint_path=self.checkpoint_path)
        self.contract = FeatureContract(self.manifest['feature_contract'])
        if self.contract.dim == 18 and self.history_provider is None:
            raise ContractError('18D serving requires an explicit pre-match timeline history provider')
        import tensorflow as tf

        with self._lock:
            try:
                logger.info("Loading neural checkpoint from %s...", self.checkpoint_path)
                model = tf.keras.models.load_model(self.checkpoint_path, custom_objects=NNRIOT_CUSTOM_OBJECTS, compile=False)
                self.contract.validate_model(model)
                if not isinstance(model.output, dict) or 'winner' not in model.output or tuple(model.output['winner'].shape)[1:] != (2,):
                    raise ContractError('Checkpoint must expose a two-class winner head')
                if set(self.manifest.get('output_heads', [])) != set(model.output):
                    raise ContractError('Manifest output heads disagree with checkpoint')
                self._model = model
            except Exception as e:
                raise ContractError(f'Cannot load checkpoint {self.checkpoint_path}: {e}') from e

            # Load temperature calibration
            self._temperature = self.manifest.get('_temperature', 1.0)
            self._is_calibrated = '_temperature' in self.manifest
            self._verified_calibration = (model, self._temperature) if self._is_calibrated else None
            self._verified_manifest = deepcopy(self.manifest)
            self._verified_decision_policy = deepcopy(self.manifest.get('_verified_decision_policy'))

    def is_ready(self) -> bool:
        """Check if service has a loaded model ready for inference."""
        return self._model is not None

    def verification_status(self, action_threshold: float | None = None, *, priors_provenance: dict | None = None) -> dict:
        """Use the same artifact and configuration gates for inference and status."""
        tau = self.action_threshold if action_threshold is None else float(action_threshold)
        policy = serving_decision_policy(tau, self.high_threshold)
        calibrated = (
            self._is_calibrated and self._verified_calibration is not None
            and self._model is self._verified_calibration[0]
            and self._temperature == self._verified_calibration[1]
        )
        # Feature contracts v1 fix the global history window for serving/evaluation.
        history_limit = getattr(self.history_provider, 'history_limit', None)
        history_compatible = self.history_provider is None or (
            type(history_limit) is int and history_limit == HISTORY_WINDOW_LIMIT
        )
        if priors_provenance is not None and 'history_window_limit' in priors_provenance:
            supplied_limit = priors_provenance['history_window_limit']
            history_compatible = history_compatible and (
                type(supplied_limit) is int and supplied_limit == HISTORY_WINDOW_LIMIT
            )
        validated = (
            calibrated and self.manifest is not None and self.manifest == self._verified_manifest
            and self._verified_decision_policy == policy and history_compatible
        )
        validation = (
            deepcopy(self.manifest['serving_validation']) if validated else {
                'status': 'unverified' if self.manifest else 'in_memory_unverified',
                'reason': 'independent evidence unavailable or effective configuration changed',
            }
        )
        return {'calibrated': calibrated, 'validated': validated,
                'serving_validation': validation, 'decision_policy': policy}

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

        # 1. Feature extraction via Pipeline Strategy
        if self.history_provider is not None and 'priors_provenance' not in match_dict:
            match_dict = self.history_provider.enrich(match_dict, include_timeline=self.contract.dim == 18)
        verification = self.verification_status(
            action_threshold, priors_provenance=match_dict.get('priors_provenance'),
        )
        tau = verification['decision_policy']['action_threshold']
        calibrated = verification['calibrated']
        validated = verification['validated']
        input_names = set(getattr(self._model, "input_names", None) or [])
        input_names.update(t.name.split(":")[0] for t in getattr(self._model, "inputs", []) or [])
        x_nn = extract_aligned_inputs(match_dict, self.contract.version,
                                      pipeline=self.feature_pipeline, macro='macro_clash' in input_names)

        # 2. Neural model prediction
        with self._lock:
            preds = self._model.predict(x_nn, verbose=0)
            if not isinstance(preds, dict) or 'winner' not in preds:
                raise ContractError('Inference returned no winner head')
            validate_probabilities(preds['winner'], 'winner')
            shapley_credits = compute_counterfactual_shapley(
                self._model, x_nn["team_a"], x_nn["team_b"],
            )

        if not isinstance(preds, dict) or 'winner' not in preds:
            raise ContractError('Inference returned no winner head')
        validate_probabilities(preds['winner'], 'winner')
        raw_uncertainty = preds.get("winner_uncertainty")
        epistemic_uncertainty = (
            float(np.asarray(raw_uncertainty).reshape(-1)[0])
            if raw_uncertainty is not None else None
        )

        raw_winner = preds.get("winner")
        p_b_raw = float(raw_winner[0, 1])

        raw_draft = preds.get("draft_advantage")
        if raw_draft is not None and len(raw_draft.shape) >= 2:
            validate_probabilities(raw_draft, 'draft_advantage')
            p_draft_a = float(raw_draft[0, 0])
            p_draft_b = float(raw_draft[0, 1])
            favored = (
                "team_a"
                if p_draft_a > p_draft_b + 0.01
                else ("team_b" if p_draft_b > p_draft_a + 0.01 else "even")
            )
            draft_advantage = {
                "team_a": round(p_draft_a, 4),
                "team_b": round(p_draft_b, 4),
                "favored": favored,
            }
        else:
            draft_advantage = None

        # 3. Apply calibration
        if calibrated:
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
        if epistemic_uncertainty is not None and (not np.isfinite(epistemic_uncertainty) or not 0 <= epistemic_uncertainty <= 1):
            raise ContractError('winner_uncertainty must be a finite value in [0, 1]')
        if not calibrated:
            sel = sel._replace(prediction='abstain', tier='UNCALIBRATED', is_actionable=False)
        elif not validated:
            sel = sel._replace(prediction='abstain', tier='UNVALIDATED_MODEL', is_actionable=False)

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
            "model": {
                "id": self.manifest['model_id'] if self.manifest else 'in_memory_unverified',
                "checkpoint_sha256": self.manifest['checkpoint']['sha256'] if self.manifest else None,
                "feature_contract": self.contract.version,
                "legacy_architecture": 'draft_advantage' not in preds,
                "training_provenance_status": self.manifest.get('training', {}).get('status', 'unknown') if self.manifest else 'unknown',
                "serving_validation": verification['serving_validation'],
            },
            "probabilities_raw": {"team_a": 1.0 - p_b_raw, "team_b": p_b_raw},
            "probabilities_calibrated": {"team_a": p_a, "team_b": p_b} if calibrated else None,
            "calibration": {"status": 'artifact_verified' if calibrated else 'unavailable', "temperature": self._temperature if calibrated else None},
            "priors_provenance": match_dict.get('priors_provenance', {'policy': 'declared_cold_start_defaults', 'history_available': False}),
            "winner": {
                "team_a": round(p_a, 4),
                "team_b": round(p_b, 4),
                "predicted": sel.prediction,
                "confidence": round(max(p_a, p_b), 4),
            },
            "draft_advantage": draft_advantage,
            "heads_raw": {k: np.asarray(v)[0].tolist() for k, v in preds.items()},
            "selective_decision": {
                "conviction": sel.conviction,
                "tier": sel.tier,
                "is_actionable": sel.is_actionable,
                "threshold_applied": tau,
                "high_threshold": self.high_threshold,
                "abstained": not sel.is_actionable,
                "recommendation": (
                    f"Bet on {sel.prediction.upper()}" if sel.is_actionable
                    else {
                        "UNCALIBRATED": "Abstain - winner calibration unavailable",
                        "UNVALIDATED_MODEL": "Abstain - independent serving validation unavailable",
                    }.get(sel.tier, "Abstain - conviction below threshold")
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


def get_prediction_service(checkpoint_path: str | Path | None = None, *, db_path=None, manifest_path=None) -> PredictionService:
    """Return shared global PredictionService singleton."""
    global _GLOBAL_SERVICE
    from core import config
    checkpoint_path = Path(config.get_checkpoint_path(checkpoint_path, require_exists=True))
    manifest_path = Path(config.get_model_manifest_path(manifest_path, checkpoint_path=checkpoint_path))
    identity = (str(checkpoint_path), str(manifest_path), str(db_path))
    if _GLOBAL_SERVICE is None or getattr(_GLOBAL_SERVICE, '_identity', None) != identity:
        manifest = load_model_manifest(manifest_path, checkpoint_path=checkpoint_path)
        provider = None
        if db_path is not None or manifest['feature_contract'] == 'nnriot.players18.v1':
            from core.pre_match_history import DatabaseHistoryProvider
            provider = DatabaseHistoryProvider(config.get_db_path(db_path, require_exists=True))
        _GLOBAL_SERVICE = PredictionService(checkpoint_path=checkpoint_path, manifest_path=manifest_path, history_provider=provider)
        _GLOBAL_SERVICE._identity = identity
    return _GLOBAL_SERVICE
