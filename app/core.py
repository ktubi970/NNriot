"""Module-level state and helper functions for the NNriot Flask app.

This is the single source of truth for:
  - The global ContinuousTrainer (loaded once at import).
  - Locks for prediction and history-collection background jobs.
  - The shared "collection_status" dict updated by collect/history routes.
  - The multi-output prediction response shaper.

Routes import this module by name (``from .. import core``) and read
``core.global_trainer`` / ``core.tf_available`` etc. via attribute access,
so test fixtures can monkeypatch these at runtime.
"""
import os
import threading
import logging

import database
import json_utils
from feature_labels import VECTOR_DIM

logger = logging.getLogger(__name__)

# Project directory (used for checkpoint discovery). Resolves to the
# repo root, so checkpoint glob works regardless of CWD.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# History Collection Background Job State
collection_lock = threading.Lock()
collection_status = {
    "status": "idle",  # idle, collecting, completed, error
    "game_name": None,
    "tag_line": None,
    "count": 0,
    "error": None,
}

# Prediction Threading Lock to prevent Keras concurrent prediction crashes
predict_lock = threading.Lock()


def init_trainer():
    """Initialize the continuous trainer. Returns (trainer, tf_available)."""
    try:
        from continuous_trainer import ContinuousTrainer

        trainer = ContinuousTrainer()
        logger.info("TensorFlow model loaded successfully.")
        return trainer, True
    except Exception as e:
        logger.error(f"Failed to load TF trainer: {e}", exc_info=True)
        return None, False


# Module-level trainer — initialized once at import.
global_trainer, tf_available = init_trainer()


def _format_multi_output_response(preds: dict) -> dict:
    """
    Convert raw Keras dict-output (shape (1, k) per head) into the
    public API response shape.

    Regression heads are trained on normalized targets (see
    continuous_trainer.REGRESSION_STATS); _denorm restores them to
    original units for the public API.
    """
    # Lazy import: avoids importing TF-heavy continuous_trainer at module
    # top-level (init_trainer() already handles the TF-unavailable path).
    from continuous_trainer import REGRESSION_STATS

    def _scalar(name):
        """Extract the single scalar from a (1, 1) regression/sigmoid head."""
        return float(preds[name][0][0])

    def _denorm(name):
        """Denormalize a regression head's output back to original scale."""
        mean, std = REGRESSION_STATS[name]
        return _scalar(name) * std + mean

    def _two_class(name):
        """Extract (p_team_a, p_team_b) from a (1, 2) softmax head."""
        return float(preds[name][0][0]), float(preds[name][0][1])

    def _three_class(name):
        """Extract (p_team_a, p_team_b, p_none) from a (1, 3) softmax head."""
        return float(preds[name][0][0]), float(preds[name][0][1]), float(preds[name][0][2])

    p_a, p_b = _two_class("winner")
    pk_a, pk_b = _two_class("team_b_kill_lead")

    response = {
        "success": True,
        "winner": {
            "team_a": round(p_a, 3),
            "team_b": round(p_b, 3),
            "predicted": "B" if p_b > p_a else "A",
            "confidence": round(max(p_a, p_b), 3),
        },
        "team_b_kill_lead": {
            "team_a": round(pk_a, 3),
            "team_b": round(pk_b, 3),
            "predicted": "B" if pk_b > pk_a else "A",
        },
        "kills": {
            "total":           round(_denorm("total_kills"), 1),
            "team_a":          round(_denorm("team_a_kills"), 1),
            "team_b":          round(_denorm("team_b_kills"), 1),
            "handicap":        round(_denorm("kill_handicap"), 1),
            "odd_probability": round(_scalar("kills_odd"), 3),
        },
        "first": {
            event: {
                "team_a": round(a, 3),
                "team_b": round(b, 3),
                "none":   round(n, 3),
            }
            for event, (a, b, n) in (
                ("blood",     _three_class("first_blood")),
                ("baron",     _three_class("first_baron")),
                ("inhibitor", _three_class("first_inhibitor")),
                ("tower",     _three_class("first_tower")),
                # Timeline kills thresholds — additive keys, backward-compatible
                ("kills_5",   _three_class("first_to_5_kills")),
                ("kills_10",  _three_class("first_to_10_kills")),
                ("kills_15",  _three_class("first_to_15_kills")),
                ("kills_20",  _three_class("first_to_20_kills")),
            )
        },
        "totals": {
            "barons":  round(_denorm("total_barons"), 2),
            "dragons": round(_denorm("total_dragons"), 2),
            "towers":  round(_denorm("total_towers"), 1),
        },
        "both_teams": {
            "baron":     round(_scalar("both_baron"), 3),
            "inhibitor": round(_scalar("both_inhibitor"), 3),
            "dragon":    round(_scalar("both_dragon"), 3),
        },
        "elder_dragon": round(_scalar("elder_dragon"), 3),
    }

    return response


def final_predict_match_outcome(match_data: dict) -> dict:
    """
    Multi-output prediction. Returns a dict with named keys: winner,
    team_b_kill_lead, kills, first.*, totals.*, both_teams.*, elder_dragon.
    """
    try:
        if not tf_available or global_trainer is None:
            return {"error": "TensorFlow model is not loaded or available"}

        feature_dict = json_utils.extract_match_features(match_data)
        sparse_vec = json_utils.json_to_vector([feature_dict], dim=VECTOR_DIM)
        dense_vec = sparse_vec.toarray()

        with predict_lock:
            preds = global_trainer.predict(dense_vec)
        # preds is a dict: {head_name: ndarray(1, k)}

        return _format_multi_output_response(preds)
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return {"error": str(e)}
