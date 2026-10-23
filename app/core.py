import os
from pathlib import Path
import threading
import logging

from core import database
from core import json_utils
from core.feature_labels import VECTOR_DIM
from core.calibration import UncalibratedModelError
from core.pre_match_dataset import to_pre_match_features

logger = logging.getLogger(__name__)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

collection_lock = threading.Lock()
collection_status = {
    "status": "idle",
    "game_name": None,
    "tag_line": None,
    "count": 0,
    "error": None,
}

predict_lock = threading.Lock()

global_trainer = None
tf_available = False
trainer_loading = False

def _bg_load_trainer(db_path=None):
    global global_trainer, tf_available, trainer_loading
    try:
        logger.info("Background thread: starting to load TensorFlow model (this may take a minute)...")
        from scripts.continuous_trainer import ContinuousTrainer
        trainer = ContinuousTrainer(db_path=db_path)
        global_trainer = trainer
        tf_available = True
        logger.info("TensorFlow model loaded successfully in background!")
    except Exception as e:
        logger.error(f"Failed to load TF trainer: {e}", exc_info=True)
    finally:
        trainer_loading = False

def init_trainer_async(db_path=None):
    global trainer_loading
    trainer_loading = True
    t = threading.Thread(target=_bg_load_trainer, args=(db_path,), daemon=True)
    t.start()

def _format_multi_output_response(preds: dict) -> dict:
    from core.training_utils import REGRESSION_STATS

    def _scalar(name, default=0.5):
        if name in preds:
            return float(preds[name][0][0])
        return default

    def _denorm(name):
        mean, std = REGRESSION_STATS[name]
        if name in preds:
            return float(preds[name][0][0]) * std + mean
        return mean

    def _two_class(name, default=(0.5, 0.5)):
        if name in preds:
            return float(preds[name][0][0]), float(preds[name][0][1])
        return default

    def _three_class(name, default=(0.45, 0.45, 0.10)):
        if name in preds:
            return float(preds[name][0][0]), float(preds[name][0][1]), float(preds[name][0][2])
        return default

    p_a, p_b = _two_class("winner")
    pk_a, pk_b = _two_class("team_b_kill_lead", default=(p_a, p_b))

    response = {
        "success": True,
        "is_pre_match": True,
        "winner": {
            "team_a": round(p_a, 3), "team_b": round(p_b, 3),
            "predicted": "B" if p_b > p_a else "A", "confidence": round(max(p_a, p_b), 3),
        },
        "team_b_kill_lead": {
            "team_a": round(pk_a, 3), "team_b": round(pk_b, 3),
            "predicted": "B" if pk_b > pk_a else "A",
        },
        "kills": {
            "total": round(_denorm("total_kills"), 1), "team_a": round(_denorm("team_a_kills"), 1),
            "team_b": round(_denorm("team_b_kills"), 1), "handicap": round(_denorm("kill_handicap"), 1),
            "odd_probability": 0.5,
            "odd_is_coinflip": True,
        },
        "first": {
            event: {"team_a": round(a, 3), "team_b": round(b, 3), "none": round(n, 3)}
            for event, (a, b, n) in (
                ("blood", _three_class("first_blood")), ("baron", _three_class("first_baron")),
                ("inhibitor", _three_class("first_inhibitor")), ("tower", _three_class("first_tower")),
                ("kills_5", _three_class("first_to_5_kills")), ("kills_10", _three_class("first_to_10_kills")),
                ("kills_15", _three_class("first_to_15_kills")), ("kills_20", _three_class("first_to_20_kills")),
            )
        },
        "totals": {
            "barons": round(_denorm("total_barons"), 2), "dragons": round(_denorm("total_dragons"), 2),
            "towers": round(_denorm("total_towers"), 1),
        },
        "both_teams": {
            "baron": round(_scalar("both_baron", 0.0), 3), "inhibitor": round(_scalar("both_inhibitor", 0.0), 3),
            "dragon": round(_scalar("both_dragon", 0.0), 3),
        },
        "elder_dragon": round(_scalar("elder_dragon", 0.0), 3),
    }
    return response

def final_predict_match_outcome(match_data: dict, *, db_path=None, checkpoint_path=None, manifest_path=None) -> dict:
    """All match inference goes through the declared serving facade."""
    from core.prediction_service import get_prediction_service
    from core.pre_match_history import to_serving_features
    from core.serving_contract import ContractError
    try:
        service = get_prediction_service(checkpoint_path=checkpoint_path, db_path=db_path, manifest_path=manifest_path)
    except (ContractError, ValueError, OSError) as exc:
        return {"success": False, "error": str(exc), "error_code": "model_contract", "is_actionable": False}
    try:
        return service.predict_match(to_serving_features(match_data))
    except ContractError as exc:
        return {"success": False, "error": str(exc), "error_code": "feature_contract", "is_actionable": False}
