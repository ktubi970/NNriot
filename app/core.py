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

def final_predict_match_outcome(match_data: dict) -> dict:
    # 1. Check if a modern candidate/active model is available via PredictionService Facade
    try:
        from core import config
        from core.prediction_service import get_prediction_service
        chk_path = config.get_checkpoint_path(require_exists=False)
        if chk_path and Path(chk_path).exists():
            service = get_prediction_service(checkpoint_path=chk_path)
            if service.is_ready():
                feature_dict = to_pre_match_features(match_data)
                return service.predict_match(feature_dict)
    except Exception as e:
        logger.warning("PredictionService attempt encountered error, falling back to legacy trainer: %s", e)

    # 2. Legacy trainer fallback
    if global_trainer is None:
        if trainer_loading:
            return {"error": "TensorFlow model is still loading... Please wait a minute and try again."}
        return {"error": "TensorFlow model is not loaded or available"}

    trainer_model = getattr(global_trainer, "model", None)
    is_calibrated = False
    if trainer_model is not None:
        is_calibrated = getattr(trainer_model, "is_calibrated", False) is True
    elif getattr(global_trainer, "is_calibrated", False) is True:
        is_calibrated = True

    if not is_calibrated:
        return {
            "error": "No calibrated active model is available. Candidate training does not automatically activate a model; validate and promote a checkpoint before requesting predictions.",
            "uncalibrated": True,
        }

    try:
        feature_dict = to_pre_match_features(match_data)
        raw_shape = getattr(trainer_model, "input_shape", None)
        if isinstance(raw_shape, (tuple, list)) and len(raw_shape) > 0 and isinstance(raw_shape[-1], int):
            expected_dim = raw_shape[-1]
        else:
            expected_dim = VECTOR_DIM

        from core.draft_features import DENSE_FEATURE_DIM, extract_dense_draft_features
        from core.enriched_features import ENRICHED_DENSE_DIM, extract_enriched_features

        if expected_dim == ENRICHED_DENSE_DIM:
            dense_vec = extract_enriched_features(feature_dict, mode="dense").reshape(1, -1)
        elif expected_dim == DENSE_FEATURE_DIM:
            dense_vec = extract_dense_draft_features(feature_dict).reshape(1, -1)
        else:
            sparse_vec = json_utils.json_to_vector([feature_dict], dim=expected_dim)
            dense_vec = sparse_vec.toarray()

        with predict_lock:
            preds = global_trainer.predict(dense_vec)
        return _format_multi_output_response(preds)
    except UncalibratedModelError as exc:
        return {"error": str(exc), "uncalibrated": True}
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return {"error": str(e)}
