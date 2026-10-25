import os
import sqlite3
import threading
import logging

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

# Facade failures the client can fix; every other failure is a service fault.
_CLIENT_ERROR_CODES = frozenset({"feature_contract"})


def get_prediction_service(**kwargs):
    """Load TensorFlow only when the serving facade is actually requested."""
    from core.prediction_service import get_prediction_service as get_service
    return get_service(**kwargs)


def prediction_http_status(result: dict) -> int:
    """400 for inputs the client can fix; 503 when model, history or output is unavailable."""
    return 400 if result.get("error_code") in _CLIENT_ERROR_CODES else 503


def _prediction_failure(error_code: str, exc: Exception) -> dict:
    return {"success": False, "error": str(exc), "error_code": error_code, "is_actionable": False}


def final_predict_match_outcome(match_data: dict, *, db_path=None, checkpoint_path=None, manifest_path=None) -> dict:
    """All match inference goes through the declared serving facade."""
    from core.pre_match_history import to_serving_features
    from core.serving_contract import ContractError, FeatureUnavailableError
    from core.serving_guards import ModelOutputError, validate_participant_teams
    try:
        service = get_prediction_service(checkpoint_path=checkpoint_path, db_path=db_path, manifest_path=manifest_path)
    except (ContractError, ValueError, OSError, ImportError) as exc:
        return _prediction_failure("model_contract", exc)
    try:
        validate_participant_teams(match_data)
        return service.predict_match(to_serving_features(match_data))
    except FeatureUnavailableError as exc:
        return _prediction_failure("feature_unavailable", exc)
    except ModelOutputError as exc:
        return _prediction_failure("model_output", exc)
    except ContractError as exc:
        return _prediction_failure("feature_contract", exc)
    except sqlite3.Error as exc:
        return _prediction_failure("feature_unavailable", exc)
