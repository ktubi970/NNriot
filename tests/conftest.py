"""Keep test collection, app startup, and default model writes off live data."""
from pathlib import Path
import gc
import os
import tempfile

# This must run before test modules import config, repositories, or the app.
_state = tempfile.TemporaryDirectory(prefix="nnriot-pytest-")
_state_path = Path(_state.name).resolve()
_keys = ("NNRIOT_DB_PATH", "NNRIOT_ANALYTICS_DB_PATH", "NNRIOT_CHECKPOINT_PATH")
_original_env = {key: os.environ.get(key) for key in _keys}
os.environ.update({
    "NNRIOT_DB_PATH": str(_state_path / "source.db"),
    "NNRIOT_ANALYTICS_DB_PATH": str(_state_path / "analytics.db"),
    "NNRIOT_CHECKPOINT_PATH": str(_state_path / "model.keras"),
})
from core import database

database.init_db(os.environ["NNRIOT_DB_PATH"])
import app

# Existing endpoint tests import this compatibility entry point. No trainer
# thread should start before their mocks and isolated fixtures are installed.
app.app = app.create_app({"TESTING": True, "NNRIOT_DB_PATH": os.environ["NNRIOT_DB_PATH"]})


def pytest_unconfigure(config):
    gc.collect()
    for key, value in _original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    _state.cleanup()
