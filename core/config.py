import os
from pathlib import Path
from dotenv import dotenv_values, load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DB_KEYS = ("NNRIOT_DB_PATH", "NNRIOT_ANALYTICS_DB_PATH", "ANALYTICS_DB_PATH")
_PROCESS_DB_ENV = {key: os.environ[key] for key in _DB_KEYS if key in os.environ}
_IMPORTED_DOTENV = dotenv_values(PROJECT_ROOT / ".env")

# Existing modules consume non-database settings directly from os.environ.
# Preserve that behavior, while keeping database precedence explicit below.
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _configured_path(value, env_keys, default):
    if value is not None:
        if not str(value).strip():
            raise ValueError("Database path must not be empty")
        return str(value)

    dotenv = dotenv_values(PROJECT_ROOT / ".env")
    for key in env_keys:
        candidate = os.environ.get(key)
        if candidate and (
            key in _PROCESS_DB_ENV
            or key not in _IMPORTED_DOTENV
            or candidate != _IMPORTED_DOTENV[key]
        ):
            return candidate
    for key in env_keys:
        if dotenv.get(key):
            return dotenv[key]
    return default


def _absolute_db_path(value, env_keys, default, require_exists):
    selected = Path(_configured_path(value, env_keys, default)).expanduser()
    if not selected.is_absolute():
        selected = PROJECT_ROOT / selected
    resolved = selected.resolve()
    if require_exists and not resolved.is_file():
        raise FileNotFoundError(f"Database file does not exist: {resolved}")
    return str(resolved)


def get_db_path(value=None, *, require_exists=False):
    """Return the source database path, anchored to the project directory."""
    return _absolute_db_path(value, ("NNRIOT_DB_PATH",), "training_data.db", require_exists)


def get_analytics_db_path(value=None, *, require_exists=False):
    """Return the analytics mart path, anchored to the project directory."""
    return _absolute_db_path(
        value,
        ("NNRIOT_ANALYTICS_DB_PATH", "ANALYTICS_DB_PATH"),
        "analytics.db",
        require_exists,
    )


def get_checkpoint_path(value=None, *, require_exists=False):
    """Return the active model checkpoint path, anchored to the project directory."""
    default = "runs/candidate_deepsets_v3/model.keras"
    path = _absolute_db_path(
        value,
        ("NNRIOT_CHECKPOINT_PATH", "MODEL_CHECKPOINT_PATH"),
        default,
        False,
    )
    if require_exists and not Path(path).is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist: {path}")
    return path


def get_model_manifest_path(value=None, *, checkpoint_path=None):
    """Explicit override, otherwise an adjacent manifest; default reference is fixed."""
    configured = value or os.environ.get("NNRIOT_MODEL_MANIFEST_PATH")
    if configured:
        p = Path(configured).expanduser()
        return str((p if p.is_absolute() else PROJECT_ROOT / p).resolve())
    checkpoint = Path(checkpoint_path or get_checkpoint_path())
    sidecar = checkpoint.with_suffix('.manifest.json')
    if sidecar.is_file():
        return str(sidecar.resolve())
    if checkpoint.resolve() == (PROJECT_ROOT / 'runs/candidate_deepsets_v3/model.keras').resolve():
        return str(PROJECT_ROOT / 'docs/model_serving/reference.manifest.json')
    raise ValueError(f'No model manifest for checkpoint: {checkpoint}; set NNRIOT_MODEL_MANIFEST_PATH')


# App Configuration
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 5000))
CORS_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000,http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()
]

# Database Configuration
DB_PATH = get_db_path()
ANALYTICS_DB_PATH = get_analytics_db_path()
MAX_DB_SIZE = int(os.environ.get("NNRIOT_MAX_DB_SIZE_GB", "50")) * 1024 * 1024 * 1024

# Riot API Configuration
RIOT_API_KEY = os.getenv("RIOT_API_KEY")
FETCH_TIMELINES = os.environ.get("NNRIOT_FETCH_TIMELINES", "1") == "1"

# Training Configuration
BATCH_SIZE = int(os.environ.get("NNRIOT_BATCH_SIZE", "3000"))
EPOCHS_PER_BATCH = int(os.environ.get("NNRIOT_EPOCHS", "15"))

# Disable TF oneDNN warnings globally if set
if os.environ.get("TF_ENABLE_ONEDNN_OPTS", "0") == "0":
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
