import os
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# App Configuration
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 5000))
CORS_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000,http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()
]

# Database Configuration
DB_PATH = os.environ.get("NNRIOT_DB_PATH", "training_data.db")
ANALYTICS_DB_PATH = os.environ.get("ANALYTICS_DB_PATH", "analytics/data/analytics.db")
MAX_DB_SIZE = int(os.environ.get("NNRIOT_MAX_DB_SIZE_GB", "50")) * 1024 * 1024 * 1024

# Riot API Configuration
RIOT_API_KEY = os.getenv("RIOT_API_KEY")
FETCH_TIMELINES = os.environ.get("NNRIOT_FETCH_TIMELINES", "0") == "1"

# Training Configuration
BATCH_SIZE = int(os.environ.get("NNRIOT_BATCH_SIZE", "3000"))
EPOCHS_PER_BATCH = int(os.environ.get("NNRIOT_EPOCHS", "15"))

# Disable TF oneDNN warnings globally if set
if os.environ.get("TF_ENABLE_ONEDNN_OPTS", "0") == "0":
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
