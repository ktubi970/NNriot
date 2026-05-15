"""Flask application factory and module-level app instance.

The factory wires the blueprints together; the module-level ``app =
create_app()`` keeps ``from app import app`` working for tests and WSGI
servers that expect a top-level WSGI application.
"""
import os
import logging
from flask import Flask
from flask_cors import CORS

import database


def create_app():
    """Application factory."""
    logging.basicConfig(level=logging.INFO)

    # Ensure DB exists and is at the latest schema. Idempotent — no-op if
    # already migrated. Critical for fresh deployments with no
    # training_data.db.
    try:
        database.init_db()
    except Exception:
        logging.getLogger(__name__).error(
            "Database initialization failed", exc_info=True
        )
        raise

    app = Flask(
        __name__,
        static_folder="../static",
        template_folder="../templates",
    )
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB

    # CORS allowlist — match the previous final_web_app.py defaults.
    allowed_origins = [
        o.strip()
        for o in os.getenv(
            "CORS_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000"
        ).split(",")
        if o.strip()
    ]
    CORS(app, origins=allowed_origins)

    # Import core at app creation to ensure the trainer is loaded eagerly
    # (matches previous final_web_app.py behavior: trainer ready by the
    # time the first request comes in).
    from . import core  # noqa: F401

    # Register blueprints
    from .routes.views import views_bp
    from .routes.api import api_bp
    from .routes.history import history_bp
    from .routes.lolpros import lolpros_bp

    app.register_blueprint(views_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(lolpros_bp)

    return app


# Module-level app for tests / WSGI servers.
app = create_app()
