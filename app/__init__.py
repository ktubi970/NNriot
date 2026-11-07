"""Flask application factory and module-level app instance."""
import logging
import os
from flask import Flask
from flask_cors import CORS
from werkzeug.local import LocalProxy

from core import config, database


def create_app(overrides=None):
    """Build an app bound to one resolved source database."""
    logging.basicConfig(level=logging.INFO)
    app = Flask(
        __name__,
        static_folder="../static",
        template_folder="../templates",
    )
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
    if overrides:
        app.config.update(overrides)

    source = config.get_db_path(
        app.config.get("NNRIOT_DB_PATH"),
        require_exists=not app.config.get("TESTING", False),
    )
    analytics = config.get_analytics_db_path(app.config.get("ANALYTICS_DB_PATH"))
    app.config["NNRIOT_DB_PATH"] = source
    app.config["ANALYTICS_DB_PATH"] = analytics

    try:
        database.init_db(db_path=source)
    except Exception:
        logging.getLogger(__name__).error("Database initialization failed", exc_info=True)
        raise

    allowed_origins = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000"
        ).split(",")
        if origin.strip()
    ]
    CORS(app, origins=allowed_origins)

    if not app.config.get("TESTING"):
        from . import core
        core.init_trainer_async(db_path=source)

    from .routes.views import views_bp
    from .routes.api import api_bp
    from .routes.history import history_bp
    from .routes.lolpros import lolpros_bp
    from .routes.analytics import analytics_bp

    app.register_blueprint(views_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(lolpros_bp)
    app.register_blueprint(analytics_bp)
    return app


_default_app = None


def _get_default_app():
    global _default_app
    if _default_app is None:
        _default_app = create_app()
    return _default_app


# WSGI compatibility without opening a database merely by importing the factory.
app = LocalProxy(_get_default_app)
